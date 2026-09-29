"""Bit-packed NSNQuant window representation.

This module separates the reference round trip into storage and reconstruction:

    encode_window_k(...) -> PackedWindow      # what a real deployment would store
    decode_window_k(PackedWindow) -> bf16     # what attention reads

``decode(encode(x))`` matches ``_transform_window_{k,v}(x)``. The stored fields are:

  * `idx` (uint8) indexes the same codebook, so `codebook[idx]` is the original `vq_val`;
  * `norm`/`mean` are stored as the 4-bit RTN *codes* plus their (scale, min), and
    `q * scale + min` recomputed in the original dtype reproduces `_rtn4_roundtrip`'s output
    bit-for-bit -- integers 0..15 are exact in bf16/fp16, so nothing is lost in the uint8 hop;
  * `norm2` is stored as fp16 -- `nsn_quant._scale_adjust` now rounds to exactly that precision
    (`NORM2_STORE_DTYPE`), so the value stored here is the value it computed. It stays 16-bit
    rather than 4-bit because `_scale_adjust` multiplies by num/den *after* the RTN round-trip,
    leaving the value off the 4-bit grid.

Bit accounting (head_dim=128, ws=64, G=8, K=256, bf16 metadata), per token per head:

    idx     16 groups x 8 bits                        = 128.0 bits
    norm2   1 x 16 bits                                =  16.0
    norm    4 bits + (scale+min = 32 bits) / ws        =   4.5
    mean    (128 x 4 bits + 4 groups x 32 bits) / ws   =  10.0
                                                        ------
                                                         158.5 bits / 128 channels
                                                       = 1.238 bit/channel   (bf16 cache = 16.0)

``bits_per_channel()`` computes the rate from the stored tensor shapes. ``norm2`` uses fp16 to
preserve mantissa resolution while keeping metadata at 16 bits.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch

from sglang.srt.mem_cache.nsn_quant import (
    _apply_rope,
    _hadamard,
    _scale_adjust,
    _vq_nearest,
)

RTN4_LEVELS = 15.0  # asymmetric 4-bit: 16 levels, 15 intervals (mirrors _rtn4_roundtrip)


@dataclass
class PackedWindow:
    """One (layer, request, head-group) window in its stored form.

    Shapes for a [H, ws, D] window: idx [H, ws, D//8] uint8; norm2 [H, ws, 1] dtype;
    norm_q [H, ws] uint8 with norm_scale/norm_min [H, 1]; mean_q [H, 1, D] uint8 with
    mean_scale/mean_min [H, D//mean_group, 1]. `kind` is "k" or "v" -- they decode differently
    (K rotates the mean and Hadamards the residual; folded-V does neither).
    """

    kind: str
    idx: torch.Tensor
    norm2: torch.Tensor
    norm_q: torch.Tensor
    norm_scale: torch.Tensor
    norm_min: torch.Tensor
    mean_q: torch.Tensor
    mean_scale: torch.Tensor
    mean_min: torch.Tensor
    mean_group: int
    window_size: int
    head_dim: int
    dtype: torch.dtype
    v_hadamard_folded: bool = False
    # dtype `_scale_adjust` returned in. norm2 is STORED fp16 but must be promoted back to this
    # before use: with a bf16 cache the original arithmetic is fp32, and decoding in fp16 would
    # drag the 128-dim Hadamard down with it.
    norm2_arith_dtype: torch.dtype = torch.float32
    # Height of the original window for Hadamard reconstruction. Padding slices to this height
    # keeps TF32 tiling and reduction order consistent with whole-window decoding.
    hadamard_ws: int = None

    def nbytes(self) -> int:
        """Stored bytes, counting idx/norm_q/mean_q at their true 4-or-8-bit width rather than
        at the uint8 they are carried in (see pack_nibbles for the actual packing)."""
        b = self.idx.numel()                     # 8 bits each
        b += self.norm2.numel() * 2              # fp16, whatever container it arrived in
        b += self.norm_q.numel() / 2             # 4 bits each
        b += (self.norm_scale.numel() + self.norm_min.numel()) * self.norm_scale.element_size()
        b += self.mean_q.numel() / 2
        b += (self.mean_scale.numel() + self.mean_min.numel()) * self.mean_scale.element_size()
        return int(b)

    def bits_per_channel(self) -> float:
        # leading axes may be [H] (one window) or [B, H] (a batch of them)
        return self.nbytes() * 8.0 / (self.norm_q.numel() * self.head_dim)


def _rtn4_encode(w: torch.Tensor, group_size: int):
    """`_rtn4_roundtrip` split at its midpoint: returns the codes and their (scale, min).

    Same amax/amin/clamp/round in the same dtype, so `_rtn4_decode` of these codes is the exact
    tensor `_rtn4_roundtrip` would have returned. The codes leave as uint8 because 0..15 is
    exact in every float dtype involved -- the hop cannot round.
    """
    shape = w.shape
    flat = w.reshape(-1, group_size)
    w_max = flat.amax(dim=-1, keepdim=True)
    w_min = flat.amin(dim=-1, keepdim=True)
    scale = torch.clamp((w_max - w_min) / RTN4_LEVELS, min=1e-5)
    q = ((flat - w_min) / scale).clamp_(0, RTN4_LEVELS).round_()
    return q.to(torch.uint8).reshape(shape), scale, w_min


def _rtn4_decode(q: torch.Tensor, scale: torch.Tensor, w_min: torch.Tensor,
                 group_size: int, dtype: torch.dtype) -> torch.Tensor:
    shape = q.shape
    flat = q.reshape(-1, group_size).to(dtype)
    return (flat * scale + w_min).reshape(shape)


def _nsn_encode(x: torch.Tensor, window_size: int, mean_group: int = 32):
    """`_nsn_transform` with the two RTN round-trips split into encode/decode halves.

    Returns (x_normalised, norm_parts, mean_parts, norm2) where the *decoded* norm/mean are used
    downstream -- exactly the values `_nsn_transform` uses, so the normalised x is identical.
    """
    *lead, ws, D = x.shape
    dtype = x.dtype

    x_norm = x.norm(p=2, dim=-1, keepdim=True) / math.sqrt(D)
    nq, nsc, nmin = _rtn4_encode(x_norm.reshape(*lead, 1, ws), ws)
    x_norm = _rtn4_decode(nq, nsc, nmin, ws, dtype).reshape(*lead, ws, 1)
    x = x / x_norm

    x_mean = x.mean(dim=-2, keepdim=True)  # [H, 1, D]
    mq, msc, mmin = _rtn4_encode(x_mean, mean_group)
    x_mean = _rtn4_decode(mq, msc, mmin, mean_group, dtype)
    x = x - x_mean

    x_norm2 = x.norm(p=2, dim=-1, keepdim=True) / math.sqrt(D)
    x = x / x_norm2
    return x, (nq.reshape(*lead, ws), nsc, nmin), (mq, msc, mmin), x_norm2


_FAST_VQ = os.environ.get("SGLANG_NSN_FAST_VQ", "1") == "1"


def _vq_indices(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """`_vq_nearest`'s argmin, kept instead of discarded.

    The default fused Triton search avoids materializing the full distance matrix.
    ``SGLANG_NSN_FAST_VQ=0`` restores the ``cdist`` reference."""
    if _FAST_VQ:
        from sglang.srt.mem_cache.nsn_vq_fast import vq_indices_fast

        return (vq_indices_fast(x.reshape(-1, 8).float(), codebook.float())
                .to(torch.uint8).reshape(*x.shape[:-1], x.shape[-1] // 8))
    return (
        torch.cdist(x.reshape(-1, 8).float(), codebook.float())
        .argmin(dim=-1)
        .to(torch.uint8)
        .reshape(*x.shape[:-1], x.shape[-1] // 8)
    )


# Fusing the encode may reassociate reductions, so it is disabled by default when bitwise
# agreement with the eager path is required.
_COMPILE = os.environ.get("SGLANG_NSN_COMPILE_ENCODE") == "1"


def _maybe_compile(fn):
    return torch.compile(fn, dynamic=False) if _COMPILE else fn


@torch.no_grad()
def encode_window_k(k_postrope, cos, sin, codebook, window_size, mean_group: int = 32):
    """[H, ws, D] post-RoPE K -> PackedWindow. Mirrors `_transform_window_k`'s first half."""
    *_lead, ws, D = k_postrope.shape
    cos_h, sin_h = cos.unsqueeze(-3), sin.unsqueeze(-3)
    k_prerope = _apply_rope(k_postrope, cos_h, -sin_h)
    x, (nq, nsc, nmin), (mq, msc, mmin), norm2 = _nsn_encode(k_prerope, window_size, mean_group)
    x = _apply_rope(x, cos_h, sin_h)
    x_had = _hadamard(x)

    idx = _vq_indices(x_had, codebook)
    # scale_adjust needs vq_val, which is exactly the codebook gather the decoder will redo
    norm2 = _scale_adjust(x_had, codebook[idx.long()].reshape(x_had.shape), norm2)
    arith = norm2.dtype
    return PackedWindow("k", idx, norm2.to(torch.float16), nq, nsc, nmin, mq, msc, mmin,
                        mean_group, window_size, D, k_postrope.dtype,
                        norm2_arith_dtype=arith)


@torch.no_grad()
def encode_window_v(v, codebook, window_size, v_hadamard_folded=False, mean_group: int = 32):
    """[H, ws, D] V -> PackedWindow. Mirrors `_transform_window_v`'s first half."""
    *_lead, ws, D = v.shape
    x, (nq, nsc, nmin), (mq, msc, mmin), norm2 = _nsn_encode(v, window_size, mean_group)
    x_had = x if v_hadamard_folded else _hadamard(x)
    idx = _vq_indices(x_had, codebook)
    norm2 = _scale_adjust(x_had, codebook[idx.long()].reshape(x_had.shape), norm2)
    arith = norm2.dtype
    return PackedWindow("v", idx, norm2.to(torch.float16), nq, nsc, nmin, mq, msc, mmin,
                        mean_group, window_size, D, v.dtype, v_hadamard_folded,
                        norm2_arith_dtype=arith)


@torch.no_grad()
def decode_window(p: PackedWindow, codebook, cos=None, sin=None) -> torch.Tensor:
    """PackedWindow -> [H, ws, D], bitwise equal to `_transform_window_{k,v}`'s output.

    cos/sin are required for kind == "k" (the mean is stored pre-RoPE, as the original computes
    it, and rotated on the way out).
    """
    *lead, ws = p.norm_q.shape
    norm = _rtn4_decode(p.norm_q.reshape(*lead, 1, ws), p.norm_scale, p.norm_min,
                        ws, p.dtype).reshape(*lead, ws, 1)
    mean = _rtn4_decode(p.mean_q, p.mean_scale, p.mean_min, p.mean_group, p.dtype)

    # NO cast to p.dtype here. `_vq_nearest` returns the gather in the CODEBOOK's dtype (fp16),
    # so with a bf16 cache the original's `vq_val * norm2` is fp16*bf16 -> fp32, and the rest of
    # the window rides that promotion. Casting the gather to bf16 first makes it bf16*bf16 and
    # shifts the low bits of every element -- it also flips the odd near-tie downstream, which is
    # how this showed up: fp16 matched bitwise while bf16 was off by up to 8.7e-3.
    vq_val = codebook[p.idx.long()].reshape(*lead, ws, p.head_dim)
    recon_had = vq_val * p.norm2.to(p.norm2_arith_dtype)

    def _had(t):
        n = p.hadamard_ws
        if n is None or n == t.shape[-2]:
            return _hadamard(t)
        pad = t.new_zeros(*t.shape[:-2], n, t.shape[-1])
        pad[..., : t.shape[-2], :] = t
        return _hadamard(pad)[..., : t.shape[-2], :]

    if p.kind == "k":
        assert cos is not None and sin is not None, "K decode needs the window's cos/sin"
        cos_h, sin_h = cos.unsqueeze(-3), sin.unsqueeze(-3)
        mean_roped = _apply_rope(mean.expand(*mean.shape[:-2], ws, -1), cos_h, sin_h)
        return (_had(recon_had) + mean_roped) * norm

    recon_nohad = recon_had if p.v_hadamard_folded else _had(recon_had)
    return (recon_nohad + mean.expand(*mean.shape[:-2], ws, -1)) * norm


def pack_nibbles(q: torch.Tensor) -> torch.Tensor:
    """[..., 2n] uint8 codes in 0..15 -> [..., n] uint8, low nibble first.

    Separate from encode so the numeric path never depends on the physical layout: encode/decode
    are byte-identical whether or not the nibbles are packed, and this pair is what turns the
    accounting in `nbytes()` into actual bytes on the wire.
    """
    assert q.dtype == torch.uint8 and q.shape[-1] % 2 == 0
    # masked rather than asserted: `int(q.max())` is a device->host sync, which is illegal
    # during CUDA-graph capture and is on the captured flush path now
    lo, hi = q[..., 0::2] & 0x0F, q[..., 1::2] & 0x0F
    return (lo | (hi << 4)).contiguous()


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    out = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.uint8,
                      device=packed.device)
    out[..., 0::2] = packed & 0x0F
    out[..., 1::2] = packed >> 4
    return out
