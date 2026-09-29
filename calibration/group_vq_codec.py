#!/usr/bin/env python3
"""Real group-VQ codebook trainer for K-cache compression.

Fills the gap flagged during the faithful-report audit (2026-07-07): every
VQ-tagged file in this directory (`vq_fused.cu`, `vq_throughput_fused.py`,
`decode_e2e.py`, `bw_clean.py`) times decode against a *synthetic random*
codebook -- none of them trains one, so the 0.71/0.96 top-1/top-5 numbers in
`notes/entropy_coding_throughput_report.md` don't trace to any script here.

Groups the QPCA-transformed residual r = (k - mean) @ F into consecutive
chunks of G coordinates (deployment default G=4: 2 bits/coord * 4 = 8 bits/group
-> K=256-entry codebook; 2 KB/head fp16 or 256 B/head with fp8 codebook -- fits
L1 for the fastest gather path in `fused_decode_all.py` VEC/VEC8). d=128 is
evenly divisible by G=4 (128 = 32*4), no trailing remainder group needed.
Alternate configs: G=6 (K=4096, ~1MB/head, exceeds 192KB L1 -> L2 gather, ~5x
slower). The original G=6 prototypes silently dropped the trailing 2 coords
(d_eff=126); this trainer handles remainder groups via `group_boundaries`.

roundtrip(k) interface matches PerCoordCompressor's, so a GroupVQCompressor
plugs directly into test_codec_on_data.py's `comps[method][b][(l,h)]`.
"""
from __future__ import annotations

import torch


def group_boundaries(d: int, G: int, bpc: int = 2) -> list[tuple[int, int, int]]:
    """Return (start, end, bits_for_group) for each group covering [0, d).

    Each full group gets G coords at `bpc` bits/coord (K = 2**(bpc*G)); a
    trailing remainder group (if d % G != 0) gets bpc*rem bits, so the whole
    vector is covered at a uniform `bpc` bits/coord. Default bpc=2.
    """
    bounds = []
    n_full = d // G
    for i in range(n_full):
        bounds.append((i * G, (i + 1) * G, bpc * G))
    rem = d - n_full * G
    if rem > 0:
        bounds.append((n_full * G, d, bpc * rem))
    return bounds


def _kmeans(x: torch.Tensor, K: int, iters: int = 25, seed: int = 0,
            ecvq_lambda: float = 0.0) -> torch.Tensor:
    """Batched Lloyd's algorithm. x: (N, g). Returns centroids (K, g).

    K-means++-lite init (random distinct points, no distance-weighted
    sampling -- fine at K in the thousands where random points already
    spread out well). Empty clusters are reseeded from the globally
    farthest-from-its-centroid points each iteration so no centroid starves
    permanently when N is small relative to K (the expected regime here:
    ~8k calibration tokens per head against up to 4096 centroids).

    ECVQ (Chou-Lookabaugh-Gray 1989): with ecvq_lambda>0 the assignment is
    penalized by the codeword rate, argmin_i ||x-c_i||^2 + lambda*(-log2 p_i),
    where p_i is the previous epoch's empirical usage. This is a RATE-DISTORTION
    training objective (biases toward popular/cheap centroids so the index
    entropy drops); the DECODER is unchanged (fixed-length index gather).
    ecvq_lambda=0 recovers plain k-means exactly (uniform-init penalty is a
    per-argmin constant at epoch 0).
    """
    N, g = x.shape
    gen = torch.Generator(device=x.device).manual_seed(seed)
    if N <= K:
        # Fewer samples than centroids: every point is its own centroid,
        # pad the rest by resampling with replacement (with tiny jitter so
        # duplicate centroids don't collapse to one point post-Lloyd).
        idx = torch.randint(0, N, (K,), generator=gen, device=x.device)
        cent = x[idx].clone()
        cent[N:] += 1e-4 * torch.randn(cent[N:].shape, generator=gen, device=x.device)
        return cent
    idx = torch.randperm(N, generator=gen, device=x.device)[:K]
    cent = x[idx].clone()

    # ECVQ per-centroid rate penalty -log2(p_i), from the PREVIOUS epoch's usage.
    # Uniform init (constant) => epoch 0 == plain k-means. Only used when lambda>0.
    cost = torch.zeros(K, device=x.device, dtype=x.dtype)

    # Chunk the assignment over N so the (block, K) distance matrix stays bounded:
    # the full (N, K) cdist OOMs once N (~1M tokens) x K (up to 8192) exceeds GPU
    # memory. Cap block x K at ~2e8 entries (~0.8 GB fp32).
    block = max(4096, min(N, int(2e8 // max(K, 1))))

    for _ in range(iters):
        new_cent = torch.zeros_like(cent)
        counts = torch.zeros(K, device=x.device, dtype=x.dtype)
        min_d = torch.empty(N, device=x.device, dtype=x.dtype)
        assign_all = torch.empty(N, device=x.device, dtype=torch.long)
        for s in range(0, N, block):
            xb = x[s:s + block]
            db = torch.cdist(xb, cent)          # (b, K) euclidean
            if ecvq_lambda > 0.0:
                # penalized objective uses SQUARED distance + rate; store sqrt-equiv
                # for the min_d reseed metric below (still a monotone distortion proxy).
                obj = db * db + ecvq_lambda * cost[None, :]
            else:
                obj = db
            ab = obj.argmin(dim=1)              # (b,)
            new_cent.index_add_(0, ab, xb)
            counts.index_add_(0, ab, torch.ones(xb.shape[0], device=x.device, dtype=x.dtype))
            min_d[s:s + block] = db.gather(1, ab.unsqueeze(1)).squeeze(1)
            assign_all[s:s + block] = ab
        empty = counts == 0
        nonempty = ~empty
        new_cent[nonempty] /= counts[nonempty].unsqueeze(-1)

        n_empty = int(empty.sum())
        if n_empty > 0:
            # Reseed empties from the points currently farthest from their
            # assigned centroid (classic Lloyd's dead-cluster fix).
            worst = torch.argsort(min_d, descending=True)[:n_empty]
            new_cent[empty] = x[worst]

        # Update rate penalty for the NEXT epoch from this epoch's usage.
        # Reseeded/dead centroids get cost 0 so they can attract points again
        # (otherwise a -log2(0) penalty would freeze them out and defeat reseeding).
        if ecvq_lambda > 0.0:
            p = counts / counts.sum().clamp_min(1.0)
            cost = -torch.log2(p.clamp_min(1.0 / (N * K)))
            cost[empty] = 0.0

        cent = new_cent
    return cent


def _assign(x: torch.Tensor, cent: torch.Tensor) -> torch.Tensor:
    """Nearest-centroid indices. x: (N, g), cent: (K, g) -> (N,) int64."""
    return torch.cdist(x, cent).argmin(dim=1)


class GroupVQCompressor:
    """Group vector quantizer over a QPCA-basis residual, 2 bits/coord.

    codebooks[i]: (K_i, g_i) tensor for group i (group boundaries from
    `group_boundaries`). forward_map/inverse_map/mean follow the same
    row-vector convention as PerCoordCompressor: transformed = (k - mean) @
    forward_map; recon = (r_hat @ inverse_map) + mean.
    """

    def __init__(self, forward_map, inverse_map, mean, codebooks, bounds, pertoken_norm=False):
        self.forward_map = forward_map
        self.inverse_map = inverse_map
        self.mean = mean
        self.codebooks = codebooks   # list of (K_i, g_i)
        self.bounds = bounds         # list of (start, end, bits)
        # OSCAR/KIVI-style per-token dynamic scale: divide each token's residual by
        # its own RMS before the codebook lookup, multiply back on decode. Keeps the
        # fixed codebook in-distribution at any context length (a token at RoPE
        # position 60k is scaled back to the calibrated scale). Costs 1 scalar/token
        # of metadata (~0.06 b/coord at fp8); the codebook must be trained with the
        # same normalization (train_group_vq_alloc --pertoken-norm).
        self.pertoken_norm = pertoken_norm

    def to(self, device):
        self.forward_map = self.forward_map.to(device)
        self.inverse_map = self.inverse_map.to(device)
        self.mean = self.mean.to(device)
        self.codebooks = [c.to(device) for c in self.codebooks]
        return self

    def encode_idx(self, k: torch.Tensor) -> list[torch.Tensor]:
        """k: (T, d) -> per-group nearest-centroid indices, one (T,) int64 tensor
        per group. This is the "compressed representation" -- what a real cache
        would store -- so decode-only timing can be isolated from re-quantization."""
        r = (k.double() - self.mean.double()) @ self.forward_map.double()
        return [_assign(r[:, s:e], cb.double()) for (s, e, _bits), cb in zip(self.bounds, self.codebooks)]

    def decode_idx(self, idx_list: list[torch.Tensor], dtype=torch.float32) -> torch.Tensor:
        """Inverse of encode_idx: group indices -> k_hat: (T, d). The actual
        decode-time cost (gather + inverse transform), matching what a real
        serving system pays per decode step against a resident quantized cache."""
        d = self.bounds[-1][1]
        T = idx_list[0].shape[0]
        r_hat = torch.empty(T, d, dtype=torch.float64, device=idx_list[0].device)
        for (s, e, _bits), cb, idx in zip(self.bounds, self.codebooks, idx_list):
            r_hat[:, s:e] = cb.double()[idx]
        k_hat = r_hat @ self.inverse_map.double() + self.mean.double()
        return k_hat.to(dtype)

    def roundtrip(self, k: torch.Tensor) -> torch.Tensor:
        """k: (..., d) -> k_hat: (..., d). Flattens leading dims so the same codec
        serves the 2-D accuracy scorer and the (B, S, d) tensors a kvpress hook passes."""
        dtype = k.dtype
        lead = k.shape[:-1]
        kf = k.reshape(-1, k.shape[-1])
        r = (kf.double() - self.mean.double()) @ self.forward_map.double()
        if self.pertoken_norm:
            scale = r.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
            r = r / scale
        r_hat = torch.empty_like(r)
        for (s, e, _bits), cb in zip(self.bounds, self.codebooks):
            idx = _assign(r[:, s:e], cb.double())
            r_hat[:, s:e] = cb.double()[idx]
        if self.pertoken_norm:
            r_hat = r_hat * scale
        k_hat = r_hat @ self.inverse_map.double() + self.mean.double()
        return k_hat.reshape(*lead, k.shape[-1]).to(dtype)

    @property
    def bits_per_coord(self) -> float:
        d = self.bounds[-1][1]
        return sum(b for _, _, b in self.bounds) / d


def train_group_vq_compressors(F, inv, k_mean, fetch_calib, L, Hkv, d, G=4,
                                iters=25, device="cuda", seed=0, verbose=True):
    """Train one GroupVQCompressor per (layer, kv_head).

    fetch_calib(l, h) -> (N_calib_tokens, d) centered+transformed residual
    codes, e.g. `run_pca_ec_deadzone._codes_for_idx(...)`'s returned fetch.
    """
    bounds = group_boundaries(d, G)
    comps = {}
    for l in range(L):
        for h in range(Hkv):
            r = fetch_calib(l, h).to(device)
            codebooks = []
            for (s, e, bits) in bounds:
                K = 1 << bits
                seg = r[:, s:e]
                cb = _kmeans(seg, K, iters=iters, seed=seed + l * Hkv + h)
                codebooks.append(cb)
            comps[(l, h)] = GroupVQCompressor(
                F[l, h].to(device), inv[l, h].to(device), k_mean[l, h].to(device),
                codebooks, bounds)
        if verbose:
            print(f"  [group-VQ] layer {l}/{L} trained ({Hkv} heads)", flush=True)
    return comps


class SinkRecentWrap:
    """Keep the first `sink` and last `recent` sequence positions in fp16 and
    quantize only the middle with the wrapped compressor -- a small outlier-
    protection band (same idea as OSCAR's sink/recent, but sized tiny so the
    bit-rate cost stays negligible). k: (B, S, d) from the kvpress hook, so the
    sequence axis is dim -2. `.roundtrip` / `.to` mirror GroupVQCompressor."""

    def __init__(self, inner, sink: int = 0, recent: int = 0):
        self.inner = inner
        self.sink = int(sink)
        self.recent = int(recent)
        self.forward_map = getattr(inner, "forward_map", None)

    def to(self, device):
        self.inner.to(device)
        return self

    def roundtrip(self, k):
        if k.dim() < 2 or (self.sink == 0 and self.recent == 0):
            return self.inner.roundtrip(k)
        S = k.shape[-2]
        s = min(self.sink, S)
        r = min(self.recent, max(S - s, 0))
        if s + r >= S:
            return k                                  # whole (short) seq protected
        out = k.clone()
        out[..., s:S - r, :] = self.inner.roundtrip(k[..., s:S - r, :])
        return out


class OutlierProtectWrap:
    """Content-based (not positional) outlier protection: quantize every token
    with the wrapped compressor, then restore the `frac` fraction of tokens with
    the largest reconstruction error to full precision. Targets atypical keys a
    fixed codebook can't represent (e.g. a NIAH needle) -- the variable-rate
    behaviour that a fixed-rate VQ otherwise lacks. Stacks on top of a
    SinkRecentWrap (positional band) so the two protections compose; the band's
    exact tokens have ~0 error so they're never double-counted. k: (..., S, d),
    sequence axis is dim -2. `.roundtrip`/`.to` mirror GroupVQCompressor.

    Rate cost: `frac` of tokens stored at `store_bits` b/coord instead of the
    codec's base rate -> +frac*(store_bits - base) b/coord (fp8 store_bits=8)."""

    def __init__(self, inner, frac: float = 0.0):
        self.inner = inner
        self.frac = float(frac)
        self.forward_map = getattr(inner, "forward_map", None)

    def to(self, device):
        self.inner.to(device)
        return self

    def roundtrip(self, k):
        out = self.inner.roundtrip(k)
        if self.frac <= 0 or k.dim() < 2:
            return out
        S = k.shape[-2]
        n = int(self.frac * S)
        if n <= 0:
            return out
        err = ((k.double() - out.double()) ** 2).sum(-1)         # (..., S)
        idx = err.topk(n, dim=-1).indices                        # worst-n token positions
        # emulate an fp8 side-buffer for the protected keys (realistic rate, not fp16 ceiling)
        prot = k.gather(-2, idx.unsqueeze(-1).expand(*idx.shape, k.shape[-1]))
        prot = prot.to(torch.float8_e4m3fn).to(out.dtype)
        out.scatter_(-2, idx.unsqueeze(-1).expand(*idx.shape, k.shape[-1]), prot)
        return out
