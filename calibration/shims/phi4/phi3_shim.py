"""Expose q_proj / k_proj / v_proj on HF Phi3 attention, which packs them into one qkv_proj.

Why this is needed: every calibration stage in this repo reaches for the three projections by
name as real submodules --

  * third_party/kvquant/run_fisher.py       `get_modules_kv(layer)` -> layer.self_attn.{k,v}_proj
  * third_party/kvquant/llama_simquant.py   keys quantizers on names containing 'k_proj'/'v_proj'
                                           and builds 'model.layers.%d.%s'
  * codebooks/make_tasq.py                  registers forward hooks on s.k_proj and s.q_proj

-- and HF's Phi3ForCausalLM has only `self_attn.qkv_proj`. Serving is unaffected: sglang runs this
model through Phi3ForCausalLM(LlamaForCausalLM), which already builds a fused QKVParallelLinear and
carries the pre-RoPE VQ hook.

Design. The three Linears are installed as direct children of `self_attn`, so their names match
llama's exactly (`model.layers.N.self_attn.k_proj`) and nothing downstream needs to learn a second
naming convention. `qkv_proj` is then REPLACED by a thin wrapper that calls them and concatenates,
so:

  * HF's own forward still calls `self_attn.qkv_proj(x)` and still gets one [.., q+2kv] tensor;
  * a forward hook on k_proj/q_proj/v_proj actually FIRES, which is the whole point -- adding the
    submodules without rerouting the forward would leave every hook silent and every codebook
    fitted on nothing.

The weights are SLICES of the fused parameter, i.e. views sharing its storage, so this costs no
extra memory. The wrapper holds its parts in a plain tuple rather than as attributes, so they are
not registered twice in named_modules().

Numerics: three matmuls over row slices instead of one fused matmul. Mathematically identical --
each output row depends only on its own weight row -- but cuBLAS may pick different tiling per
shape, so `verify_split` measures the deviation instead of assuming it is zero.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FusedFromParts(nn.Module):
    """Stands in for the original fused qkv_proj, computing it from the three parts.

    Resolves q_proj/k_proj/v_proj on its OWNER at call time, and that is the whole point rather
    than a style choice. run_fisher's `replace_linear_with_linearact` walks the model and does
    `setattr(self_attn, 'k_proj', LinearAct(...))` -- a NEW module holding a copy of the weights.
    A wrapper that captured the original Linears at construction would keep calling those, so the
    LinearAct that `get_modules_kv` later reads would never see a forward and `k_proj.act` would
    stay None. That is exactly how the first Phi-4 Fisher run died
    (`AttributeError: 'NoneType' object has no attribute 'grad'`), with the shim otherwise
    reporting success. Late binding makes any later replacement -- hooks, wrappers, quantised
    stand-ins -- take effect.
    """

    def __init__(self, owner: nn.Module):
        super().__init__()
        object.__setattr__(self, "_owner", owner)   # not a child: avoids a cycle in named_modules

    def forward(self, x):
        o = self._owner
        return torch.cat((o.q_proj(x), o.k_proj(x), o.v_proj(x)), dim=-1)


def _slice_linear(fused: nn.Linear, lo: int, hi: int) -> nn.Linear:
    lin = nn.Linear(fused.in_features, hi - lo, bias=fused.bias is not None,
                    device="meta", dtype=fused.weight.dtype)
    lin.weight = nn.Parameter(fused.weight[lo:hi], requires_grad=False)   # view, no copy
    if fused.bias is not None:
        lin.bias = nn.Parameter(fused.bias[lo:hi], requires_grad=False)
    else:
        lin.bias = None
    return lin


def unfuse_qkv_(model, *, verbose: bool = True) -> int:
    """Give every layer real q_proj/k_proj/v_proj. Idempotent; returns the number of layers done."""
    cfg = model.config
    nq = cfg.num_attention_heads * (cfg.hidden_size // cfg.num_attention_heads)
    nkv = cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads)
    done = 0
    for layer in model.model.layers:
        sa = layer.self_attn
        if hasattr(sa, "k_proj") and isinstance(getattr(sa, "k_proj"), nn.Module):
            continue
        fused = sa.qkv_proj
        if fused.out_features != nq + 2 * nkv:
            raise RuntimeError(
                f"qkv_proj out_features {fused.out_features} != q+2kv = {nq + 2 * nkv}; the "
                "packing order or head counts are not what this shim assumes."
            )
        q = _slice_linear(fused, 0, nq)
        k = _slice_linear(fused, nq, nq + nkv)
        v = _slice_linear(fused, nq + nkv, nq + 2 * nkv)
        sa.q_proj, sa.k_proj, sa.v_proj = q, k, v
        sa.qkv_proj = _FusedFromParts(sa)
        done += 1
    if verbose:
        print(f"[phi3_shim] unfused qkv_proj on {done} layers "
              f"(q={nq}, kv={nkv}, in={model.config.hidden_size})", flush=True)
    return done


def verify_split(weight: torch.Tensor, nq: int, nkv: int, *, n: int = 4, seed: int = 0):
    """Compare cat(three slice matmuls) against one fused matmul on random input.

    Returns (max_abs, max_rel). Called by test_shim.py on a real shard so the claim "the split is
    numerically the same" is measured on the actual weights before any codebook built from it is
    trusted.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, weight.shape[1], generator=g, dtype=torch.float32).to(weight.dtype)
    fused = F.linear(x, weight)
    parts = torch.cat((F.linear(x, weight[:nq]),
                       F.linear(x, weight[nq:nq + nkv]),
                       F.linear(x, weight[nq + nkv:nq + 2 * nkv])), dim=-1)
    d = (fused.float() - parts.float()).abs()
    scale = fused.float().abs().clamp_min(1e-6)
    return d.max().item(), (d / scale).max().item()
