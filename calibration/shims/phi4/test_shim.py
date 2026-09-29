"""Gate the qkv shim on the REAL Phi-4 weights before any codebook is fitted from it.

Two checks, both on shard 1 (no full model load, no GPU):
  1. the slice algebra: cat(q,k,v matmuls) vs the fused matmul, on the actual layer-0 weight;
  2. the packing ORDER: that rows [0:nq] really are Q and [nq:nq+nkv] really are K. A shim that
     silently swapped K and V would pass check 1 and poison every bundle, so this compares each
     slice against the same rows fetched independently by name from the checkpoint... which for a
     fused tensor is impossible -- instead it checks the only observable that distinguishes them:
     Q and K are rotated by RoPE and V is not, so Q/K rows carry the paired structure that the
     RoPE frequency layout implies, while V does not. Cheap proxy: the row-norm profile of the
     three blocks. Reported, not asserted, since it is heuristic.
"""
import glob
import json
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phi3_shim import verify_split  # noqa: E402

SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--microsoft--Phi-4-reasoning-plus/snapshots/*"))[0]
cfg = json.load(open(os.path.join(SNAP, "config.json")))
hd = cfg["hidden_size"] // cfg["num_attention_heads"]
nq, nkv = cfg["num_attention_heads"] * hd, cfg["num_key_value_heads"] * hd
print(f"config: {cfg['num_hidden_layers']}L  hidden={cfg['hidden_size']}  "
      f"heads={cfg['num_attention_heads']}/{cfg['num_key_value_heads']}  head_dim={hd}")
print(f"        rope_theta={cfg['rope_theta']}  partial_rotary={cfg.get('partial_rotary_factor')}  "
       f"sliding_window={cfg.get('sliding_window')}")
assert hd == 128, f"the VQ kernels are specialised to head_dim 128, got {hd}"
assert cfg.get("partial_rotary_factor", 1.0) == 1.0, "partial rotary would break the kernel's rotation"
assert cfg.get("sliding_window") in (None, 0), "sliding window changes the KV pool contract"

name = "model.layers.0.self_attn.qkv_proj.weight"
shard = None
idx = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))
shard = idx["weight_map"][name]
with safe_open(os.path.join(SNAP, shard), framework="pt") as f:
    w = f.get_tensor(name)
print(f"\n{name}: {tuple(w.shape)} {w.dtype}   expect ({nq + 2 * nkv}, {cfg['hidden_size']})")
assert tuple(w.shape) == (nq + 2 * nkv, cfg["hidden_size"]), "unexpected fused shape"

ma, mr = verify_split(w, nq, nkv)
print(f"slice algebra: max abs {ma:.3e}   max rel {mr:.3e}")
assert mr < 1e-2, f"split deviates too much ({mr:.3e}); not a tiling difference"

qn = w[:nq].float().norm(dim=1)
kn = w[nq:nq + nkv].float().norm(dim=1)
vn = w[nq + nkv:].float().norm(dim=1)
print(f"row-norm means  Q {qn.mean():.4f}   K {kn.mean():.4f}   V {vn.mean():.4f}  "
      f"(reported, not asserted)")
print("\nSHIM GATE PASSED")
