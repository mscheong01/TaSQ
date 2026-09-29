"""OSCAR-style 1-bit V as a vq_v codebook bundle, for the NovaKV arm at matched bits.

OSCAR's original V quantization is rotate-then-scalar-quantize; at 1 bit that is
sign(v_rotated) * scale. The unified pool's vq_v machinery expresses this exactly with a
G=8 codebook holding ALL 256 sign patterns: nearest-codeword search over equal-magnitude
+-c patterns IS the sign function (argmin ||x - c*s||^2 = argmax <x, s> = s = sign(x)),
and pertoken_norm supplies the per-(token, head) RMS scale. Entries are +-c with
c = sqrt(2/pi) ~= 0.7979: after per-token RMS normalization the coordinates are ~unit-RMS,
and E[|x|] = sqrt(2/pi) * RMS for a Gaussian, which is the MSE-optimal magnitude for a
sign reconstruction (recon = c * sign(x) * RMS ~= E[|x|] * sign(x)).

Bit accounting: 8 code bits / 8 coords = 1.0 + fp16 per-token scale (16/128) = 1.125
nominal BPA for V. The pool still applies the artifact's R_v rotation before encode
(OSCAR convention), so this quantizes in rotated space exactly like the original.

forward/inverse are identity and mean is zero: the sign quantizer needs no extra map,
and the nova serving path (unlike TaSQ's PERM_ROPE) reads V reconstructions directly in
the stored (R_v-rotated) space.
"""
import argparse
import itertools
import math

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, required=True)
ap.add_argument("--kv-heads", type=int, default=8)
ap.add_argument("--head-dim", type=int, default=128)
ap.add_argument("--out", required=True)
args = ap.parse_args()

L, H, D, G = args.layers, args.kv_heads, args.head_dim, 8
NG = D // G
c = math.sqrt(2.0 / math.pi)

signs = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=G)), dtype=torch.float32)  # [256, 8]
cb_group = (signs * c).contiguous()

codebooks = {(l, h): [cb_group.clone() for _ in range(NG)] for l in range(L) for h in range(H)}
bounds = [(i * G, (i + 1) * G, G) for i in range(NG)]  # 8 bits per 8-dim group

eye = torch.eye(D, dtype=torch.float32).expand(L, H, D, D).contiguous()
payload = dict(
    forward=eye.clone(),
    inverse=eye.clone(),
    mean=torch.zeros(L, H, D, dtype=torch.float32),
    codebooks=codebooks,
    bounds=bounds,
    pertoken_norm=True,
    pool_heads_scale=False,
)
torch.save(payload, args.out)
print(f"SAVED {args.out}: L={L} H={H} D={D} G={G} NG={NG} K=256 (+-{c:.4f} sign patterns), "
      f"nominal V bits = {G}/{G} + 16/{D} = {1.0 + 16.0 / D:.3f}")
