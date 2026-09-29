"""Convert TaSQ's fitted codebooks and transform metadata to the vq2 bundle schema.

The published K configuration applies channel weighting, one FP16 RMS scale pooled
across all KV heads of a token, and a RoPE-pair-preserving permutation. The KV-head
count is read from the model configuration by ``build_bundles.sh``. V uses plain
contiguous CQ without these transforms.

K side: The encoder's order is weight -> post-norm scale -> permute
-> VQ. The post-norm scale is a genuine per-token dynamic quantity (RMS
pooled across all KV heads and channels), so it cannot be folded into a static
matrix. It is stored once per token and layer in FP16 and handled at serving time
by ``vq_codebook.py``'s ``pool_heads_scale`` path.

Weight and permutation ARE both static (channel-wise, data-independent at
serve time) and DO fold into one linear map per head:
    x_weighted  = x * w                       (elementwise)
    x_permuted  = x_weighted[perm]             (gather)
  =>  forward = diag(w) @ P        where P[i, perm[i]] = 1
Scale is computed from the weighted+permuted vector's sum-of-squares, which a
permutation leaves invariant, so computing it post-``forward`` (i.e. after
weight AND permute) gives the identical value the reference computes between
weight and permute -- order relative to the permutation doesn't matter here.
"""
import argparse
import pickle
import re

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantizer_path", required=True,
                     help="TaSQ's own CQ-format codebook pickle, e.g. "
                          "quantizers_..._8c10b_g8_..._matching.pickle")
    ap.add_argument("--tasq_path", required=True,
                     help="tasq_c8_....pkl (perms/weights/head_order)")
    ap.add_argument("--out_k", required=True)
    ap.add_argument("--out_v", required=True)
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument("--group_dim", type=int, default=8)
    args = ap.parse_args()

    with open(args.quantizer_path, "rb") as f:
        q = pickle.load(f)
    with open(args.tasq_path, "rb") as f:
        tasq = pickle.load(f)

    layers = sorted({int(m.group(1)) for k in q if (m := re.search(r"layers\.(\d+)\.", k))})
    assert layers == list(range(len(layers)))
    L = len(layers)
    D = args.head_dim
    G = args.group_dim
    per_head_groups = D // G

    def groups_for(layer, proj):
        key = f"model.layers.{layer}.self_attn.{proj}"
        t = q[key]
        assert t.shape[-1] == G
        H = t.shape[0] // per_head_groups
        return H, t

    H, t0 = groups_for(0, "k_proj")
    K = t0.shape[1]
    print(f"layers={L} heads={H} codebook_size={K} group_dim={G} head_dim={D}")

    bounds = [(g * G, (g + 1) * G, K.bit_length() - 1) for g in range(per_head_groups)]

    # ---- K: forward = diag(w) @ P per (layer, head), for the ENCODE side only
    # (vq_map_k at prefill/flush write time still needs to land raw K in the
    # same permuted+weighted space the codebook was trained in). ----
    #
    # Decode does NOT use forward/inverse/q_map at all -- see decode_attention.py's
    # PERM_ROPE branch: it unweights (WPerm) and rotates each stored pair using
    # its ORIGINAL frequency index (FreqIdx), landing K on the plain original
    # channels FreqIdx[pair]/FreqIdx[pair]+D/2, and Q is gathered at those SAME
    # channels (never mapped). No D x D matrix or matrix inverse at decode time,
    # and the HP/exact tier (raw K, plain Q) needs no special-casing at all.
    half = D // 2
    k_forward = torch.zeros(L, H, D, D, dtype=torch.float64)
    k_inverse = torch.zeros(L, H, D, D, dtype=torch.float64)
    freq_idx = torch.zeros(L, H, half, dtype=torch.int64)
    w_perm = torch.zeros(L, H, D, dtype=torch.float64)
    for l in range(L):
        perm_l = np.asarray(tasq["perms"][l])       # [H, D]
        w_l = np.asarray(tasq["weights"][l])         # [H, D]
        for h in range(H):
            w = torch.from_numpy(w_l[h]).double()
            perm = torch.from_numpy(perm_l[h]).long()
            assert w.shape[0] == D and perm.shape[0] == D
            assert (w != 0).all(), f"zero weight at layer {l} head {h} -- forward not invertible"
            pair_diff = perm[1::2] - perm[0::2]
            assert (pair_diff == half).all(), (
                f"layer {l} head {h}: perm does not preserve RoPE pairs at "
                f"adjacent stored positions (expected perm[2k+1]-perm[2k]=={half} "
                f"for all k) -- the PERM_ROPE decode kernel assumes this exactly."
            )
            freq_idx[l, h] = perm[0::2]
            w_perm[l, h] = w[perm]
            # P must satisfy (x @ P)[j] == x[perm[j]] (row-vector gather
            # convention, matching k_transformed[j] = k_orig[perm[j]]*w[perm[j]]),
            # i.e. P[perm[j], j] = 1 -- NOT P[j, perm[j]] (that gives the
            # inverse gather, x[perm^-1[j]], and silently produces a wrong
            # q_map with no shape/dtype error to catch it).
            P = torch.zeros(D, D, dtype=torch.float64)
            P[perm, torch.arange(D)] = 1.0
            fwd = torch.diag(w) @ P
            k_forward[l, h] = fwd
            k_inverse[l, h] = torch.linalg.inv(fwd)
    k_mean = torch.zeros(L, H, D, dtype=torch.float64)

    codebooks_k = {}
    for l in range(L):
        Hp, tk = groups_for(l, "k_proj")
        assert Hp == H
        for h in range(H):
            codebooks_k[(l, h)] = [
                tk[h * per_head_groups + g].to(torch.float16).contiguous()
                for g in range(per_head_groups)
            ]
    bundle_k = {
        "forward": k_forward.to(torch.float32),
        "inverse": k_inverse.to(torch.float32),
        "mean": k_mean.to(torch.float32),
        "codebooks": codebooks_k,
        "bounds": bounds,
        "pertoken_norm": True,
        "pool_heads_scale": True,
        "perm_rope": True,
        "freq_idx": freq_idx,
        "w_perm": w_perm.to(torch.float32),
    }
    torch.save(bundle_k, args.out_k)
    print(f"wrote {args.out_k}")

    # ---- V: untouched in this config (use_*_v all False) -- identical to plain CQ ----
    eye = torch.eye(D, dtype=torch.float32)
    v_forward = eye.unsqueeze(0).unsqueeze(0).expand(L, H, D, D).contiguous()
    v_mean = torch.zeros(L, H, D, dtype=torch.float32)
    codebooks_v = {}
    for l in range(L):
        Hp, tv = groups_for(l, "v_proj")
        assert Hp == H
        for h in range(H):
            codebooks_v[(l, h)] = [
                tv[h * per_head_groups + g].to(torch.float16).contiguous()
                for g in range(per_head_groups)
            ]
    bundle_v = {
        "forward": v_forward,
        "inverse": v_forward.clone(),
        "mean": v_mean,
        "codebooks": codebooks_v,
        "bounds": bounds,
        "pertoken_norm": False,
        "pool_heads_scale": False,
    }
    torch.save(bundle_v, args.out_v)
    print(f"wrote {args.out_v}")


if __name__ == "__main__":
    main()
