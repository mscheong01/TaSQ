"""Convert this project's own CoupledQuantizer (CQ) pickle -- plain contiguous
8-dim group-VQ, no permutation/weighting/per-token norm -- into this fork's
vq2 codebook bundle schema (forward/inverse/mean/codebooks/bounds/pertoken_norm),
so it can be served through the SAME mixed-KV-window/residual machinery nova's
own method uses, via the new wide-group-dim (G=8) decode kernel.

CQ's centroid tensor per layer is [H*D//c, K, c] (c=8), with the group axis
already in the CoupledQuantizer.forward() convention: for a token's full
[H, D] vector flattened to [H*D] then split into c-sized groups, group index
g maps to head=g//(D//c), local_group=g%(D//c) -- i.e. contiguous within a
head, heads concatenated in head-index order. No permutation/weighting/mean-
centering/per-token-norm are applied (CQ is the un-permuted baseline), so:
  forward = inverse = identity (D x D)
  mean = 0
  pertoken_norm = False
"""
import argparse
import pickle
import re

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantizer_path", required=True,
                     help="CQ pickle, e.g. cq_centroids/quantizers_..._8c10b_....pickle")
    ap.add_argument("--out_k", required=True, help="output K bundle .pt path")
    ap.add_argument("--out_v", required=True, help="output V bundle .pt path")
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument("--group_dim", type=int, default=8)
    args = ap.parse_args()

    with open(args.quantizer_path, "rb") as f:
        q = pickle.load(f)

    layers = sorted({int(m.group(1)) for k in q if (m := re.search(r"layers\.(\d+)\.", k))})
    assert layers == list(range(len(layers))), f"expected contiguous 0..N-1 layers, got {layers}"
    L = len(layers)
    D = args.head_dim
    G = args.group_dim
    per_head_groups = D // G
    assert D % G == 0

    def groups_for(layer, proj):
        key = f"model.layers.{layer}.self_attn.{proj}"
        t = q[key]  # [H*per_head_groups, K, G]
        assert t.shape[-1] == G, f"{key}: expected group_dim {G}, got {t.shape[-1]}"
        H = t.shape[0] // per_head_groups
        assert H * per_head_groups == t.shape[0]
        return H, t

    H, t0 = groups_for(0, "k_proj")
    K = t0.shape[1]
    print(f"layers={L} heads={H} codebook_size={K} group_dim={G} head_dim={D}")

    bounds = [(g * G, (g + 1) * G, K.bit_length() - 1) for g in range(per_head_groups)]
    eye = torch.eye(D, dtype=torch.float32)
    forward = eye.unsqueeze(0).unsqueeze(0).expand(L, H, D, D).contiguous()
    inverse = forward.clone()
    mean = torch.zeros(L, H, D, dtype=torch.float32)

    for proj, out_path in (("k_proj", args.out_k), ("v_proj", args.out_v)):
        codebooks = {}
        for l in range(L):
            Hp, tp = groups_for(l, proj)
            assert Hp == H
            for h in range(H):
                codebooks[(l, h)] = [
                    tp[h * per_head_groups + g].to(torch.float16).contiguous()
                    for g in range(per_head_groups)
                ]
        bundle = {
            "forward": forward,
            "inverse": inverse,
            "mean": mean,
            "codebooks": codebooks,
            "bounds": bounds,
            "pertoken_norm": False,
        }
        torch.save(bundle, out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
