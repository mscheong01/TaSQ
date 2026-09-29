#!/usr/bin/env python3
"""Generate the NSN 8-dim VQ codebook.

NSNQuant's codebook is *synthetic*: k-means over samples from a standard normal, with no model
and no corpus involved. That is what makes the NSN arm calibration-free -- there is nothing to
share between nodes, so every node generates its own.

    python codebooks/make_nsn_codebook.py --out work/nsn_1bit_codebook.pt

Only numpy/scipy/torch are needed. Note the upstream generator also offers a `learned` mode that
refines the centroids with a CUDA extension; that mode is not reproduced here (it would drag in
that extension), so a codebook from this script is statistically equivalent to, but not bitwise
identical with, one from the `learned` path.
"""
import argparse
import random

import numpy as np
import torch
from scipy.cluster.vq import kmeans

N_SAMPLES = 16384
GROUP_DIM = 8       # NSN quantizes K/V in 8-dim groups
N_CENTROIDS = 256   # 8 bits per group == 1 bit/coord


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--samples", type=int, default=N_SAMPLES)
    ap.add_argument("--group-dim", type=int, default=GROUP_DIM)
    ap.add_argument("--centroids", type=int, default=N_CENTROIDS)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = np.random.normal(size=(args.samples, args.group_dim))
    codebook = torch.from_numpy(kmeans(data, args.centroids)[0])
    if codebook.shape[0] != args.centroids:
        raise SystemExit(f"k-means returned {codebook.shape[0]} centroids, expected {args.centroids}")
    torch.save(codebook, args.out)
    print(f"wrote {args.out}  shape={tuple(codebook.shape)}  seed={args.seed}")


if __name__ == "__main__":
    main()
