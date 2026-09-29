#!/usr/bin/env python3
"""Does Fisher-weighted k-means++ init actually lower the weighted objective?

The claim under test is narrow: the Lloyd steps were
already Fisher-weighted, only the init was not, so seeding on w_i * D_i^2 instead of D_i^2 should
land in a better local optimum of

    J = sum_i w_i ||x_i - c_{z_i}||^2.

Two things this checks that a single run cannot:

* The comparison is paired per seed. k-means++ is stochastic and the seed-to-seed spread is
  larger than the effect, so an unpaired A/B on one seed each says nothing.
* J is measured with the SAME assignment rule the deployed quantizer uses (nearest centroid,
  unweighted argmin -- the weight cannot change which centroid is closest), not with the
  internal fp16 kernel, so this measures the codebook, not the kernel.

    python test_fisher_kmeans_init.py            # synthetic
    python test_fisher_kmeans_init.py --real <activations.pt>
"""
import argparse

import torch

import kvquant.gpu_kmeans as gk
from kvquant.gpu_kmeans import weighted_kmeans_batch

try:
    import kmeans_tools  # noqa: F401
except ImportError as e:
    # The vendored .so is built per-node against a specific torch. Its only job inside
    # weighted_kmeans_batch is the assignment step (nearest centroid, fp16), so a plain torch
    # reference stands in exactly -- the codebook this test compares is unaffected, only speed.
    print(f"[test] kmeans_tools unavailable ({type(e).__name__}); using the torch reference "
          f"assignment kernel")

    def _ref_argmin(X, C):
        return torch.cdist(X.float(), C.float()).argmin(dim=-1).to(torch.int64)

    gk.get_dist_argmin_half_batched = lambda D: _ref_argmin


def objective(X, w, C):
    """J = sum_i w_i * min_k ||x_i - c_k||^2, per batch element."""
    d2 = torch.cdist(X.float(), C.float()) ** 2          # (B, N, k)
    return (w.float() * d2.min(dim=-1).values).sum(dim=-1)


def synthetic(B, N, D, k, seed, device):
    """Heterogeneous importance is the whole point, so weights must be far from uniform: a
    heavy-tailed lognormal, with the high-weight mass deliberately placed in a SMALL cluster that
    plain k-means++ has little reason to cover."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_small = N // 20
    bulk = torch.randn(B, N - n_small, D, generator=g) * 1.0
    small = torch.randn(B, n_small, D, generator=g) * 0.15 + 4.0
    X = torch.cat([bulk, small], dim=1)
    w = torch.exp(torch.randn(B, N, generator=g) * 1.2)
    w[:, N - n_small:] *= 25.0                            # the important, rare region
    w = w / w.amax(dim=1, keepdim=True)
    return X.to(device), w.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None,
                    help="path to a .pt holding (B,N,D) activations and optional (B,N) weights")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--k", type=int, default=1024, help="2^10 = the CQ code width (abits 10)")
    ap.add_argument("--iters", type=int, default=100)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if a.real:
        blob = torch.load(a.real, map_location=device)
        X, w = (blob["X"], blob["w"]) if isinstance(blob, dict) else (blob[0], blob[1])
        X, w = X.to(device).float(), w.to(device).float()
        print(f"real data: X{tuple(X.shape)} w{tuple(w.shape)}  "
              f"w range [{w.min():.3g}, {w.max():.3g}]")
    else:
        X, w = synthetic(a.B, a.N, a.D, a.k, 0, device)
        print(f"synthetic: X{tuple(X.shape)} w{tuple(w.shape)}")

    wins = 0
    rows = []
    for seed in range(a.seeds):
        torch.manual_seed(seed)
        C_plain, _ = weighted_kmeans_batch(X, w, a.k, num_iters=a.iters, fisher_init=False)
        torch.manual_seed(seed)          # same stream, so only the init rule differs
        C_fish, _ = weighted_kmeans_batch(X, w, a.k, num_iters=a.iters, fisher_init=True)

        j_plain = objective(X, w, C_plain).sum().item()
        j_fish = objective(X, w, C_fish).sum().item()
        rel = 100.0 * (j_fish - j_plain) / j_plain
        wins += rel < 0
        rows.append(rel)
        print(f"  seed {seed}: J plain {j_plain:.6g}   fisher {j_fish:.6g}   {rel:+.2f}%")

    mean = sum(rows) / len(rows)
    print(f"\nweighted objective J, fisher-init vs plain-init: {mean:+.2f}% mean over "
          f"{a.seeds} paired seeds; fisher lower in {wins}/{a.seeds}")
    print("(negative = fisher init reached a better local optimum)")


if __name__ == "__main__":
    main()
