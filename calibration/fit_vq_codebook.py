#!/usr/bin/env python3
"""Fit the group-VQ key-tier codebook (the ``codebook.pt`` bundle) from a QKV dump.

Standalone port of the trainer that produced the shipped bundles
(train_group_vq_alloc.py, "stratified flat ptn" config): per-(layer, kv_head)
QPCA basis from calibration second moments, stratified column permutation,
per-token RMS normalization, k-means codebooks (G coords/group, 2 bits/coord
-> K=256), fp8 codebook quantization. Output loads through
sglang/srt/mem_cache/vq_codebook.py unchanged.

Input: the layout written by dump_qkv.py --
    <dump>/layer_<id>/{q,k}/<chunk>.pt   [T, H, head_dim] fp16
(chunk 0 is a dummy slice and is skipped). Second moments are computed from
the dump; --basis-moments substitutes a precomputed moments file (the schema
gptoss_calibrate.py / capture_mixed_concat.py emit), reproducing a canonical
build's basis exactly.

Build command (canonical gpt-oss-20b config; ~16+ GPQA prompts):

    python calibration/dump_qkv.py --model openai/gpt-oss-20b \
        --prompts gpqa_diamond.csv --num-prompts 16 --out <dump>
    python calibration/fit_vq_codebook.py --dump <dump> --out codebook.pt
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import torch

from group_vq_codec import _kmeans, group_boundaries

FP8 = {"e4m3": (torch.float8_e4m3fn, 448.0), "e5m2": (torch.float8_e5m2, 57344.0)}


# ---------------------------------------------------------------------------
# Dump reading.
# ---------------------------------------------------------------------------
def dump_layout(root: Path) -> tuple[int, list[int]]:
    layers = sorted(
        int(m.group(1))
        for p in root.iterdir()
        if (m := re.fullmatch(r"layer_(\d+)", p.name))
    )
    assert layers == list(range(len(layers))), f"non-contiguous layer dirs in {root}"
    chunks = sorted(int(p.stem) for p in (root / "layer_0" / "k").glob("*.pt"))
    chunks = [c for c in chunks if c != 0]  # chunk 0 is a dummy duplicate slice
    assert chunks, f"no data chunks under {root}/layer_0/k"
    return len(layers), chunks


def compute_moments(root: Path, n_layers: int, chunks: list[int]):
    """Per-(layer, kv_head) uncentered second moments + K mean/cov, float64
    accumulation, GQA-sum pooling of Sigma_Q over each kv head's query group --
    the same math as the reference calibration (calib_moments / cmd_concat)."""
    sumk = sq2 = sk2 = None
    ntok = 0
    for c in chunks:
        for l in range(n_layers):
            q = torch.load(root / f"layer_{l}" / "q" / f"{c}.pt",
                           map_location="cpu", weights_only=False).double()  # [T,Hq,d]
            k = torch.load(root / f"layer_{l}" / "k" / f"{c}.pt",
                           map_location="cpu", weights_only=False).double()  # [T,Hkv,d]
            if sumk is None:
                Hq, Hkv, d = q.shape[1], k.shape[1], k.shape[-1]
                sumk = torch.zeros(n_layers, Hkv, d, dtype=torch.float64)
                sq2 = torch.zeros(n_layers, Hq, d, d, dtype=torch.float64)
                sk2 = torch.zeros(n_layers, Hkv, d, d, dtype=torch.float64)
            sumk[l] += k.sum(0)
            sq2[l] += torch.einsum("thd,the->hde", q, q)
            sk2[l] += torch.einsum("thd,the->hde", k, k)
            if l == 0:
                ntok += q.shape[0]
    Hq, Hkv, d = sq2.shape[1], sk2.shape[1], sk2.shape[-1]
    gs = Hq // Hkv
    Eqq, Ekk, mk = sq2 / ntok, sk2 / ntok, sumk / ntok
    sigma_q = Eqq.reshape(n_layers, Hkv, gs, d, d).sum(2)
    k_cov = Ekk - torch.einsum("lhd,lhe->lhde", mk, mk)
    meta = dict(n_layers=n_layers, n_q_heads=Hq, n_kv_heads=Hkv, d_head=d,
                group_size=gs)
    print(f"moments: {len(chunks)} chunks, {ntok} tokens | "
          f"L={n_layers} Hq={Hq} Hkv={Hkv} d={d}", flush=True)
    # fp32 storage matches the canonical basis_moments.pt files bit-for-bit.
    return (sigma_q.float(), Ekk.float(), mk.float(), k_cov.float(), meta)


# ---------------------------------------------------------------------------
# QPCA basis (build_qpca_basis, tau=1, on the CENTERED K covariance).
# ---------------------------------------------------------------------------
def _sym(x):
    return 0.5 * (x + x.transpose(-1, -2))


def build_qpca_basis(sigma_q: torch.Tensor, k_cov: torch.Tensor):
    """forward = Sigma_Q^{1/2} V, inverse = V^T Sigma_Q^{-1/2}, where V
    diagonalizes Sigma_Q^{1/2} Cov(K) Sigma_Q^{1/2} (eigenvalues descending).
    Non-orthogonal on purpose: Euclidean distortion in code space equals the
    Sigma_Q-weighted (softmax-logit) distortion in key space. float64."""
    sq = _sym(sigma_q.to(torch.float64))
    sk = _sym(k_cov.to(torch.float64))
    ev, U = torch.linalg.eigh(sq)
    ev = ev.clamp_min(1e-30)
    sqrt_mq = U @ torch.diag_embed(ev.pow(0.5)) @ U.transpose(-1, -2)
    isqrt_mq = U @ torch.diag_embed(ev.pow(-0.5)) @ U.transpose(-1, -2)
    A = _sym(sqrt_mq @ sk @ sqrt_mq)
    lam, V = torch.linalg.eigh(A)
    order = torch.argsort(lam, dim=-1, descending=True)
    V = torch.gather(V, -1, order.unsqueeze(-2).expand(*V.shape[:-1], -1))
    return sqrt_mq @ V, V.transpose(-1, -2) @ isqrt_mq


def stratified_perm(d: int, G: int) -> torch.Tensor:
    """rank r -> group (r % NG), contiguous in the permuted basis, so every
    G-group spans the whole eigenvalue spectrum instead of a narrow band."""
    NG = math.ceil(d / G)
    perm = [g + j * NG for g in range(NG) for j in range(G) if g + j * NG < d]
    assert sorted(perm) == list(range(d)), "not a permutation"
    return torch.tensor(perm, dtype=torch.long)


# ---------------------------------------------------------------------------
# Codebook fit.
# ---------------------------------------------------------------------------
def load_layer_k(root: Path, chunks: list[int], l: int, stride: int) -> list[torch.Tensor]:
    """Per-chunk [T, Hkv, d] fp16 keys for layer l, tokens subsampled by
    `stride` (the reference pool kept every 4th token). Kept as a list: the
    reference trainer transforms each example separately before concatenating,
    and matching that keeps the codes (hence k-means) bit-identical to it."""
    return [torch.load(root / f"layer_{l}" / "k" / f"{c}.pt",
                       map_location="cpu", weights_only=False)[::stride]
            for c in chunks]


def fp8_roundtrip_codebooks(codebooks: dict, fmt: str) -> dict:
    """Deploy format: per-group amax scale to the fp8 max, cast, dequantize
    back to fp16 (make_fp8.py's recipe)."""
    dt, fmax = FP8[fmt]
    out = {}
    for lh, cbs in codebooks.items():
        q = []
        for c in cbs:
            c = c.float()
            s = c.abs().amax().clamp_min(1e-12) / fmax
            q.append(((c / s).to(dt).to(torch.float32) * s).half())
        out[lh] = q
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True, help="dump_qkv.py output directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--basis-moments", default=None,
                    help="precomputed second-moments .pt (sigma_q/sigma_k/k_mean/"
                         "k_cov/meta); default: compute from the dump")
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--bpc", type=int, default=2, help="bits per coordinate")
    ap.add_argument("--bits-per-group", type=int, default=None,
                    help="override the per-group bit budget directly (default bpc*G); "
                         "lets a group carry a non-integer bits/coord, e.g. G=8 with 9 "
                         "bits -> 1.125 bpc code + the fp16 per-token rho scale (16/128 "
                         "= 0.125) = 1.25 nominal BPA, matching CQ/TaSQ's 10-bit-over-G=8 "
                         "accounting convention")
    ap.add_argument("--grouping", choices=["consecutive", "stratified"],
                    default="stratified")
    ap.add_argument("--no-pertoken-norm", dest="pertoken_norm", action="store_false",
                    help="disable the per-token RMS scale (canonical builds use it)")
    ap.add_argument("--pool-stride", type=int, default=4,
                    help="token subsampling for the k-means training set")
    ap.add_argument("--iters", type=int, default=25, help="k-means iterations")
    ap.add_argument("--fp8", choices=["e5m2", "e4m3", "none"], default="e5m2",
                    help="fp8 codebook roundtrip; e5m2 is the served default, "
                         "'none' keeps raw float64 centroids")
    ap.add_argument("--seed", type=int, default=0,
                    help="k-means seed base (reference: seed = l*Hkv + h)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.dump)

    n_layers, chunks = dump_layout(root)
    if args.basis_moments:
        B = torch.load(args.basis_moments, map_location="cpu", weights_only=False)
        sigma_q, k_mean, k_cov, meta = B["sigma_q"], B["k_mean"], B["k_cov"], B["meta"]
        assert meta["n_layers"] == n_layers, (
            f"moments have {meta['n_layers']} layers, dump has {n_layers}")
        print(f"basis moments: {args.basis_moments} | ntok={B.get('ntok', '?')}",
              flush=True)
    else:
        sigma_q, _sigma_k, k_mean, k_cov, meta = compute_moments(root, n_layers, chunks)
    L, Hkv, d = meta["n_layers"], meta["n_kv_heads"], meta["d_head"]

    F, inv = build_qpca_basis(sigma_q, k_cov)           # (L,Hkv,d,d) float64
    if args.grouping == "stratified":
        perm = stratified_perm(d, args.G)
        F = F[:, :, :, perm]                            # permute code coords
        inv = inv[:, :, perm, :]
    bounds = group_boundaries(d, args.G, args.bpc)
    if args.bits_per_group is not None:
        assert d % args.G == 0, "--bits-per-group assumes no remainder group"
        bounds = [(s, e, args.bits_per_group) for (s, e, _) in bounds]

    print(f"cfg G={args.G} grouping={args.grouping} bpc={args.bpc} "
          f"ptn={args.pertoken_norm} fp8={args.fp8} | L={L} Hkv={Hkv} d={d} "
          f"NG={len(bounds)} | out={args.out}", flush=True)

    codebooks = {}
    for l in range(L):
        k_l = load_layer_k(root, chunks, l, args.pool_stride)
        for h in range(Hkv):
            r = torch.cat([(kc[:, h].double() - k_mean[l, h].double())
                           @ F[l, h].double() for kc in k_l]).to(dev)  # (N, d)
            if args.pertoken_norm:
                r = r / r.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
            cbs = []
            for (s, e, gb) in bounds:
                K = 1 << gb
                if K <= 1:
                    cbs.append(r[:, s:e].mean(0, keepdim=True))
                else:
                    cbs.append(_kmeans(r[:, s:e], K, iters=args.iters,
                                       seed=args.seed + l * Hkv + h))
            codebooks[(l, h)] = [c.cpu() for c in cbs]
        n = sum(kc.shape[0] for kc in k_l)
        print(f"  layer {l + 1}/{L} ({n} samples/head)", flush=True)

    payload = dict(forward=F.cpu(), inverse=inv.cpu(), mean=k_mean.cpu(),
                   codebooks=codebooks, bounds=bounds, G=args.G,
                   grouping=args.grouping, allocation="flat", whiten=False,
                   ecvq_lambda=0.0, pertoken_norm=args.pertoken_norm,
                   bits_per_coord=float(sum(b for (_, _, b) in bounds)) / d)
    if args.fp8 != "none":
        payload["codebooks"] = fp8_roundtrip_codebooks(codebooks, args.fp8)
        payload["fp8_fmt"] = args.fp8
    torch.save(payload, args.out)
    print(f"SAVED {args.out} | bits/coord = {payload['bits_per_coord']:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
