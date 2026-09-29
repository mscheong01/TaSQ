"""Compute per-layer channel weights and grouping permutations for TaSQ.

The calibration follows the encoder order:

  W  M_h = E over valid causal (t, p<=t) query-key pairs of R_p^T q_t q_t^T R_p; W_h = diag(M_h),
     and the key is transformed as k^w = W^{1/2} k. Stored as w = sqrt(diag(M)/mean(diag(M))),
     so mean(w^2) = 1 and k^w = diag(w) k.
  N  pooled RMS over ALL KV heads and channels of the WEIGHTED key, one scalar per token:
     s_p = RMS(k^w_p),  k^wn = k^w / s_p. The same pooling the served encoder does.
  P  min-weight matching over Cov(k^wn) -- the covariance of the vector VQ actually sees, so no
     separate "sandwich" reconstruction exists to get wrong. RoPE pairs {j, j+d/2} stay atomic.

Two passes are used: the first estimates query-guided weights, and the second computes
``Cov(k^wn)`` after applying weighting and cross-head normalization.
"""
# The rotary helpers below are adapted from NSNQuant/src/utils.py and Hugging Face Transformers'
# modeling_llama.py under their respective licenses.
import argparse, itertools, os, pickle, random

import networkx as nx
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


# --- rotate_half / apply_rotary_pos_emb_single: verbatim from NSNQuant/src/utils.py, which in
# --- turn derives from HuggingFace transformers modeling_llama.py (Apache-2.0). Inlined here so
# --- no dependency on an NSNQuant checkout. ---
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_single(v, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (v * cos) + (rotate_half(v) * sin)


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
ap.add_argument("--nsamples", type=int, default=64)
ap.add_argument("--seqlen", type=int, default=2048)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--coupled", type=int, default=8, help="channels per VQ group")
ap.add_argument("--out", required=True)
ap.add_argument("--dataset", default="gpqa_code",
                choices=["wikitext2", "gpqa", "gpqa_code",
                         "gpqa_diamond", "gpqa_diamond_windowed", "code"],
                help="calibration corpus -- must match the codebook's --dataset "
                     "(llama_simquant.py) so partition, weights and codebooks see identical data")
ap.add_argument("--mix-code-n", type=int, default=16,
                help="gpqa_code only: how many of --nsamples windows come from real Python "
                     "(codeparrot-clean). A COUNT, not a ratio.")
a = ap.parse_args()

DEV = "cuda"
tok = AutoTokenizer.from_pretrained(a.model, use_fast=False)


def windows(text, n, seqlen, seed):
    enc = tok(text, return_tensors="pt")
    random.seed(seed)
    return [enc.input_ids[:, (i := random.randint(0, enc.input_ids.shape[1] - seqlen - 1)):i + seqlen]
            for _ in range(n)]


if a.dataset == "gpqa_diamond":
        # NovaKV's own calibration recipe, verified against calibration/dump_qkv.py
        # directly: 198 individual GPQA-DIAMOND prompts (not gpqa_main), each wrapped in that
        # script's GPQA_TMPL plus the model's chat template and run as its own sequence -- no
        # windowing or concatenation.
    GPQA_TMPL = (
        "Answer the following multiple choice question. The last line of your response "
        "should be of the following format: 'Answer: $LETTER' (without quotes) where "
        "LETTER is one of ABCD. Think step by step before answering.\n\n{Question}\n\n"
        "A) {A}\nB) {B}\nC) {C}\nD) {D}"
    )
    gpd = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    n = a.nsamples if a.nsamples and a.nsamples > 0 else len(gpd)
    BATCH = []
    for ex in list(gpd)[:n]:
        p = GPQA_TMPL.format(Question=ex["Question"], A=ex["Correct Answer"],
                             B=ex["Incorrect Answer 1"], C=ex["Incorrect Answer 2"],
                             D=ex["Incorrect Answer 3"])
        txt = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, tokenize=False)
        ids = tok(txt, return_tensors="pt", add_special_tokens=False).input_ids
        BATCH.append(ids[:, :a.seqlen] if a.seqlen else ids)
elif a.dataset == "gpqa_diamond_windowed":
    gpd = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    text = "\n\n".join(f"Question: {ex['Question']}\n\nExplanation: {ex['Explanation']}\n\n"
                       f"Answer: {ex['Correct Answer']}" for ex in gpd)
    BATCH = windows(text, a.nsamples, a.seqlen, a.seed)
elif a.dataset == "gpqa_code":
    gp = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
    gpqa_text = "\n\n".join(f"Question: {ex['Question']}\n\nExplanation: {ex['Explanation']}\n\n"
                            f"Answer: {ex['Correct Answer']}" for ex in gp)
    code_ds = load_dataset("codeparrot/codeparrot-clean", streaming=True, split="train")
    code_text = "\n\n".join(ex["content"] for i, ex in zip(range(300), code_ds))
    n_gpqa = max(0, a.nsamples - a.mix_code_n)
    BATCH = (windows(gpqa_text, n_gpqa, a.seqlen, a.seed)
             + windows(code_text, a.mix_code_n, a.seqlen, a.seed))
elif a.dataset == "gpqa":
    gp = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
    text = "\n\n".join(f"Question: {ex['Question']}\n\nExplanation: {ex['Explanation']}\n\n"
                       f"Answer: {ex['Correct Answer']}" for ex in gp)
    BATCH = windows(text, a.nsamples, a.seqlen, a.seed)
elif a.dataset == "code":
    # CodeParrot alone -- the single-corpus counterpart of the gpqa_code mixture's code half.
    # Same 300 streamed documents and same window builder the mixture uses, so a code-only
    # calibration differs from the mixture's code portion only in how many windows it draws.
    code_ds = load_dataset("codeparrot/codeparrot-clean", streaming=True, split="train")
    code_text = "\n\n".join(ex["content"] for i, ex in zip(range(300), code_ds))
    BATCH = windows(code_text, a.nsamples, a.seqlen, a.seed)
elif a.dataset == "wikitext2":
    tr = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    BATCH = windows("\n\n".join(tr["text"]), a.nsamples, a.seqlen, a.seed)
else:
    # Previously this was a bare `else: wikitext2`, so a corpus name this file does not know
    # calibrated TaSQ on wikitext while every other step used the requested corpus, with nothing
    # in the log to say so. Unknown names now stop.
    raise SystemExit(
        f"make_tasq.py: unknown --dataset {a.dataset!r}. Add a branch here; this script builds "
        "its own calibration windows and does not share kvquant/datautils.py's table.")

model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.half,
                                             device_map=DEV).eval()
# Fused-QKV architectures (Phi-4-reasoning-plus, anything packing qkv_proj): this chain reaches
# for self_attn.{q,k,v}_proj by name, so a shim installs real slice Linears and reroutes the fused
# module through them (calibration/shims/phi4/phi3_shim.py). No-op on llama/qwen3 -- the guard is the absence of
# k_proj, not a model-name match.
import sys as _sys  # noqa: E402
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration", "shims", "phi4"))
from inject import maybe_unfuse as _maybe_unfuse  # noqa: E402
_maybe_unfuse(model)

cfg = model.config
L = cfg.num_hidden_layers
NQ, NKV = cfg.num_attention_heads, cfg.num_key_value_heads
NREP = NQ // NKV
D = getattr(cfg, "head_dim", cfg.hidden_size // NQ)
HALF = D // 2
C = a.coupled
assert C % 2 == 0 and D % C == 0, f"--coupled {C} must be even and divide head_dim {D}"

buf = {}


def hook(li, nm, nh, norm_mod=None):
    def h(_, i, o):
        # Qwen3 (any arch with QK-RMSNorm) applies q_norm/k_norm to this per-head-reshaped tensor
        # before RoPE -- see src/models/qwen3.py. The deployed quantizer never sees pre-norm
        # activations, so the statistics must not either.
        reshaped = o.view(1, -1, nh, D)
        if norm_mod is not None:
            reshaped = norm_mod(reshaped)
        buf[(li, nm)] = reshaped.transpose(1, 2)[0].detach()
    return h


def install_hooks(which):
    hs = []
    for li in range(L):
        s = model.model.layers[li].self_attn
        if "k" in which:
            hs.append(s.k_proj.register_forward_hook(hook(li, "k", NKV, getattr(s, "k_norm", None))))
        if "q" in which:
            hs.append(s.q_proj.register_forward_hook(hook(li, "q", NQ, getattr(s, "q_norm", None))))
    return hs


# ---------------------------------------------------------------- pass 1: W, and the head scale
#
# M_h = E over valid causal pairs of R_p^T q_t q_t^T R_p. Summed over key positions this is
#   sum_p R_p^T S_p R_p     with     S_p = sum_{t>=p} q_t q_t^T   (a SUFFIX sum),
# and R_p is block-diagonal on the RoPE pair {c, c+D/2}, so its diagonal needs only the suffix
# sums of the per-token diagonal and pair-cross moments -- no D x D matrix, one pass:
#   diag_c = E_p[ cos^2 S_p[c,c] + sin^2 S_p[pc,pc] + 2 sigma_c cos sin S_p[c,pc] ]
# The (T-p) weight this carries is part of the definition: early keys are seen by more queries.
mkc_num = torch.zeros(L, NKV, D, dtype=torch.double, device=DEV)
mkc_den = torch.zeros((), dtype=torch.double, device=DEV)
kss = torch.zeros(L, NKV, D, dtype=torch.double, device=DEV)   # for the head-order RMS
ktok = 0.0
SGN = torch.cat([torch.ones(HALF), -torch.ones(HALF)]).to(DEV).double()
_checked = False

hs = install_hooks("kq")
print(f"[pass 1/2] causal query moments over {len(BATCH)} windows", flush=True)
with torch.no_grad():
    for bi, b in enumerate(BATCH):
        b = b.to(DEV)
        pos = torch.arange(b.shape[1], device=DEV).unsqueeze(0)
        cs, sn = model.model.rotary_emb(model.model.embed_tokens(b), pos)
        model(b, use_cache=False)
        # rotary_emb returns (B, T, D) on every model here, but the guard is free and a
        # 2-D return would otherwise slice the TIME axis and corrupt W silently.
        cosp = (cs[0] if cs.dim() == 3 else cs).double()                  # (T, D)
        sinp = (sn[0] if sn.dim() == 3 else sn).double()
        c2, s2, csx = cosp ** 2, sinp ** 2, cosp * sinp
        for li in range(L):
            k = buf[(li, "k")].double()                                   # (NKV, T, D)
            kss[li] += (k ** 2).sum(1)
            q = apply_rotary_pos_emb_single(buf[(li, "q")].unsqueeze(0), cs, sn)[0]
            qt = q.view(NKV, NREP, -1, D).double()                        # (NKV, NREP, T, D)
            T = qt.shape[2]
            d_t = (qt ** 2).sum(1) / NREP                                 # (NKV, T, D)
            x_t = (qt[..., :HALF] * qt[..., HALF:]).sum(1) / NREP         # (NKV, T, HALF)
            Sd = torch.flip(torch.cumsum(torch.flip(d_t, [1]), 1), [1])   # suffix sums over t
            Sx = torch.flip(torch.cumsum(torch.flip(x_t, [1]), 1), [1])
            Sd_sw = torch.cat([Sd[..., HALF:], Sd[..., :HALF]], -1)       # S[p, pc]
            Sx_f = torch.cat([Sx, Sx], -1)                                # S_x[p, pair]
            num = (c2.unsqueeze(0) * Sd + s2.unsqueeze(0) * Sd_sw
                   + 2.0 * SGN * csx.unsqueeze(0) * Sx_f).sum(1)
            mkc_num[li] += num
            if li == 0:
                mkc_den += torch.arange(T, 0, -1, device=DEV, dtype=torch.double).sum()
                ktok += T
                if not _checked:
                    # Brute force on head 0 of this window. A wrong sign, pair map or suffix
                    # direction yields a plausible W and a silently different codebook, so this
                    # fails loudly rather than shipping one.
                    _checked = True
                    O = torch.einsum("rtd,rte->tde", qt[0], qt[0]) / NREP   # (T, D, D)
                    S = torch.flip(torch.cumsum(torch.flip(O, [0]), 0), [0])
                    Sm = torch.zeros(D, D, dtype=torch.double, device=DEV)
                    idx = torch.arange(HALF)
                    Sm[idx, idx + HALF] = -1.0
                    Sm[idx + HALF, idx] = 1.0
                    Rm = torch.diag_embed(cosp) + sinp.unsqueeze(-1) * Sm.unsqueeze(0)
                    ref = torch.einsum("pdc,pde,pec->c", Rm, S, Rm)
                    rel = (num[0] - ref).abs().max() / ref.abs().max()
                    print(f"  [check] causal diag(M): closed form vs full einsum, max rel "
                          f"{rel:.3e}", flush=True)
                    assert rel < 1e-6, "causal diag(M) closed form disagrees with the einsum"
        if (bi + 1) % 8 == 0:
            print(f"  {bi + 1}/{len(BATCH)}", flush=True)
for h in hs:
    h.remove()

# W, once, from one expression. A RELATIVE floor, not an absolute constant: the basis is a
# query-energy scale that varies by orders of magnitude across models and layers.
basis = (mkc_num / mkc_den).cpu().numpy()                                 # (L, NKV, D)
weps = np.clip(basis.max(-1, keepdims=True) * 1e-6, 1e-30, None)
Wsqrt = np.sqrt(np.clip(basis, weps, None)
                / np.clip(basis, weps, None).mean(-1, keepdims=True))     # mean(w^2) = 1

# ---------------------------------------------------------------- pass 2: Cov(k^wn)
#
# The ONLY place the grouping covariance is formed, and it is formed from the transformed vector
# itself -- there is no second weighting step here that could disagree with Wsqrt above.
cov = torch.zeros(L, NKV, D, D, dtype=torch.double, device=DEV)
csum = torch.zeros(L, NKV, D, dtype=torch.double, device=DEV)
cn = 0.0
Wt = torch.as_tensor(Wsqrt, dtype=torch.double, device=DEV)               # (L, NKV, D)

hs = install_hooks("k")
print("[pass 2/2] Cov of the weighted, pooled-normalised key", flush=True)
with torch.no_grad():
    for bi, b in enumerate(BATCH):
        model(b.to(DEV), use_cache=False)
        for li in range(L):
            k = buf[(li, "k")].double()                                   # (NKV, T, D)
            kw = k * Wt[li].unsqueeze(1)                                  # [W]  k^w
            s = kw.pow(2).mean(dim=(0, 2), keepdim=True).sqrt().clamp_min(1e-8)   # (1,T,1)
            z = kw / s                                                    # [N]  k^wn
            cov[li] += torch.einsum("htd,hte->hde", z, z)
            csum[li] += z.sum(1)
            if li == 0:
                cn += z.shape[1]
        if (bi + 1) % 8 == 0:
            print(f"  {bi + 1}/{len(BATCH)}", flush=True)
for h in hs:
    h.remove()

mean_z = (csum / cn).cpu().numpy()
Sigma = (cov / cn).cpu().numpy() - np.einsum("lhd,lhe->lhde", mean_z, mean_z)


# ---------------------------------------------------------------- P
def det_cost(S, idx, eps):
    """det(S[idx,idx] + eps I)^(1/|idx|), via slogdet -- the |idx|-th root keeps costs comparable
    across matching rounds despite the growing submatrix."""
    sub = S[np.ix_(idx, idx)] + eps * np.eye(len(idx))
    sign, logdet = np.linalg.slogdet(sub)
    if sign <= 0:
        return np.inf     # not PSD (numerical noise) -- never prefer merging these
    return float(np.exp(logdet / len(idx)))


def match_groups(S, coupled, eps):
    """Group the D/2 RoPE pairs into D/coupled groups of `coupled` channels by iterated
    min-weight perfect matching, doubling the unit size each round. Pairs stay atomic throughout,
    so the decoder's rotation is always local to one codeword. A heuristic decomposition of the
    k-way balanced partition, not an exact solver."""
    units = [[p, p + HALF] for p in range(HALF)]
    for _ in range(int(np.log2(coupled // 2))):
        n = len(units)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i, j in itertools.combinations(range(n), 2):
            G.add_edge(i, j, weight=det_cost(S, units[i] + units[j], eps))
        m = nx.min_weight_matching(G, weight="weight")
        assert len(m) == n // 2, "expected a perfect matching on a complete graph"
        units = [units[i] + units[j] for i, j in m]
    return units



perms, ws, horder = {}, {}, {}
for li in range(L):
    P = np.zeros((NKV, D), dtype=np.int64)
    for h in range(NKV):
        S = Sigma[li, h]
        eps = np.clip(np.diag(S).mean() * 1e-4, 1e-30, None)
        groups = match_groups(S, C, eps)
        P[h] = np.concatenate([np.array(g) for g in groups])
        assert sorted(P[h].tolist()) == list(range(D))
        for g in groups:
            gs = set(g)
            assert all((j + HALF) in gs for j in gs if j < HALF), "RoPE pair split"
    perms[li] = P
    ws[li] = Wsqrt[li].astype(np.float32)
    # Head order for the cross-head shared scale: sort by the per-head RMS of the WEIGHTED key,
    # so heads of similar magnitude end up sharing a scale.
    rms = np.sqrt((kss[li].cpu().numpy() / ktok) * (Wsqrt[li] ** 2)).mean(-1)
    horder[li] = np.argsort(rms).astype(np.int64)
    if (li + 1) % 8 == 0:
        print(f"  layer {li + 1}/{L} grouped", flush=True)

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
pickle.dump(dict(model=a.model, nkv=NKV, d=D, coupled=C,
                 perms=perms, weights=ws, head_order=horder,
                 normalization="cross_head_shared_rms",
                 normalization_heads=NKV, dataset=a.dataset,
                 # Recorded so a pickle states the method it was built under instead of relying
                 # on its filename. There is exactly one method; these are constants.
                 w_source="mkc", p_source="matching",
                 cov_space="Cov(k^wn): weighted, then pooled-normalised"),
            open(a.out, "wb"))
LI = min(16, L - 1)
print(f"\n[done] {L} layers -> {a.out}")
print(f"  L{LI} h0 perm[:8]   = {perms[LI][0][:8].tolist()}")
print(f"  L{LI} h0 w  min/max = {ws[LI][0].min():.3f} / {ws[LI][0].max():.3f}")
