#!/bin/bash
# Check every built artifact before a paper run: TaSQ must come from the _matching centroids, CQ
# from the plain ones, CQ/TaSQ rotations must be identity while nova's must be real. Each of these
# has silently been wrong at least once, and each produces a plausible-but-lower number rather
# than an error.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
MODEL="${1:?usage: verify_artifacts.sh <hf-model>}"
source "$REPO/config.sh" >/dev/null 2>&1
TAG=$(model_tag "$MODEL"); HF="${MODEL//\//_}"
have_conda; conda activate "$ENV_SERVE"
NSN_CODEBOOK="${NSN_CODEBOOK:-$REPO/assets/nsn/1bit_codebook.pt}"
ARTIFACTS="$ARTIFACTS" WORK="$WORK" TAG="$TAG" HF="$HF" NOVA_CALIB_TAG="$NOVA_CALIB_TAG" MODEL="$MODEL" \
CODEBOOKS="$CODEBOOKS" NSN_CODEBOOK="$NSN_CODEBOOK" NSN_WINDOW="$NSN_WINDOW" python3 - <<'PY'
import os, glob, pickle, numpy as np, torch
A, W, TAG, HF = os.environ["ARTIFACTS"], os.environ["WORK"], os.environ["TAG"], os.environ["HF"]
ok = True
def load(p): return torch.load(p, map_location="cpu", weights_only=False)

# ---------------------------------------------------------------- shapes vs the model config
# The other checks here ask WHICH data a codebook came from; this one asks whether its shape
# matches the model. A V codebook built with the wrong --kv-heads passes every provenance check
# and fails only at boot ("codebook has 8 heads, rank wants [0, 10)"). Llama-3.1-8B and every
# Qwen3 here have exactly 8 KV heads, so only a model like Phi-4-reasoning-plus (10) exposes it.
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(os.environ["MODEL"], trust_remote_code=True)
L_want = cfg.num_hidden_layers
H_want = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
D_want = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
print(f"  model config: layers={L_want} kv_heads={H_want} head_dim={D_want}")

def check_shape(path, key="forward"):
    global ok
    if not os.path.exists(path):
        return
    t = load(path).get(key)
    if t is None or not hasattr(t, "shape") or t.ndim < 3:
        print(f"  {os.path.relpath(path, A):48s} no usable '{key}' tensor  *** WRONG ***"); ok = False; return
    L, H, D = t.shape[0], t.shape[1], t.shape[-1]
    good = (L == L_want and H == H_want and D == D_want)
    ok &= good
    print(f"  {os.path.relpath(path, A):48s} {key} {tuple(t.shape)}  "
          f"{'OK ' if good else f'*** WRONG *** want L={L_want} H={H_want} D={D_want}'}")

CALIB = os.environ.get("NOVA_CALIB_TAG", "calib16")
NOVA10 = "nova1375_" + CALIB   # the bundle used by the NovaKV evaluation arm
for arm in ("cq_g8", "tasq_g8", NOVA10):
    check_shape(f"{A}/{TAG}_{arm}/codebook.pt")
    # nova1375 deliberately ships NO vq_v_codebook.pt: nova1b serves V on the scalar OSCAR
    # tier, which needs no artifact. check_shape returns silently on a missing file, so this
    # is stated rather than left to look like an accident.
    if arm != NOVA10:
        check_shape(f"{A}/{TAG}_{arm}/vq_v_codebook.pt")

cb10 = f"{A}/{TAG}_{NOVA10}/codebook.pt"
if os.path.exists(cb10):
    b = load(cb10)
    lo, hi, bits = b["bounds"][0]
    g, ng = hi - lo, len(b["bounds"])
    good = (g == 8 and bits == 10 and ng * g == D_want)
    ok &= good
    print(f"  {TAG}_{NOVA10}/codebook.pt  G={g} bits/group={bits} NG={ng}  "
          f"{'OK ' if good else '*** WRONG *** want G=8 bits=10'}")

# The nsn bundle carries no layer or head count -- its codebook is a single synthetic table
# shared by every layer and head -- so the only model-specific things in it are head_dim, the
# RoPE frequencies and the model name it was built for. Those are exactly what a cross-model
# mix-up corrupts: Qwen3 uses rope_theta 1e6 against Llama-3.1's and Phi-4's 5e5, so a bundle
# fetched from the wrong directory rotates every key wrongly while looking perfectly well formed.
# The nsn bundle carries no layer or head count -- one synthetic codebook serves every layer and
# head -- so what identifies it is head_dim, the RoPE frequencies and the model it was built for.
# Rather than re-derive inv_freq here, REBUILD the bundle into a temp file with the same builder
# and compare: the derivation is not a one-liner (Llama-3.1 has rope_type "llama3", so its
# inv_freq comes from ROPE_INIT_FUNCTIONS and is NOT theta**(-2i/d); Qwen3-4B-Thinking uses
# rope_theta 5e6 and Phi-4 5e5, and in transformers 5.x none of the three exposes rope_theta at
# the top level any more -- it moved inside rope_scaling/rope_parameters). A copy of that logic
# here would drift from the builder and start reporting false mismatches, which is worse than no
# check. Rebuilding is data-free and takes a couple of seconds.
nsn = f"{A}/{TAG}_nsn_1bit/nsn_bundle.pt"
if os.path.exists(nsn):
    import subprocess, tempfile
    b = load(nsn)
    bad = []
    if b.get("head_dim") != D_want:
        bad.append(f"head_dim: got {b.get('head_dim')!r} want {D_want}")
    if b.get("model_name") != os.environ["MODEL"]:
        bad.append(f"model_name: got {b.get('model_name')!r} want {os.environ['MODEL']!r}")
    with tempfile.TemporaryDirectory() as td:
        ref = os.path.join(td, "ref.pt")
        r = subprocess.run(["python3", os.environ["CODEBOOKS"] + "/build_nsn_bundle.py",
                            "--codebook_path", os.environ["NSN_CODEBOOK"],
                            "--model_name", os.environ["MODEL"],
                            "--window_size", os.environ["NSN_WINDOW"], "--out", ref],
                           capture_output=True, text=True)
        if r.returncode != 0:
            bad.append(f"reference rebuild failed: {r.stderr.strip().splitlines()[-1][:80]}")
        else:
            rb = load(ref)
            for k in ("codec", "n_bits", "head_dim", "window_size", "model_name"):
                if b.get(k) != rb.get(k):
                    bad.append(f"{k}: got {b.get(k)!r} want {rb.get(k)!r}")
            for k in ("codebook", "inv_freq"):
                x, y = b.get(k), rb.get(k)
                if x is None or y is None or x.shape != y.shape or not torch.allclose(
                        x.float(), y.float(), rtol=1e-5, atol=1e-8):
                    bad.append(f"{k}: differs from a fresh build")
    ok &= not bad
    print(f"  {TAG}_nsn_1bit/nsn_bundle.pt   " +
          ("OK  matches a fresh build (codebook, inv_freq, head_dim, window, model_name)"
           if not bad else "*** WRONG *** " + "; ".join(bad)))
else:
    print(f"  {TAG}_nsn_1bit{'':37s} not built")

for arm, want in [("cq_g8", "plain"), ("tasq_g8", "tasq")]:
    cb = f"{A}/{TAG}_{arm}/codebook.pt"
    if not os.path.exists(cb):
        print(f"  {arm:8s} not built"); continue
    b = load(cb)
    bun = np.sort(torch.stack(b["codebooks"][(0, 0)]).float().numpy().ravel()); n = len(bun)
    best, bd = None, None
    for f in glob.glob(f"{W}/{TAG}/quantizers_{HF}_*.pickle"):
        t = pickle.load(open(f, "rb")).get("model.layers.0.self_attn.k_proj")
        if t is None: continue
        d = np.abs(np.sort(t.float().numpy().ravel()[:n]) - bun).max()
        if bd is None or d < bd: best, bd = os.path.basename(f), d
    # Which pickle, not which filename fragment: build_bundles.sh writes CQ's centroids to
    # quantizers_<model>_<width>_<corpus>.pickle and TaSQ's to the same name plus _tasq, so the
    # suffix is the discriminator. TaSQ MUST NOT be served from CQ's pickle -- its codebook has to
    # be fit in the weighted, normalised, permuted space its encoder quantizes in.
    got = "tasq" if best and best.endswith("_tasq.pickle") else "plain"
    flag = "OK " if got == want else "*** WRONG ***"
    ok &= got == want
    print(f"  {arm:8s} centroids <- {best} (max|d|={bd:.2e})  want={want}  {flag}")

for arm, want_identity in [("cq_g8", True), ("tasq_g8", True), (NOVA10, False)]:
    r = f"{A}/{TAG}_{arm}/k_rotation_qqt_r_h_pbr.pt"
    if not os.path.exists(r): continue
    obj = load(r).get("objective")
    is_id = obj == "identity"
    flag = "OK " if is_id == want_identity else "*** WRONG ***"
    ok &= is_id == want_identity
    print(f"  {arm:16s} k_rotation objective={obj}  {flag}")
raise SystemExit(0 if ok else 1)
PY
