"""Two-tier decode read for SGLANG_NSN_PACKED: packed span + raw bf16 tail, merged by LSE.

A served NSN request is `[0, committed)` in the packed store and `[committed, seq_len)` still raw
in the ordinary pool -- the flush loop commits rows 8 at a time once they are RECENT+8 from the
tail, so the split moves every few steps and is per request. `PackedStore.committed_t` is that
watermark, kept on device so this builds both tiers' index lists without a host loop.

The packed tier goes through ``nsn_fused`` and reads the stored 4-bit form directly; the raw tier
uses the stock ``decode_attention_fwd``. Each returns an output and its
log-sum-exp, and the two combine with the standard weighting -- the same structure CQ's
`_unified_stage2` uses across its HP and quant tiers.
"""
from __future__ import annotations

import os

import torch

from sglang.srt.mem_cache import nsn_hp_pool as HPP
from sglang.srt.mem_cache import nsn_packed_store as PS
from sglang.srt.mem_cache import nsn_read_glue as RG

# SGLANG_NSN_PACKED_AUDIT=1: with KEEP_BF16 on, the bf16 pool still holds a valid reconstruction
# for EVERY row, so full-span attention over it is exactly what the un-packed arm computes. This
# recomputes that and reports how far the two-tier packed read is from it, per layer.
AUDIT = os.environ.get("SGLANG_NSN_PACKED_AUDIT") == "1"
_AUD = {"n": 0, "worst": 0.0, "worst_layer": -1}


def _tier_indices(req_to_token, req_idx, lo, hi, fixed_width=None):
    """Flat kv_indices + indptr for the per-request ranges [lo, hi).

    fixed_width: use this instead of the data-dependent max, so no device->host sync happens.
    Required during CUDA-graph capture; the extra rows are masked off by ar < ln."""
    ln = (hi - lo).clamp_min(0)
    B = ln.numel()
    indptr = torch.zeros(B + 1, device=ln.device, dtype=torch.int32)
    indptr[1:] = ln.cumsum(0)
    if fixed_width is not None:
        # Static-shape build for CUDA-graph capture. `rows[mask]` has a data-dependent size
        # and needs a sync, which is the cudaErrorStreamCaptureUnsupported this replaced.
        # Instead: a [B*W+1] buffer, every (b, j) scattered to indptr[b]+j when valid and to
        # the trailing trash entry otherwise. Only [indptr[b], indptr[b+1]) is ever read, so
        # shapes are static and the VALUES track the real lengths on every replay -- the same
        # contract SGLang's own decode buffers use.
        W = fixed_width
        ar = torch.arange(W, device=ln.device)
        valid = ar[None, :] < ln[:, None]                               # [B, W]
        pos = (lo[:, None] + ar[None, :]).clamp_(0, req_to_token.shape[1] - 1)
        rows = torch.gather(req_to_token[req_idx], 1, pos)             # [B, W]
        flat = indptr[:-1].to(torch.long)[:, None] + ar[None, :]       # [B, W]
        trash = B * W
        dst = torch.where(valid, flat, torch.full_like(flat, trash))
        buf = torch.zeros(B * W + 1, device=ln.device, dtype=torch.int32)
        buf.scatter_(0, dst.reshape(-1), rows.reshape(-1).to(torch.int32))
        return buf[: B * W].contiguous(), indptr, ln
    mx = int(ln.max().item()) if B else 0
    if mx == 0:
        return torch.empty(0, device=ln.device, dtype=torch.int32), indptr, ln
    ar = torch.arange(mx, device=ln.device)
    pos = (lo[:, None] + ar[None, :]).clamp_(0, req_to_token.shape[1] - 1)
    rows = torch.gather(req_to_token[req_idx], 1, pos)
    return rows[ar[None, :] < ln[:, None]].to(torch.int32).contiguous(), indptr, ln


def _merge(o1, l1, o2, l2):
    m = torch.maximum(l1, l2)
    w1, w2 = torch.exp(l1 - m), torch.exp(l2 - m)
    return (o1 * w1[..., None] + o2 * w2[..., None]) / (w1 + w2)[..., None]


def _cb_for(codebook, KVH):
    # (KVH,) rather than the codebook's data_ptr, and never allocated during capture -- see
    # nsn_encode_kernel._consts for what a graph-pool constant does to this arm.
    key = ("cb", KVH)
    if key not in _CB and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("nsn_packed_read._cb_for: first allocation inside CUDA-graph "
                           "capture; build it before capture (memory_pool prewarm).")
    if key not in _CB:
        _CB[key] = codebook.float().unsqueeze(0).expand(KVH, -1, -1).contiguous().half()
    return _CB[key]


_CB: dict = {}
GLUE = os.environ.get("SGLANG_NSN_READ_GLUE", "1") == "1"
RAW_FOLD = os.environ.get("SGLANG_NSN_RAW_FOLD", "1") == "1"
# SGLANG_NSN_AUDIT_GRAPH=1 audits CUDA-graph execution with tensor-only bookkeeping. It requires
# KEEP_BF16=1 and a residual ring that covers the context.
AUDIT_GRAPH = os.environ.get("SGLANG_NSN_AUDIT_GRAPH") == "1"
AUDIT_LAYERS = int(os.environ.get("SGLANG_NSN_AUDIT_LAYERS", "4"))
_AUD_T = {}


def _zeros_like_cached(t):
    z = _AUD_T.get("z")
    if z is None or z.shape != t.shape or z.dtype != t.dtype:
        z = torch.zeros_like(t)
        _AUD_T["z"] = z
    return z


def audit_tensor(nlayers, device):
    t = _AUD_T.get("t")
    if (t is None or t.numel() < nlayers) and torch.cuda.is_current_stream_capturing():
        # allocated inside capture -> lives in that graph's private pool -> the eager reader
        # sees zeros and the audit silently reports "no error at all" (observed 2026-09-10)
        raise RuntimeError("nsn_packed_read.audit_tensor: allocate before capture "
                           "(memory_pool prewarm calls it).")
    if t is None or t.numel() < nlayers:
        t = torch.zeros(max(nlayers, 64), device=device, dtype=torch.float32)
        _AUD_T["t"] = t
    return t


def audit_report(tag=""):
    t = _AUD_T.get("t")
    if t is None:
        return
    v = t.tolist()
    nz = [(i, round(x, 5)) for i, x in enumerate(v) if x > 0]
    print(f"[nsn audit-graph{tag}] worst rel per layer: {nz[:40]} | max="
          f"{max(v) if v else 0:.4e}", flush=True)
    t.zero_()


@torch.no_grad()
def _audit_graph(layer_idx, o, q, kb, vb, r2t, req_idx, seq, ps, _hp, ws, MAXW,
                 decode_fwd, md, max_kv_splits, layer, k_descale, v_descale):
    if not AUDIT_GRAPH or layer_idx >= AUDIT_LAYERS:
        return
    W = r2t.shape[1]
    bf, _ = RG.read_indices(r2t, req_idx, seq, _zeros_like_cached(ps.committed_t), _hp.map,
                            _hp.compact, W, ws, MAXW, _hp.trash, prefix=0, tag="audit")
    o_full = torch.empty_like(o)
    decode_fwd(q, kb, vb, o_full, bf["r_ptr"], bf["r_ind"], md.attn_logits, md.attn_lse,
               md.num_kv_splits, max_kv_splits, layer.scaling, k_descale, v_descale)
    d = (o.float() - o_full.float()).norm() / o_full.float().norm().clamp_min(1e-30)
    t = audit_tensor(layer_idx + 1, o.device)
    t[layer_idx] = torch.maximum(t[layer_idx], d)
RAW_SPLITS = int(os.environ.get("SGLANG_NSN_RAW_SPLITS", "2"))


@torch.no_grad()
def _torch_glue(ls, q, o, layer, forward_batch, bundle, decode_fwd, md, max_kv_splits, hadamard,
                k_descale, v_descale, splits, req_idx, seq, r2t, B, Hq, D, dev, _hp, ws, ps):
    """The original torch glue (reference; SGLANG_NSN_READ_GLUE=0). Returns merged fp32 or
    (None, ...) after writing `o` when the packed span is empty. PREFIX=0 only."""
    from sglang.srt.mem_cache.nsn_fused_kernel import nsn_fused

    assert _hp is None or _hp.P == 0, "torch read glue does not implement PREFIX; use the kernel glue"

    nq = (ps.committed_t[req_idx] if ls is not None else torch.zeros_like(seq))
    nq = torch.minimum(nq.to(torch.int32), seq)
    zero = torch.zeros_like(seq)
    # Shapes are static ALWAYS, not only under capture. Deriving the width from the data when
    # not capturing made the graph runner's pre-capture warmup allocate different buffers from
    # the ones capture then asks for -- the read-path twin of the branch that made the write
    # path capture-incompatible. Keeping one width means sglang's own warmup covers NSN exactly
    # as it covers every other arm, with no NSN-specific prewarming. The kernels mask per
    # request, so the wider span costs a little eager-decode work and changes no result.
    _W = r2t.shape[1]
    _cap = torch.cuda.is_current_stream_capturing()
    # --- raw tail, always present
    r_ind, r_ptr, _ = _tier_indices(r2t, req_idx, nq, seq, fixed_width=_W)
    # the bf16 tier is small and slot-indexed only through the indirection
    if _hp is not None and r_ind.numel():
        r_ind = _hp.lookup(r_ind.long()).long().contiguous()
    o_raw = torch.empty((B, Hq, D), device=dev, dtype=q.dtype)
    lse_raw = torch.empty((B, Hq), device=dev, dtype=torch.float32)
    decode_fwd(q, forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
               forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
               o_raw, r_ptr, r_ind, md.attn_logits, md.attn_lse, md.num_kv_splits,
               max_kv_splits, layer.scaling, k_descale, v_descale, output_lse=lse_raw)

    if ls is None or (not _cap and int(nq.max().item()) == 0):
        o.copy_(o_raw)
        return None, None, None, None, None, None, None, None

    # --- packed span
    p_ind, p_ptr, _ = _tier_indices(r2t, req_idx, zero, nq, fixed_width=_W)
    maxq = _W if _cap else int(nq.max().item())
    nw = (maxq + ws - 1) // ws
    # sm_scale. The raw tier gets it as an argument; nsn_fused has no such parameter and
    # produces RAW logits, so it goes into the query instead -- every logit term is linear in q,
    # so scaling q scales all of them identically. Without this the two tiers' logits are on
    # different scales: the packed softmax is far too peaked AND the LSE merge weights are
    # wrong, which is an O(1) error and is invisible in any test where the packed span is empty.
    qf = (q.float() * layer.scaling).contiguous()
    hq = hadamard(qf.reshape(B * Hq, 1, D)).reshape(B, Hq, D).contiguous()
    KVH = ls.idx_k.shape[1]
    cb = bundle["codebook"].float().unsqueeze(0).expand(KVH, -1, -1).contiguous().half()
    # Window-arena rows by the SAME arithmetic the fused flush commits with
    # (nsn_quant_fused._packed_commit): compact[req] * MAXW + ordinal. No free-list table, so
    # the read stops depending on host-side allocator state -- required for graph capture.
    from sglang.srt.mem_cache.nsn_quant_fused import MAXW

    _ring = _hp
    _comp = (_ring.compact[req_idx].to(torch.long) if _ring is not None
             else req_idx.to(torch.long))
    wmap = (_comp[:, None] * MAXW
            + torch.arange(nw, device=dev)[None, :]).to(torch.int32).contiguous()
    nat = dict(n2=ls.n2, nrm8=ls.nrm8, nsc=ls.nsc, mk_q=ls.mk_q, mv_q=ls.mv_q,
               mk_s=ls.mk_s, mv_s=ls.mv_s, win_map=wmap)
    o_pk, lse_pk = nsn_fused(
        hq, qf, ls.idx_k, ls.idx_v, cb, None, None, None, None, None, None, None,
        p_ptr, p_ind, torch.zeros(B, device=dev, dtype=torch.int32), maxq, ws,
        splits=splits, dot_f16=True, hadamard=hadamard, inv_freq=bundle["inv_freq"],
        native=nat, nw_span=nw, return_lse=True)

    # requests with an empty packed span must not contribute
    lse_pk = torch.where((nq > 0)[:, None], lse_pk, torch.full_like(lse_pk, float("-inf")))
    merged = _merge(o_pk, lse_pk, o_raw.float(), lse_raw)
    return merged, o_raw, lse_raw, o_pk, lse_pk, p_ind, p_ptr, nq


@torch.no_grad()
def two_tier_decode(layer_idx, q, o, layer, forward_batch, bundle, decode_fwd, md,
                    max_kv_splits, hadamard, k_descale=None, v_descale=None,
                    splits=16):
    """q [B, Hq, D] (model dtype); writes the merged attention output into `o` [B, Hq, D]."""
    from sglang.srt.mem_cache.nsn_fused_kernel import nsn_fused

    ps = PS.store()
    ls = ps.layers.get(layer_idx)
    B, Hq, D = q.shape
    dev = q.device
    req_idx = forward_batch.req_pool_indices
    seq = forward_batch.seq_lens.to(torch.int32)
    r2t = forward_batch.req_to_token_pool.req_to_token

    _hp = HPP.hp()
    ws = bundle["window_size"]
    if ls is not None and _hp is not None and GLUE:
        # ---- kernel glue (nsn_read_glue): both tiers' indices, split point and window map in
        # one launch; LSE merge in one. Static shapes (W = the pool's context bound), no syncs.
        from sglang.srt.mem_cache.nsn_quant_fused import MAXW

        W = r2t.shape[1]
        bf, nw = RG.read_indices(r2t, req_idx, seq, ps.committed_t, _hp.map, _hp.compact, W, ws,
                                 MAXW, _hp.trash, prefix=_hp.P)
        nq = bf["nq"]
        kb = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        vb = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        if not RAW_FOLD:
            o_raw = torch.empty((B, Hq, D), device=dev, dtype=q.dtype)
            lse_raw = torch.empty((B, Hq), device=dev, dtype=torch.float32)
            decode_fwd(q, kb, vb, o_raw, bf["r_ptr"], bf["r_ind"], md.attn_logits, md.attn_lse,
                       md.num_kv_splits, max_kv_splits, layer.scaling, k_descale, v_descale,
                       output_lse=lse_raw)
        qf = (q.float() * layer.scaling).contiguous()
        hq = hadamard(qf.reshape(B * Hq, 1, D)).reshape(B, Hq, D).contiguous()
        cb = _cb_for(bundle["codebook"], ls.idx_k.shape[1])
        nat = dict(n2=ls.n2, nrm8=ls.nrm8, nsc=ls.nsc, mk_q=ls.mk_q, mv_q=ls.mv_q,
                   mk_s=ls.mk_s, mv_s=ls.mv_s, win_map=bf["wmap"])
        p_ind, p_ptr = bf["p_ind"], bf["p_ptr"]
        if RAW_FOLD:
            # raw tail read by the same launch pair; k/v descale are 1 for a bf16 pool
            raw = dict(q=q.contiguous(), k=kb, v=vb, indptr=bf["r_ptr"], indices=bf["r_ind"],
                       splits=RAW_SPLITS, sm_scale=layer.scaling)
            out = nsn_fused(
                hq, qf, ls.idx_k, ls.idx_v, cb, None, None, None, None, None, None, None,
                p_ptr, p_ind, bf["mean_idx"], W, ws,
                splits=splits, dot_f16=True, hadamard=hadamard, inv_freq=bundle["inv_freq"],
                native=nat, nw_span=nw, raw=raw, pos_offset=_hp.P)
            o.copy_(out)
            _audit_graph(layer_idx, o, q, kb, vb, r2t, req_idx, seq, ps, _hp, ws, MAXW,
                         decode_fwd, md, max_kv_splits, layer, k_descale, v_descale)
            if not AUDIT:
                return o
            merged = o.float()
            # audit diagnostics below expect the split tensors; recompute them the slow way
            o_raw = torch.empty((B, Hq, D), device=dev, dtype=q.dtype)
            lse_raw = torch.empty((B, Hq), device=dev, dtype=torch.float32)
            decode_fwd(q, kb, vb, o_raw, bf["r_ptr"], bf["r_ind"], md.attn_logits, md.attn_lse,
                       md.num_kv_splits, max_kv_splits, layer.scaling, k_descale, v_descale,
                       output_lse=lse_raw)
            o_pk, lse_pk = nsn_fused(
                hq, qf, ls.idx_k, ls.idx_v, cb, None, None, None, None, None, None, None,
                p_ptr, p_ind, bf["mean_idx"], W, ws,
                splits=splits, dot_f16=True, hadamard=hadamard, inv_freq=bundle["inv_freq"],
                native=nat, nw_span=nw, return_lse=True, pos_offset=_hp.P)
        else:
            o_pk, lse_pk = nsn_fused(
                hq, qf, ls.idx_k, ls.idx_v, cb, None, None, None, None, None, None, None,
                p_ptr, p_ind, bf["mean_idx"], W, ws,
                splits=splits, dot_f16=True, hadamard=hadamard, inv_freq=bundle["inv_freq"],
                native=nat, nw_span=nw, return_lse=True, pos_offset=_hp.P)
            RG.merge_into(o, o_pk, lse_pk, o_raw, lse_raw, nq)
            _audit_graph(layer_idx, o, q, kb, vb, r2t, req_idx, seq, ps, _hp, ws, MAXW,
                         decode_fwd, md, max_kv_splits, layer, k_descale, v_descale)
            if not AUDIT:
                return o
        merged = o.float()
        lse_pk = torch.where((nq > 0)[:, None], lse_pk, torch.full_like(lse_pk, float("-inf")))
    else:
        merged, o_raw, lse_raw, o_pk, lse_pk, p_ind, p_ptr, nq = _torch_glue(
            ls, q, o, layer, forward_batch, bundle, decode_fwd, md, max_kv_splits, hadamard,
            k_descale, v_descale, splits, req_idx, seq, r2t, B, Hq, D, dev, _hp, ws, ps)
        if merged is None:
            return o
    if AUDIT:
        f_ind, f_ptr, _ = _tier_indices(r2t, req_idx, torch.zeros_like(seq), seq)
        # the bf16 tier is ring-addressed: pool slots go through the same map the raw tier uses
        if _hp is not None:
            f_ind = _hp.lookup(f_ind.long()).contiguous()
        o_full = torch.empty_like(o_raw)
        decode_fwd(q, forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                   forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                   o_full, f_ptr, f_ind, md.attn_logits, md.attn_lse, md.num_kv_splits,
                   max_kv_splits, layer.scaling, k_descale, v_descale)
        rel = ((merged - o_full.float()).norm()
               / o_full.float().norm().clamp_min(1e-30)).item()
        _AUD["n"] += 1
        if rel > _AUD["worst"]:
            _AUD["worst"], _AUD["worst_layer"] = rel, layer_idx
        if rel > 0.05:
            # split the blame: packed tier alone vs the same span read from the bf16 pool
            o_sp = torch.empty_like(o_raw)
            lse_sp = torch.empty((B, Hq), device=dev, dtype=torch.float32)
            sp_ind = _hp.lookup(p_ind.long()).contiguous() if _hp is not None else p_ind
            decode_fwd(q, forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                       forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                       o_sp, p_ptr, sp_ind, md.attn_logits, md.attn_lse, md.num_kv_splits,
                       max_kv_splits, layer.scaling, k_descale, v_descale, output_lse=lse_sp)
            rl = lambda a_, b_: ((a_ - b_).norm() / b_.norm().clamp_min(1e-30)).item()
            print(f"[nsn_packed audit] OUTLIER rel={rel:.3e} layer={layer_idx} "
                  f"nq={nq.tolist()} seq={seq.tolist()} call={_AUD['n']} | "
                  f"packed-tier {rl(o_pk, o_sp.float()):.3e} "
                  f"lse {rl(lse_pk, lse_sp):.3e} "
                  f"raw-tier-lse {lse_raw.mean().item():.3f} pk-lse {lse_pk.mean().item():.3f}",
                  flush=True)
        if _AUD["n"] % 200 == 0:
            print(f"[nsn_packed audit] calls={_AUD['n']} worst_rel={_AUD['worst']:.3e} "
                  f"(layer {_AUD['worst_layer']}) last={rel:.3e} "
                  f"nq={nq.tolist()} seq={seq.tolist()}", flush=True)
    o.copy_(merged.to(o.dtype))
    return o
