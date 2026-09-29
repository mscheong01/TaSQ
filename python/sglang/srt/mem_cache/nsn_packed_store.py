"""Packed NSNQuant KV storage for the served path.

This module stores the quantized fields directly at 1.2383 bits per channel and side.

Unlike ``nsn_pack``'s window layout, the served pool is token-granular and paged,
so a 4-bit per-token field cannot be nibble-packed along the window axis -- a slot is one token
and may sit anywhere. Packing across SIDES works instead: K's and V's norm codes are both 4 bits
for the same token, so one byte per (slot, kv head) holds both. Per-window metadata (the mean,
and norm's scale/min) is keyed by (request slot, window index), which is well defined because
NSN keys windows by req_pool_indices and requires --disable-radix-cache anyway.

Per side: 1.0 index + 0.125 norm2 + 0.0312 norm + 0.0625 mean + 0.0156 mean sc/min
          + 0.0039 norm sc/min = 1.2383 b/ch.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import torch

MEAN_GROUP = 32
ENABLED = os.environ.get("SGLANG_NSN_PACKED") == "1"
# Keep writing the bf16 reconstruction alongside the packed fields. Costs the memory the packed
# form is meant to save, but lets a run A/B the two reads against each other; the point of the
# integration is to turn it off.
KEEP_BF16 = os.environ.get("SGLANG_NSN_PACKED_KEEP_BF16", "1") == "1"


class LayerStore:
    """One layer's packed fields, indexed by the pool's own slot ids.

    Window metadata uses an arena of ``nwin`` entries instead of a dense
    ``[max_requests, max_windows]`` grid. Entries are returned when a request resets.
    """

    def __init__(self, nslots, nwin, KVH, D, ws, device, meta_dtype):
        self.KVH, self.D, self.ws = KVH, D, ws
        self.NG = D // 8
        z = lambda *s, dt: torch.zeros(s, device=device, dtype=dt)
        self.idx_k = z(nslots, KVH, self.NG, dt=torch.uint8)
        self.idx_v = z(nslots, KVH, self.NG, dt=torch.uint8)
        self.n2 = z(nslots, KVH, 2, dt=torch.float16)      # (K, V), one 4-byte load
        self.nrm8 = z(nslots, KVH, dt=torch.uint8)         # K low nibble | V high nibble
        # +1: a trailing trash row. The fused flush scatters every slot every step and routes
        # the not-due ones here, which is what keeps it shape-static and capturable.
        nwin = nwin + 1
        self.nsc = z(nwin, KVH, 4, dt=meta_dtype)
        self.mk_q = z(nwin, KVH, D // 2, dt=torch.uint8)
        self.mv_q = z(nwin, KVH, D // 2, dt=torch.uint8)
        self.mk_s = z(nwin, KVH, D // MEAN_GROUP, 2, dt=meta_dtype)
        self.mv_s = z(nwin, KVH, D // MEAN_GROUP, 2, dt=meta_dtype)

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in
                   (self.idx_k, self.idx_v, self.n2, self.nrm8, self.nsc,
                    self.mk_q, self.mv_q, self.mk_s, self.mv_s))

    @torch.no_grad()
    def commit_meta(self, pk, pv, wid):
        # imported here, not at module scope: nsn_pack imports nsn_quant, which imports this
        from sglang.srt.mem_cache.nsn_pack import pack_nibbles

        KVH, D = self.KVH, self.D
        dt = self.nsc.dtype
        self.nsc[wid, :, 0] = pk.norm_scale.reshape(KVH).to(dt)
        self.nsc[wid, :, 1] = pk.norm_min.reshape(KVH).to(dt)
        self.nsc[wid, :, 2] = pv.norm_scale.reshape(KVH).to(dt)
        self.nsc[wid, :, 3] = pv.norm_min.reshape(KVH).to(dt)
        self.mk_q[wid] = pack_nibbles(pk.mean_q.reshape(KVH, D))
        self.mk_s[wid, :, :, 0] = pk.mean_scale.reshape(KVH, -1).to(dt)
        self.mk_s[wid, :, :, 1] = pk.mean_min.reshape(KVH, -1).to(dt)
        self.mv_q[wid] = pack_nibbles(pv.mean_q.reshape(KVH, D))
        self.mv_s[wid, :, :, 0] = pv.mean_scale.reshape(KVH, -1).to(dt)
        self.mv_s[wid, :, :, 1] = pv.mean_min.reshape(KVH, -1).to(dt)

    @torch.no_grad()
    def commit_rows_batched(self, items):
        """items: [(pk, pv, slots, rows)] -- one scatter per field for the whole layer.

        Requests are staggered in a real batch, so a flush fires on nearly every step with one
        request in it: 8 requests x 28 layers x 5 scatters = ~1120 tiny scatter kernels per
        step, which measured as most of the hook's 114 ms. Gathering the sources first and
        scattering once per field per layer cuts that by the batch size.
        """
        if len(items) == 1:
            pk, pv, slots, rows = items[0]
            return self.commit_rows(pk, pv, slots, rows)
        sl = torch.cat([it[2] for it in items])
        gk = torch.cat([it[0].idx.permute(1, 0, 2)[it[3]] for it in items])
        gv = torch.cat([it[1].idx.permute(1, 0, 2)[it[3]] for it in items])
        n2k = torch.cat([it[0].norm2.permute(1, 0, 2).squeeze(-1)[it[3]] for it in items])
        n2v = torch.cat([it[1].norm2.permute(1, 0, 2).squeeze(-1)[it[3]] for it in items])
        nb = torch.cat([(it[0].norm_q.permute(1, 0)[it[3]]
                         | (it[1].norm_q.permute(1, 0)[it[3]] << 4)) for it in items])
        self.idx_k[sl] = gk
        self.idx_v[sl] = gv
        self.n2[sl, :, 0] = n2k.to(torch.float16)
        self.n2[sl, :, 1] = n2v.to(torch.float16)
        self.nrm8[sl] = nb

    @torch.no_grad()
    def commit_rows(self, pk, pv, slots, rows):
        """Scatter the per-token fields of window-rows `rows` to pool slots `slots`."""
        self.idx_k[slots] = pk.idx.permute(1, 0, 2)[rows]
        self.idx_v[slots] = pv.idx.permute(1, 0, 2)[rows]
        self.n2[slots, :, 0] = pk.norm2.permute(1, 0, 2).squeeze(-1)[rows].to(torch.float16)
        self.n2[slots, :, 1] = pv.norm2.permute(1, 0, 2).squeeze(-1)[rows].to(torch.float16)
        self.nrm8[slots] = (pk.norm_q.permute(1, 0)[rows]
                            | (pv.norm_q.permute(1, 0)[rows] << 4))


class PackedStore:
    """All layers, plus the per-request commit watermark the read path splits on."""

    def __init__(self):
        self.layers: Dict[int, LayerStore] = {}
        self.free: list = []                      # window-arena ids available
        self.wins: Dict[int, list] = {}           # req_pool_index -> its window ids, in order
        # Device-side (req slot, window ordinal) -> arena row, so the read path indexes instead
        # of rebuilding the map on the host every decode step. int32 only: at 512 x 2048 it is
        # 4 MiB, against the hundreds of MiB the same keying would cost for the DATA (which is
        # why the data lives in the free-list arena and only this index is a grid).
        self.wmap = None
        self.committed_t = None
        # req_pool_index -> number of leading tokens whose packed form is committed. The read
        # path attends to [0, n) through the packed kernels and (n, seq_len) through the bf16
        # tail, so this MUST advance exactly in step with the flush loop.
        self.committed: Dict[int, int] = {}
        self.cfg = None

    def configure(self, nslots, nwin, KVH, D, ws, device, meta_dtype):
        self.cfg = dict(nslots=nslots, nwin=nwin, KVH=KVH, D=D, ws=ws,
                        device=device, meta_dtype=meta_dtype)
        self.free = list(range(nwin - 1, -1, -1))
        nreq_slots = int(os.environ.get("SGLANG_NSN_PACKED_MAXREQ", "512"))
        # `wmap` is the eager path's ordinal->arena-row table. The packed serving path computes
        # that row arithmetically (wid_arith / _wid) and the read rebuilds its own map per call,
        # so nothing reads this grid when the fused hook is driving -- it was 4.19 MB per server
        # of allocated-but-never-read memory, and uncharged memory is the class of defect that
        # broke the 32k NIAH boot. Allocate it only for the path that uses it.
        self.wmap = None
        if os.environ.get("SGLANG_NSN_FUSED", "1") != "1":
            maxwin = int(os.environ.get("SGLANG_NSN_PACKED_MAXWIN", "2048"))
            self.wmap = torch.zeros((nreq_slots, maxwin), dtype=torch.int32, device=device)
        # committed token count per request slot, kept on device for the same reason
        self.committed_t = torch.zeros(nreq_slots, dtype=torch.int32, device=device)

    def wid_arith(self, req_compact, ordinal, maxw):
        """Device-computed arena row: no free list, so the flush stays capturable."""
        return req_compact.to(torch.long) * maxw + ordinal.to(torch.long)

    def alloc_window(self, req):
        assert self.free, (
            "NSN packed window arena exhausted; raise SGLANG_NSN_PACKED_WIN_MARGIN")
        wid = self.free.pop()
        lst = self.wins.setdefault(req, [])
        assert lst.__len__() < self.wmap.shape[1], "raise SGLANG_NSN_PACKED_MAXWIN"
        self.wmap[req, len(lst)] = wid
        lst.append(wid)
        return wid

    def release(self, req):
        for wid in self.wins.pop(req, ()):
            self.free.append(wid)
        self.committed.pop(req, None)
        if self.committed_t is not None:
            self.committed_t[req] = 0

    def advance(self, req, n):
        self.committed[req] = self.committed.get(req, 0) + n
        self.committed_t[req] = self.committed[req]

    def layer(self, layer_idx) -> LayerStore:
        st = self.layers.get(layer_idx)
        if st is None:
            assert self.cfg is not None, "PackedStore.configure() not called"
            st = LayerStore(**self.cfg)
            self.layers[layer_idx] = st
        return st

    def nbytes(self):
        return sum(s.nbytes() for s in self.layers.values())


_STORE: Optional[PackedStore] = None


def store() -> PackedStore:
    global _STORE
    if _STORE is None:
        _STORE = PackedStore()
    return _STORE
