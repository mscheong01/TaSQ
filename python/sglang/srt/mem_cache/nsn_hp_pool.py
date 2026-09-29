"""Compact BF16 residual tier for packed NSNQuant.

Rows already committed to packed storage release their BF16 copies; only the full-precision
prefix and recent tail remain in this tier. ``HPRing`` maps each request position to a fixed row:

    hp_row = compact[req_slot] * R + (pos % R),   R > RECENT + ws + gran

The arithmetic mapping avoids allocation and synchronization during CUDA graph replay. Prefill
defers flushing because the extend kernel reads the full prompt span from this tier.
"""
from __future__ import annotations

import os

import torch

ENABLED = os.environ.get("SGLANG_NSN_PACKED") == "1"


class HPRing:
    """slot/(req, pos) -> hp row, by arithmetic. Capturable: no sync, no allocation."""

    def __init__(self, nslots, ring, max_req, device, prefix=0):
        self.R = ring
        # PREFIX (attention-sink) rows are never quantised, so they live in the bf16 tier for the
        # request's whole life: rows [0, P) of each request's block are theirs, the tail ring of
        # R rows follows. Row(pos) = base + (pos < P ? pos : P + (pos - P) % R).
        self.P = prefix
        self.S = prefix + ring          # rows per request
        self.compact = torch.full((max_req,), -1, dtype=torch.int32, device=device)
        # Capture replays with dummy req slots drawn from the low ids, and nothing host-side
        # may run then, so the low slots are mapped up front (identity) and only higher ones
        # are assigned lazily. n_pre bounds the ring: rows = n_pre * R.
        self.n_pre = int(os.environ.get("SGLANG_NSN_HP_PREASSIGN", "128"))
        self.compact[: self.n_pre] = torch.arange(self.n_pre, dtype=torch.int32, device=device)
        self._next = self.n_pre
        self.max_slots = self.compact.numel()
        # rows [0, n_pre*R) are the ring; row n_pre*R is a trash row (pool slot 0 / padding)
        self.n_hp = self.n_pre * self.S + 1
        self.trash = self.n_pre * self.S
        # slot -> hp row, still needed because the READ path is handed pool slots, not (req,pos).
        # Unbound slots resolve to the trash row, so a stray lookup never aliases a live row.
        self.map = torch.full((nslots,), self.trash, dtype=torch.int32, device=device)

    def ensure(self, req_idx):
        """Assign non-preallocated request slots outside CUDA graph capture."""
        if torch.cuda.is_current_stream_capturing():
            return
        # Every slot the server can address is preassigned -- configure() refuses to boot
        # otherwise -- so this set is always empty. Both this and the boolean-mask index it
        # replaces sync; the point is to avoid the data-dependent ALLOCATION on the common
        # path, not the sync. max() raises on an empty batch, hence the numel guard.
        if req_idx.numel() == 0 or int(req_idx.max()) < self.n_pre:
            return
        hi = req_idx[req_idx >= self.n_pre]
        for r in hi.tolist():
            if self.compact[r].item() < 0:
                if self._next >= self.n_pre:
                    raise RuntimeError(
                        f"nsn_hp_pool: request slot {r} needs a {self._next + 1}th ring but only "
                        f"SGLANG_NSN_HP_PREASSIGN={self.n_pre} rings are allocated; raise it to "
                        f">= the request pool size (max_running_requests + 1).")
                self.compact[r] = self._next
                self._next += 1

    def rows(self, req_idx, pos, loc=None):
        pos = pos.to(torch.long)
        off = torch.where(pos < self.P, pos, self.P + (pos - self.P) % self.R)
        # compact is -1 for a slot with no ring, which would make every row negative: a silent
        # wrap in torch, out of bounds in the kernel. configure()'s boot check makes that
        # unreachable; clamp anyway so a weakened check gives a bounded wrong row, not memory
        # corruption.
        comp = self.compact[req_idx].to(torch.long)
        return torch.where(comp >= 0, comp * self.S + off, torch.full_like(off, self.trash))

    def bind(self, loc, req_idx, pos):
        """Record slot -> row for the read path, and return the rows to write.

        Pool slot 0 is SGLang's padding target (graph-capture padding rows, dummy tokens); it
        must never be bound to a ring row, or the padded write clobbers a live request's raw
        row. Route it to the trash row and leave map[0] there.

        A prefill longer than the ring binds several positions of one request to the same ring
        row (pos and pos+R alias). Only the newest alias can be read back (everything older is
        below the commit watermark and is quantised straight from this call's K/V), and a scatter
        with duplicate indices does not define which write lands last. So every position that has
        a newer alias in this call goes to the trash row instead: a per-request max over the call,
        then pos <= pmax - R. In decode each request has one position, so this is a no-op there
        (and shape-static, hence capture-safe)."""
        rows = self.rows(req_idx, pos, loc)
        pos_l = pos.to(torch.long)
        ridx = req_idx.to(torch.long)
        # Padding rows (loc == 0) keep a real slot id and a STALE position across a graph replay,
        # so they must not contribute to the maximum: a stale 60000 beside a live 4000 would
        # declare the live token dead and pin it to the trash row for the rest of the request.
        live = loc != 0
        pmax = torch.full((self.compact.shape[0],), -1, dtype=torch.long, device=pos_l.device)
        pmax.scatter_reduce_(0, ridx, torch.where(live, pos_l, torch.full_like(pos_l, -1)),
                             reduce="amax")
        dead = live & (pos_l >= self.P) & (pos_l <= pmax[ridx] - self.R)
        rows = torch.where(dead, torch.full_like(rows, self.trash), rows)
        rows = torch.where(loc == 0, torch.full_like(rows, self.trash), rows)
        self.map[loc] = rows.to(torch.int32)
        return rows

    def lookup(self, loc):
        return self.map[loc]


_HP = None

# Sizes of the served request pool, recorded by the model runner before the KV pool is built:
# req_pool_size (request slots; every slot needs a preassigned ring) and r2t_width (the
# req_to_token row width, which is the read glue's buffer width W).
_SERVED: dict = {}


def set_served(**kw):
    _SERVED.update({k: int(v) for k, v in kw.items() if v is not None})


def served(key, default=None):
    return _SERVED.get(key, default)


def hp():
    return _HP


def ring_len():
    """Tail ring rows per request. The extend (prefill) kernel attends the new tokens through
    the k/v it is handed directly (no radix prefix, no chunking), so the tier only ever has to
    hold what is still RAW: RECENT + one window + one eviction granule. 512 covers RECENT=256."""
    return int(os.environ.get("SGLANG_NSN_HP_RING", "512"))


def prefix_len():
    from sglang.srt.mem_cache.nsn_quant import _PREFIX_TOKENS
    return int(_PREFIX_TOKENS)


def rows_per_request():
    return prefix_len() + ring_len()


def configure(nslots, device, max_req=1024, req_pool_size=None):
    """Returns the ring; `.n_hp` is how many bf16 rows the pool must allocate.

    req_pool_size: the server's request-slot count. Every live slot needs its own ring, so a
    shortfall is a configuration error and is raised HERE rather than on the first request
    that happens to land on a high slot.
    """
    global _HP
    if _HP is None:
        _HP = HPRing(nslots, ring_len(), max_req, device, prefix=prefix_len())
        if req_pool_size is not None and req_pool_size > _HP.n_pre:
            raise RuntimeError(
                f"SGLANG_NSN_HP_PREASSIGN={_HP.n_pre} rings for a request pool of "
                f"{req_pool_size} slots: a request on slot >= {_HP.n_pre} would have no ring. "
                f"Set SGLANG_NSN_HP_PREASSIGN >= {req_pool_size} (it costs "
                f"{_HP.S} bf16 rows per slot per layer)."
            )
    return _HP
