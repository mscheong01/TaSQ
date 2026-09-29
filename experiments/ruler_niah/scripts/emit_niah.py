#!/usr/bin/env python3
"""Fold one run_cell.py NIAH JSONL into the outputs/ record the table builder reads.

NIAH is scored per row by `contains_answer`, so the cell score is just the row mean -- but it is
written under the same "results" -> "<bench>" -> "<metric>,none" shape the lm_eval cells use, so
table_lib needs no NIAH special case.
"""
import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys

raw, dst, tag, bench, method, seed, artifact = sys.argv[1:8]

rows = [json.loads(l) for l in open(raw) if l.strip()]
if not rows:
    raise SystemExit(f"[pe] empty NIAH result: {raw}")
acc = sum(bool(r["correct"]) for r in rows) / len(rows)


def git(*a):
    try:
        return subprocess.check_output(["git", *a], text=True,
                                       cwd=os.path.dirname(dst)).strip()
    except Exception:
        return "unknown"


cb = os.path.join(artifact, "codebook.pt")
ahash = "n/a"
if os.path.exists(cb):
    with open(cb, "rb") as f:
        h = hashlib.md5()
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    ahash = h.hexdigest()[:12]

os.makedirs(os.path.dirname(dst), exist_ok=True)
json.dump({
    "model": tag, "bench": bench, "method": method, "seed": int(seed),
    "results": {bench: {"exact_match,none": acc, "n_rows": len(rows)}},
    "provenance": {
        "commit": git("rev-parse", "--short", "HEAD"),
        # TRACKED modifications only. Untracked files are other experiments' scratch sitting
        # in this shared working tree; they cannot change the code that produced this
        # cell, and counting them made every run report dirty and the flag useless.
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "dirty_files": git("diff", "--name-only", "HEAD") or None,
        "artifact_md5": ahash,
        # nova and novabf16 are the same codebook, the same protocol and the same artifact_md5;
        # the served scale dtype is the only thing that separates them, so it has to be here or
        # the two rows are indistinguishable after the fact.
        "scale_dtype": os.environ.get("PE_SCALE_DTYPE"),
        "prefix_tokens": os.environ.get("PREFIX_TOKENS"),
        "recent_tokens": os.environ.get("RECENT_TOKENS"),
        "rows_file": os.path.basename(raw),
        "host": socket.gethostname(),
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    },
}, open(dst, "w"), indent=1)
print(f"[pe] {tag}/{bench}/{method}/s{seed}: {100 * acc:.1f}  ({len(rows)} rows) -> {dst}")
