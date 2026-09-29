#!/usr/bin/env python3
"""Measure ONE efficiency point against an ALREADY-RUNNING server, and write one cell JSON.

Thin wrapper around sglang's own `bench_one_batch_server --base-url`, so the numbers come from
the upstream measurement code rather than a reimplementation of it, and from the SAME served
engine the accuracy tables use (scripts/serve_method.sh, booted by run_efficiency.sh).

What the upstream reports, and what this project calls it (definitions are fixed HERE so the
figure and the README cannot drift from the data):

    last_ttft            time to first token  -> PREFILL LATENCY (panel 3)
    output_throughput    bs*out/(lat - ttft)  -> decode throughput; at bs=1,
                         1000/output_throughput is the per-token DECODE LATENCY ms (panel 1)
    overall_throughput   bs*(in+out)/lat      -> E2E THROUGHPUT (panel 2)

All three are recorded in every cell regardless of which panel asked for it, because they cost
nothing extra and a panel that later wants a different definition should not need a re-run.
"""
from __future__ import annotations

import argparse, json, os, socket, subprocess, sys, time
from pathlib import Path


def server_info(base_url: str) -> dict:
    import urllib.request
    out = {}
    for ep, keys in (("/get_server_info", ("max_total_num_tokens", "version")),):
        try:
            with urllib.request.urlopen(base_url + ep, timeout=30) as r:
                d = json.loads(r.read().decode())
            for k in keys:
                if k in d:
                    out[k] = d[k]
            # nested under server_args on some builds
            sa = d.get("server_args") or {}
            for k in ("mem_fraction_static", "chunked_prefill_size", "cuda_graph_max_bs",
                      "disable_cuda_graph", "context_length", "max_running_requests"):
                if k in sa:
                    out[k] = sa[k]
        except Exception as e:  # noqa: BLE001
            out["server_info_error"] = f"{type(e).__name__}: {e}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--out", required=True, help="cell JSON path")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--panel", required=True, choices=["decode", "thru", "prefill"])
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--input-len", type=int, required=True)
    ap.add_argument("--output-len", type=int, required=True)
    ap.add_argument("--key", required=True, help="point key, e.g. ctx65536 or bs32")
    ap.add_argument("--artifact", default="")
    ap.add_argument("--extra-env", default="")
    ap.add_argument("--timeout", type=int, default=5400)
    a = ap.parse_args()

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    resfile = outp.with_suffix(".raw.jsonl")
    if resfile.exists():
        resfile.unlink()

    cmd = [sys.executable, "-m", "sglang.bench_one_batch_server",
           "--model-path", "None", "--base-url", a.base_url,
           "--batch-size", str(a.batch_size),
           "--input-len", str(a.input_len),
           "--output-len", str(a.output_len),
           "--result-filename", str(resfile),
           "--no-append-to-github-summary"]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
    wall = time.time() - t0

    rec = None
    if resfile.exists():
        lines = [l for l in resfile.read_text().splitlines() if l.strip()]
        if lines:
            rec = json.loads(lines[-1])

    cell = {
        "model": a.tag, "model_path": a.model, "arm": a.arm, "panel": a.panel,
        "key": a.key,
        "batch_size": a.batch_size, "input_len": a.input_len, "output_len": a.output_len,
        "ok": rec is not None,
        "wall_s": round(wall, 2),
    }
    if rec is not None:
        lat, ttft = rec["latency"], rec["last_ttft"]
        cell["metrics"] = {
            "latency_s": lat,
            "prefill_latency_s": ttft,
            "output_throughput_tok_s": rec["output_throughput"],
            "overall_throughput_tok_s": rec["overall_throughput"],
            "input_throughput_tok_s": rec["input_throughput"],
            # derived, defined once (see module docstring)
            "decode_latency_ms_per_token": (
                1000.0 * a.batch_size / rec["output_throughput"]
                if rec["output_throughput"] > 0 else None),
        }
    else:
        cell["error"] = {
            "returncode": p.returncode,
            "stderr_tail": p.stderr[-4000:],
            "stdout_tail": p.stdout[-2000:],
        }
    cell["provenance"] = {
        "artifact": a.artifact,
        "artifact_md5": _md5_dir(a.artifact),
        "extra_env": a.extra_env,
        "host": socket.gethostname(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "commit": _git_commit(),
        "dirty": _git_dirty(),
        "server": server_info(a.base_url),
    }
    outp.write_text(json.dumps(cell, indent=2) + "\n")
    print(("[bench] OK   " if cell["ok"] else "[bench] FAIL ")
          + f"{a.tag}/{a.panel}/{a.arm}/{a.key} -> {outp}")
    if cell["ok"]:
        m = cell["metrics"]
        print(f"        ttft={m['prefill_latency_s']:.3f}s  "
              f"decode={m['decode_latency_ms_per_token']:.3f} ms/tok  "
              f"e2e={m['overall_throughput_tok_s']:.1f} tok/s")
    else:
        print(p.stderr[-1500:])
    return 0 if cell["ok"] else 1


def _md5_dir(d: str) -> str:
    if not d or not os.path.isdir(d):
        return ""
    import hashlib
    h = hashlib.md5()
    for f in sorted(Path(d).glob("*.pt")):
        h.update(f.name.encode())
        h.update(str(f.stat().st_size).encode())
    return h.hexdigest()[:12]


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(["git"] + args, capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[3]).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _git_commit() -> str:
    return _git(["rev-parse", "--short", "HEAD"])


def _git_dirty() -> bool:
    return bool(_git(["status", "--porcelain", "python", "scripts", "exp"]))


if __name__ == "__main__":
    raise SystemExit(main())
