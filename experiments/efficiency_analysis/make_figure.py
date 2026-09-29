#!/usr/bin/env python3
"""Build the three-panel efficiency figure from efficiency_analysis/outputs/.

    python3 make_figure.py [model_tag]     # -> figure_<tag>.pdf/.png + figure_data_<tag>.csv
                                          # default tag: qwen3_4b_think

Env knobs (optional; the defaults reproduce the figure byte for byte):
  EFF_OUTPUTS   directory holding the cell JSONs              (default ./outputs)
  FIG_SUFFIX    appended to the output basenames              (default "")
  THRU_SHAPE    throughput shape for panel (b), e.g.          (default "" = the unprefixed
                "2048x16384"; those cells are keyed           2048/32768 cells)
                "<shape>_bs<N>"

Run this with the SYSTEM python3, not the serve env: matplotlib is installed in the former and
not the latter (the same split experiments/ruler_niah/make_table.py hits).

Panels, left to right:
  1. bs=1 per-token decode latency vs context length (1k..128k)
  2. E2E throughput vs batch size at 2048 in / 32768 out; bf16 stops at its measured capacity
  3. prefill latency at 8k / 16k / 32k, grouped bars

Missing cells are DROPPED, never interpolated, and every dropped point is listed on stderr so a
gap in the figure is visible as a gap rather than as a plausible-looking curve.
"""
from __future__ import annotations

import csv, json, os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("EFF_OUTPUTS") or (HERE / "outputs"))
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3_4b_think"
FIG_SUFFIX = os.environ.get("FIG_SUFFIX", "")
# Panel (b)'s shape. "" selects the published 2048/32768 cells, whose keys carry no
# shape prefix); any other shape selects the "<shape>_bs<N>" cells run_efficiency.sh writes for
# every non-default shape.
THRU_SHAPE = os.environ.get("THRU_SHAPE", "")
THRU_IN, THRU_OUT = (THRU_SHAPE.split("x") if THRU_SHAPE else ("2048", "32768"))

# Arm -> (label, colour, marker). Only these three are in the figure, by design.
ARMS = {
    "bf16": ("BF16", "#4C4C4C", "o"),
    "cq":   ("CQ",   "#D1495B", "s"),
    "tasq": ("TaSQ (ours)", "#0F7B6C", "^"),
}
ORDER = ["bf16", "cq", "tasq"]

missing: list[str] = []


def load(panel: str) -> dict[str, dict[int, dict]]:
    """{arm: {x: metrics}} for one panel, x parsed out of the cell key."""
    got: dict[str, dict[int, dict]] = {a: {} for a in ARMS}
    for f in sorted(OUT.glob(f"{MODEL}__{panel}__*.json")):
        d = json.loads(f.read_text())
        arm = d["arm"]
        if arm not in got:
            continue
        if panel == "thru":
            # Published 2048/32768 cells are keyed "bs<N>"; every other shape uses
            # "<in>x<out>_bs<N>".
            if THRU_SHAPE:
                if not d["key"].startswith(f"{THRU_SHAPE}_"):
                    continue
            elif "x" in d["key"]:
                continue
        if not d.get("ok"):
            missing.append(f"{panel}/{arm}/{d['key']}: FAILED ({d.get('error',{}).get('returncode')})")
            continue
        x = int("".join(c for c in d["key"].rsplit("_", 1)[-1] if c.isdigit()))
        got[arm][x] = d["metrics"]
    return got


def human_ctx(v, _pos=None):
    v = int(v)
    return f"{v//1024}k" if v >= 1024 else str(v)


def main() -> int:
    dec, thru, pre = load("decode"), load("thru"), load("prefill")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax in axes:
        ax.grid(True, ls=":", lw=0.7, c="#BBBBBB", alpha=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # ---------------- panel 1: decode latency vs context ----------------
    ax = axes[0]
    for a in ORDER:
        lbl, c, m = ARMS[a]
        xs = sorted(dec[a])
        if not xs:
            continue
        ys = [dec[a][x]["decode_latency_ms_per_token"] for x in xs]
        ax.plot(xs, ys, marker=m, color=c, label=lbl, lw=1.8, ms=5.5)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(human_ctx))
    ax.set_xlabel("Context length")
    ax.set_ylabel("Decode latency (ms / token)")
    # Y limits left to matplotlib. A fixed floor was set to 10 while only Qwen3-4B was being
    # plotted (it bottoms out near 10.1 ms/token); on Llama-3.1-8B, which starts at 17.7, that
    # same floor spends a third of the panel on empty space. Autoscaling keeps the curve shape
    # readable for any model. It also means the axis does NOT start at zero, so the visual gap
    # overstates the ratio -- quote the ratio in the caption rather than eyeballing it here.
    ax.set_title("(a) Decode latency, batch = 1", fontsize=11)
    ax.legend(frameon=False, fontsize=9)

    # ---------------- panel 2: E2E throughput vs batch ----------------
    ax = axes[1]
    bf16_cap_bs = bf16_peak = None
    capf = OUT / (f"{MODEL}__thru_capacity__" + (f"{THRU_SHAPE}_" if THRU_SHAPE else "") + "bf16.json")
    if capf.exists():
        bf16_cap_bs = json.loads(capf.read_text())["capacity_batch"]
    for a in ORDER:
        lbl, c, m = ARMS[a]
        xs = sorted(thru[a])
        if not xs:
            continue
        ys = [thru[a][x]["overall_throughput_tok_s"] for x in xs]
        ax.plot(xs, ys, marker=m, color=c, label=lbl, lw=1.8, ms=5.5)
        if a == "bf16":
            bf16_peak = max(ys)
    if bf16_peak is not None:
        ax.axhline(bf16_peak, ls="--", lw=1.3, c=ARMS["bf16"][1], alpha=0.85)
        ax.annotate(f"BF16 capacity ceiling ({bf16_peak:,.0f} tok/s)",
                    xy=(0.98, bf16_peak), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=8.5, color=ARMS["bf16"][1])
    if bf16_cap_bs:
        ax.axvline(bf16_cap_bs, ls=":", lw=1.3, c=ARMS["bf16"][1], alpha=0.7)
        ax.annotate(f"BF16 max batch = {bf16_cap_bs}", xy=(bf16_cap_bs, 0.02),
                    xycoords=("data", "axes fraction"), rotation=90,
                    ha="right", va="bottom", fontsize=8.5, color=ARMS["bf16"][1])
    # Use a linear batch axis so the capacity limits remain directly comparable.
    ax.set_xlim(left=0)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.set_xlabel("Batch size")
    ax.set_ylabel("E2E throughput (tok / s)")
    ax.set_ylim(bottom=0)   # same reason as panel (a): the panel is read as a ratio
    ax.set_title(f"(b) E2E throughput, {THRU_IN} in / {THRU_OUT} out", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    # ---------------- panel 3: prefill latency bars ----------------
    ax = axes[2]
    ctxs = sorted({x for a in ORDER for x in pre[a]})
    width = 0.26
    for i, a in enumerate(ORDER):
        lbl, c, _ = ARMS[a]
        xs = [j + (i - 1) * width for j, _ in enumerate(ctxs)]
        ys = [pre[a].get(cx, {}).get("prefill_latency_s", 0) * 1000 for cx in ctxs]
        ax.bar(xs, ys, width, label=lbl, color=c, edgecolor="white", lw=0.6)
    ax.set_xticks(range(len(ctxs)))
    ax.set_xticklabels([human_ctx(c) for c in ctxs])
    ax.set_xlabel("Context length")
    ax.set_ylabel("Prefill latency (ms)")
    ax.set_title("(c) Prefill latency", fontsize=11)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    # Use per-model filenames to avoid collisions across runs.
    for ext in ("pdf", "png"):
        p = HERE / f"figure_{MODEL}{FIG_SUFFIX}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"[fig] wrote {p}")

    # the numbers behind the figure, so a reader never has to re-derive them from JSON
    with (HERE / f"figure_data_{MODEL}{FIG_SUFFIX}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "arm", "x", "metric", "value"])
        for nm, data, key in (("decode", dec, "decode_latency_ms_per_token"),
                              ("thru", thru, "overall_throughput_tok_s"),
                              ("prefill", pre, "prefill_latency_s")):
            for a in ORDER:
                for x in sorted(data[a]):
                    w.writerow([nm, a, x, key, data[a][x][key]])
    print(f"[fig] wrote {HERE / f'figure_data_{MODEL}{FIG_SUFFIX}.csv'}")
    if missing:
        print("\n[fig] MISSING / FAILED cells (left as gaps, not interpolated):", file=sys.stderr)
        for m in missing:
            print("   " + m, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
