#!/usr/bin/env python3
"""Efficiency figure variant: the two serving panels, plus a reasoning-collapse panel.

    EFF_OUTPUTS=outputs FIG_SUFFIX=_collapse \\
      /usr/bin/python3 make_figure_collapse.py qwen3_4b_think

Panels, left to right:
  (a) E2E throughput vs batch size   -- panel (b) of make_figure.py, unchanged
  (b) Prefill latency                -- panel (c) of make_figure.py, unchanged
  (c) Per-sample LiveCodeBench-v6 output length, with the cap-hit region boxed

This is a SEPARATE script and make_figure.py is not touched, so figure_<tag>_serveopt.pdf keeps
its exact current content. Panels (a) and (b) are drawn from the same cell JSONs by the same
`load()`, imported rather than copied -- the decode panel is simply dropped and the remaining two
shift left.

Panel (c) comes from outlen_<tag>.csv, which dump_outlen.py writes in the serve env (tokenizing
needs `transformers`, plotting needs `matplotlib`, and no interpreter here has both). Its cap-hit
rule is the collapse dumper's, so this panel and
Table~\\ref{tab:reasoning-collapse} report the same numbers by construction.

Three arms only -- BF16 / CQ / TaSQ -- matching the rest of the efficiency figure. NSNQuant and
Nova are out of scope here, as README.md says.
"""
from __future__ import annotations

import csv, os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# make_figure reads MODEL from sys.argv and its paths from the same env vars, so importing it
# here reuses the loader with identical behaviour rather than forking a second copy of it.
import make_figure as MF

MODEL = MF.MODEL
SUFFIX = os.environ.get("FIG_SUFFIX", "")
CAP = 32768
# which reasoning cell panel (c) is drawn from; dump_outlen.py writes one CSV per task
TASK = os.environ.get("COLLAPSE_TASK", "lcb_v6")
TASK_TITLE = {"lcb_v6": "LiveCodeBench-v6 output length, 1055 problems x 3 seeds",
              "aime24": "AIME'24 output length, 30 problems x 3 seeds"}
ORDER = MF.ORDER                      # bf16, cq, tasq
ARMS = MF.ARMS                        # arm -> (label, colour, marker)


def load_outlen() -> dict[str, list[tuple[int, int]]]:
    p = HERE / f"outlen_{MODEL}__{TASK}.csv"
    if not p.exists():
        sys.exit(f"[fig] missing {p} -- run dump_outlen.py {MODEL} {TASK} in the serve env first")
    got: dict[str, list[tuple[int, int]]] = {a: [] for a in ORDER}
    with p.open() as fh:
        for r in csv.DictReader(fh):
            if r["arm"] in got:
                got[r["arm"]].append((int(r["gen_tokens"]), int(r["cap_hit"])))
    return got


def main() -> int:
    thru, pre = MF.load("thru"), MF.load("prefill")
    outlen = load_outlen()

    # Type scale. The figure is printed at roughly half a text width per panel, so matplotlib's
    # defaults come out too small to read in the paper; everything below is set relative to this.
    plt.rcParams.update({
        "font.size": 13, "axes.labelsize": 15, "xtick.labelsize": 13,
        "ytick.labelsize": 13, "legend.fontsize": 13,
    })
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.35))
    for ax in axes:
        ax.grid(True, ls=":", lw=0.7, c="#BBBBBB", alpha=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # ---------------- panel (a): E2E throughput vs batch  [was make_figure's (b)] ------------
    ax = axes[0]
    bf16_cap_bs = bf16_peak = None
    capf = MF.OUT / (f"{MODEL}__thru_capacity__"
                     + (f"{MF.THRU_SHAPE}_" if MF.THRU_SHAPE else "") + "bf16.json")
    if capf.exists():
        import json
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
                    ha="right", va="bottom", fontsize=12.5, color=ARMS["bf16"][1])
    if bf16_cap_bs:
        ax.axvline(bf16_cap_bs, ls=":", lw=1.3, c=ARMS["bf16"][1], alpha=0.7)
        # Offset to the RIGHT of the line, not left of it: bf16's capacity batch is small (6), so
        # the line sits near the y-axis and a left-anchored rotated label lands on top of the tick
        # labels and the axis title. An offset in points keeps the clearance fixed whatever the
        # data limits are.
        ax.annotate(f"max batch = {bf16_cap_bs}", xy=(bf16_cap_bs, 0.04),
                    xycoords=("data", "axes fraction"),
                    xytext=(4, 0), textcoords="offset points", rotation=90,
                    ha="left", va="bottom", fontsize=11.5, color=ARMS["bf16"][1])
    # Fit both axes to the data instead of pinning them at the origin.
    # make_figure.py's panel (b) still starts at 0 -- deliberately, so the curves can be read as a
    # ratio against the BF16 ceiling -- so read THIS panel's vertical distances as differences, not
    # as ratios. The BF16 ceiling and max-batch guides are data points, so both stay in frame.
    xs_all = sorted({x for a in ORDER for x in thru[a]})
    ys_all = [thru[a][x]["overall_throughput_tok_s"] for a in ORDER for x in thru[a]]
    if xs_all and ys_all:
        xpad = max((xs_all[-1] - xs_all[0]) * 0.04, 0.5)
        ax.set_xlim(xs_all[0] - xpad, xs_all[-1] + xpad)
        ylo, yhi = min(ys_all), max(ys_all)
        ypad = max((yhi - ylo) * 0.08, 1.0)
        ax.set_ylim(ylo - ypad, yhi + ypad)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.set_xlabel("Batch size")
    ax.set_ylabel("E2E throughput (tok / s)")
    ax.legend(frameon=False, fontsize=13, loc="lower right")

    # ---------------- panel (b): prefill latency bars  [was make_figure's (c)] --------------
    ax = axes[1]
    ctxs = sorted({x for a in ORDER for x in pre[a]})
    width = 0.26
    for i, a in enumerate(ORDER):
        lbl, c, _ = ARMS[a]
        xs = [j + (i - 1) * width for j, _ in enumerate(ctxs)]
        ys = [pre[a].get(cx, {}).get("prefill_latency_s", 0) * 1000 for cx in ctxs]
        ax.bar(xs, ys, width, label=lbl, color=c, edgecolor="white", lw=0.6)
    ax.set_xticks(range(len(ctxs)))
    ax.set_xticklabels([MF.human_ctx(c) for c in ctxs])
    ax.set_xlabel("Context length")
    ax.set_ylabel("Prefill latency (ms)")
    ax.legend(frameon=False, fontsize=13)

    # ---------------- panel (c): output-length distribution, cap included in the curve ------
    # One continuous curve per arm: 1k-wide bins across the budget, and the cap-hits as the
    # curve's LAST point at the budget itself. They were drawn as detached stems at first, which
    # read as a separate chart bolted on; they belong to the same distribution and the steep rise
    # into that final point is the thing worth seeing. No markers -- at 32 bins they only added
    # clutter, and the three arms are already separated by colour.
    ax = axes[2]
    BINW = 1024 if TASK == "lcb_v6" else 3072
    nbin = CAP // BINW
    centers = [(j + 0.5) * BINW for j in range(nbin)]
    ymax = 0.0

    ax.axvspan(CAP - BINW, CAP + BINW, color="#B00020", alpha=0.07, zorder=1)
    for a in ORDER:
        lbl, c, _ = ARMS[a]
        vals = outlen[a]
        n = len(vals)
        counts = [0] * nbin
        for v, hit in vals:
            if not hit:
                counts[min(int(v // BINW), nbin - 1)] += 1
        pct = [100 * k / n for k in counts]
        rate = 100 * sum(h for _, h in vals) / n
        mean = sum(v for v, _ in vals) / n
        # the cap point closes the curve at the budget
        ax.plot(centers + [CAP], pct + [rate], color=c, lw=2.0,
                label=f"{lbl}  (mean {mean:,.0f} tokens)", zorder=3, solid_joinstyle="round")
        ax.annotate(f"{rate:.1f}%", xy=(CAP, rate), ha="left", va="center", fontsize=13,
                    color=c, fontweight="bold", xytext=(5, 0), textcoords="offset points",
                    zorder=5)
        ymax = max([ymax, rate] + pct)

    ax.set_ylim(0, ymax * 1.22)
    ax.annotate("cap hit", xy=(CAP, ymax * 0.82), ha="right", va="center",
                xytext=(-6, 0), textcoords="offset points",
                fontsize=13, color="#B00020", fontweight="bold")
    ax.set_xlim(0, CAP * 1.10)
    ax.xaxis.set_major_locator(MultipleLocator(8192))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{int(v)//1024}k" if v else "0"))
    ax.set_xlabel("Generated tokens per problem")
    ax.set_ylabel("Problems (%)")
    ax.legend(frameon=False, fontsize=11.5, loc="upper left", borderaxespad=0.2)

    # Panel titles removed: the letters go underneath and
    # what each panel shows belongs in the LaTeX caption, not in three separate headings.
    for ax, letter in zip(axes, "abc"):
        ax.text(0.5, -0.26, f"({letter})", transform=ax.transAxes,
                ha="center", va="top", fontsize=15)

    fig.tight_layout(w_pad=1.0, pad=0.6)
    for ext in ("pdf", "png"):
        p = HERE / f"figure_{MODEL}{SUFFIX}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"[fig] wrote {p}")
    if MF.missing:
        print("\n[fig] MISSING / FAILED cells (left as gaps, not interpolated):", file=sys.stderr)
        for m in MF.missing:
            print("   " + m, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
