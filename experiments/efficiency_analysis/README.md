# Efficiency analysis

Scripts for measuring decode latency, end-to-end throughput, and prefill latency.

The comparison includes **BF16**, **CQ**, and **TaSQ**. The default model is
**Qwen3-4B-Thinking-2507** and can be changed with `MODEL`.

## Run

```bash
./scripts/run_efficiency.sh decode    # bs=1 decode latency vs context, 1k..128k
./scripts/run_efficiency.sh thru      # throughput vs batch size, 2048 in / 32768 out
./scripts/run_efficiency.sh prefill   # prefill latency at 8k / 16k / 32k

ARMS="tasq" CTXS="65536" ./scripts/run_efficiency.sh decode   # subset re-run
```

Generate the figure after completing the measurements:

```bash
python3 make_figure_collapse.py
```

`make_figure.py` holds the arm list, the loader and the axis helpers that
`make_figure_collapse.py` imports; `scripts/bench_point.py` measures one
(shape, arm) point. Panel (c) reads `outlen_<tag>__<task>.csv`, a tokenized
output-length distribution measured once and kept here.
