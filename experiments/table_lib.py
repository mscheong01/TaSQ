"""Shared LaTeX-table builder for the paper tables.

Reads every ``outputs/<model>__<bench>__<method>[__sN][__rN].json`` a table directory has
accumulated -- from any node, in any order -- and folds them into one table. The three tables
differ only in their axes, so they each supply a config and call :func:`build`.

Table-generation rules:

* Missing cells print ``--`` and are excluded from averages.
* The metric key per benchmark is explicit (``METRICS``). lm_eval exposes several filters per
  task and picking the wrong one yields a plausible number from a different extractor.
* Provenance (commit, artifact md5, residual protocol) is checked across cells.
* Replicates fold to mean+-std. Non-reasoning runs are greedy, so repeats are replicates of
  batching non-determinism, not sampling seeds; both are handled the same way here.
"""

import glob
import json
import os
import re
from collections import defaultdict

# lm_eval metric key per benchmark. Extend as benchmarks are added.
METRICS = {
    "gsm8k_cot_llama":         ("exact_match,flexible-extract", 100.0),
    "humaneval_instruct":      ("pass@1,create_test",           100.0),
    "mbpp_instruct":           ("pass_at_1,extract_code",       100.0),
    "math500":                 ("exact_match,none",             100.0),
    # Vendored generative BBH task.
    "bbh_cot_fewshot_chat":    ("exact_match,none",             100.0),
    "lcb_v6":                  ("pass@1,none",                  100.0),
    # Generative MMLU variant; unlike log-likelihood scoring, it exercises KV-cache reads.
    "mmlu_chat":               ("exact_match,none",             100.0),
    "aime24":                  ("exact_match,none",             100.0),
    # AIME'25 shares aime24's scorer and prompt exactly (tasks/aime25 is the
    # built-in copied with process_results repointed), so it reports the same key.
    "aime25":                  ("exact_match,none",             100.0),
    # SciBench exposes one unfiltered accuracy metric.
    "scibench":                ("acc,none",                     100.0),
}
# NIAH subtasks all score the same way.
for _t in ("niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1"):
    METRICS[_t] = ("exact_match,none", 100.0)

PRETTY_MODEL = {
    "dsr1_llama8b": "DeepSeek-R1-Distill-Llama-8B",
    "llama31_8b":      r"Llama-3.1-8B-Instruct",
    "qwen3_4b":        r"Qwen3-4B",
    "qwen3_4b_think":  r"Qwen3-4B-Thinking-2507",
    "qwen3_4b_inst":   r"Qwen3-4B-Instruct-2507",
    "phi4_reason_plus": r"Phi4-14B-Reasoning-Plus",
}
PRETTY_BENCH = {
    "gsm8k_cot_llama": "GSM8K", "humaneval_instruct": "HumanEval",
    "mbpp_instruct": "MBPP", "math500": "MATH500",
    "bbh_cot_fewshot_chat": "BBH", "lcb_v6": "LCB-v6",
    "aime24": "AIME'24", "aime25": "AIME'25", "scibench": "SciBench",
    "mmlu_chat": "MMLU-CoT",
}
# Plain text by default so the emitted .tex compiles with no macros of ours defined. Set
# PE_OURS_MACRO=\ours (or whatever the paper defines) to switch the method name to a macro.
PRETTY_METHOD = {
    # Baseline IDs are normalized when loading legacy result files.
    "bf16": "BF16", "nsn": "NSNQuant", "cq": "CQ",
    # NovaKV with G=8, a 10-bit K index, and its asymmetric 1-bit scalar V quantizer.
    "nova1b": "NovaKV",
    # \method{} rather than a literal name: the paper renames the method in one place and every
    # emitted table follows. table_float.tex carries a \providecommand fallback so the float still
    # compiles standalone; a paper that defines \method itself wins, since \providecommand is a
    # no-op when the macro already exists. PE_OURS_MACRO still overrides for a one-off.
    "tasq": os.environ.get("PE_OURS_MACRO", r"\method{}"),
}
# Row order in every table: reference first, then the prior methods, then ours last.
# Anything reported as an orphan is a stale output, not a hidden row.
# Nominal bits per KV coordinate, K and V separately. Derived from the served bundles and the
# arm's scale dtype, not from memory -- every figure below is reproducible from the artifact:
#
#   CQ        bounds (0,8,10) on both K and V, pertoken_norm FALSE on both -> 10/8 with no
#             per-token metadata at all.
#   TaSQ       same code width; K adds a per-token RMS scale pooled over all H KV heads
#             (pool_heads_scale=True), so it costs 16 / (128 x H) bits per coordinate:
#             0.0156 for H=8 and 0.0125 for Phi-4's H=10. V is plain CQ.
#   NovaKV    same code width on K, but its scale is per (token, head) -- 16/128 = 0.125 at the
#             bfloat16 the arm pins. V is OSCAR's scalar quantiser at 1 bit stored 1 bit wide:
#             1.0 code + (scale, zero) = 2 x 16/128 = 0.25.
#   NSNQuant  1.238, computed by nsn_pack's own bits_per_channel() from the shapes
#             (128 idx + 16 norm2 + 4.5 norm + 10 mean = 158.5 bits / 128 channels); K and V
#             share the scheme.
#
# These are NOMINAL. Where a code is stored wider than it is (NovaKV's V was 1 bit in int2 crumbs
# until then), the difference belongs in the text, not here.
# Rows to shade. Kept next to PRETTY_METHOD so renaming the arm in one place cannot leave the
# highlight pointing at a row that no longer exists.
OURS_METHODS = {"tasq"}

METHOD_BITS = {
    "bf16":          (16.0, 16.0),
    "cq":            (1.250, 1.250),
    "nsn":           (1.238, 1.238),
    "nova1b":      (1.375, 1.250),
    "tasq": (1.266, 1.250),
}

MODEL_METHOD_BITS = {
    ("phi4_reason_plus", "tasq"): (1.263, 1.250),
}


def bits_cell(method, model=None):
    """"K/V" for the bits column, or an em dash when the arm has no entry."""
    b = MODEL_METHOD_BITS.get((model, method), METHOD_BITS.get(method))
    return "--" if b is None else f"{b[0]:.3f}/{b[1]:.3f}"


METHOD_ORDER = ["bf16", "cq", "nova1b", "nsn", "tasq"]

# Models present in outputs/ but deliberately not rendered. Empty: every model that has cells is
# in a table. Anything reported as an orphan is a stale output, not a hidden row.
HIDDEN_MODELS = []

# The method group allows SINGLE underscores but not double ones. Without the single underscore
# an arm named like `tasq_v2` misparses: the bench group swallows it and the seed becomes the
# method. Double underscore stays excluded because it is the field separator.
NAME_RE = re.compile(
    r"^(?P<model>.+?)__(?P<bench>.+?)__(?P<method>[a-z0-9]+(?:_[a-z0-9]+)*)"
    r"(?:__s(?P<seed>\d+))?(?:__r(?P<rep>\d+))?$"
)


def canonical_method(method):
    return "bf16" if method == "fp16" else method


def load(outdir):
    """outputs/ -> {(model, bench, method): [record, ...]}"""
    cells = defaultdict(list)
    runs = {}
    for path in sorted(glob.glob(os.path.join(outdir, "*.json"))):
        m = NAME_RE.match(os.path.basename(path)[:-5])
        if not m:
            print(f"[table] WARNING ignoring unparseable filename: {os.path.basename(path)}")
            continue
        try:
            with open(path) as result_file:
                rec = json.load(result_file)
        except json.JSONDecodeError:
            print(f"[table] WARNING unreadable (truncated run?): {path}")
            continue
        rec["_path"] = path
        rec["_key"] = m.groupdict()
        method = canonical_method(m["method"])
        rec["_key"]["method"] = method
        rec["method"] = method
        # The same seed/replicate may exist under both IDs after a rerun or migration.
        # Prefer the canonical file, while retaining distinct legacy seeds/replicates.
        identity = (m["model"], m["bench"], method, m["seed"], m["rep"])
        if identity in runs:
            print(f"[table] NOTE duplicate baseline IDs for {identity}: using bf16")
            if m["method"] == "fp16":
                continue
        runs[identity] = rec
    for identity, rec in runs.items():
        cells[identity[:3]].append(rec)
    _drop_short_cells(cells)
    return cells


def _n_rows(rec):
    """The row count a runner recorded for this cell, if it records one at all."""
    for node in rec.get("results", {}).values():
        if isinstance(node, dict) and "n_rows" in node:
            return node["n_rows"]
    return None


def _drop_short_cells(cells):
    """Exclude cells scored on fewer rows than the benchmark's own full count.

    A partially-served cell produces a NUMBER, not an error, and a number that can look better
    than the real one: llama31_8b/niah@32768 scored 100.0 on 4 of 200 rows when the server ran out
    of HP request slots and the rest of the requests 502'd. Nothing downstream could tell that
    apart from a perfect cell.

    The full count is taken from the data -- the maximum n_rows any cell of that benchmark reached
    -- so no per-benchmark constant has to be maintained, and benchmarks whose runner records no
    n_rows (everything that goes through lm_eval) are untouched.
    """
    full = {}
    for (_, bench, _), recs in cells.items():
        for r in recs:
            n = _n_rows(r)
            if n is not None:
                full[bench] = max(full.get(bench, 0), n)
    dropped = []
    for key in list(cells):
        keep = []
        for r in cells[key]:
            n = _n_rows(r)
            if n is not None and n < full.get(key[1], 0):
                dropped.append((os.path.basename(r["_path"])[:-5], n, full[key[1]]))
            else:
                keep.append(r)
        if keep:
            cells[key] = keep
        else:
            del cells[key]
    if dropped:
        print(f"[table] WARNING {len(dropped)} cell(s) scored on TOO FEW ROWS and are excluded "
              f"-- a partial cell still produces a plausible number:")
        for name, n, want in dropped:
            print(f"[table]   {name}: {n}/{want} rows")


def score(rec, bench):
    """Pull the one metric this benchmark is scored by, or None."""
    key, scale = METRICS.get(bench, (None, 100.0))
    res = rec.get("results", {})
    # lm_eval nests per-task; group tasks (BBH) also carry a top-level aggregate.
    node = res.get(bench)
    if node is None and len(res) == 1:
        node = next(iter(res.values()))
    if node is None:
        return None
    if key is None:
        raise SystemExit(f"[table] no metric registered for benchmark '{bench}' -- add it to "
                         f"table_lib.METRICS. Available keys: {sorted(node)}")
    if key not in node:
        print(f"[table] WARNING {bench}: metric '{key}' absent; have {sorted(k for k in node if ',' in k)}")
        return None
    return scale * node[key]


def agg(recs, bench):
    """[records] -> (mean, std, n) over replicates/seeds, or None."""
    vals = [v for v in (score(r, bench) for r in recs) if v is not None]
    if not vals:
        return None
    n = len(vals)
    mean = sum(vals) / n
    std = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return mean, std, n


def row_average(vals):
    """One method row's mean across the benchmark columns -> (mean, std, n), or None.

    Returns None if ANY column is missing. A mean over whichever benchmarks happen to have
    finished is a different quantity from a mean over the table's benchmarks, and once printed
    the two are indistinguishable -- which is the same failure this module already refuses for a
    single cell ("a missing cell prints --, never silently averaged away"). An average is not
    exempt from that rule, so a partial row gets "--" and a NOTE naming what is missing.

    The std is the std of the AVERAGE, propagated from each benchmark's own across-seed std as
    sqrt(sum std_i^2)/k. Benchmarks are separate invocations, so their seeds are independent.
    Pairing seed i of one benchmark with seed i of another and taking the empirical std would
    also be valid -- the pairing is arbitrary but does not bias the sum's variance -- yet with
    three seeds it is far noisier than propagating, and the arbitrariness would have to be
    explained in the caption. Keeps the same meaning as every other cell's "+-": spread of one
    draw, not a standard error.
    """
    if not vals or any(v is None for v in vals):
        return None
    k = len(vals)
    mean = sum(v[0] for v in vals) / k
    std = (sum(v[1] ** 2 for v in vals) ** 0.5) / k
    # n>1 is what makes fmt() print a "+-" at all, so carry the WEAKEST column's replicate count:
    # an average that leans on a single-run cell must not advertise a spread it cannot support.
    return mean, std, min(v[2] for v in vals)


def check_provenance(cells, expect_seeds=None):
    """Warn about anything that makes cells non-comparable. Never fatal -- the table still
    renders, because a warned-about table is useful and a missing one is not."""
    fields = ("commit", "artifact_md5", "prefix_tokens", "recent_tokens")
    seen = defaultdict(set)
    dirty = []
    for (model, bench, method), recs in sorted(cells.items()):
        for r in recs:
            p = r.get("provenance", {})
            if p.get("dirty"):
                dirty.append(os.path.basename(r["_path"]))
            for f in fields:
                if f in ("artifact_md5",):
                    seen[(f, model, method)].add(p.get(f))
                else:
                    seen[(f,)].add(p.get(f))
        if expect_seeds and len(recs) < expect_seeds:
            print(f"[table] WARNING {model}/{bench}/{method}: {len(recs)} run(s), "
                  f"expected {expect_seeds}")
    for k, v in sorted(seen.items()):
        v.discard(None)
        if len(v) > 1:
            print(f"[table] WARNING cells disagree on {k}: {sorted(v)}")
    if dirty:
        print(f"[table] WARNING {len(dirty)} cell(s) produced from a dirty working tree, "
              f"e.g. {dirty[0]}")


# Bolding is decided on the value the READER SEES, not the float behind it. A column where two
# methods both print 48.89 and only one is bold reads as a mistake in the table, and a 1e-9 tie
# break is not a result. Rounding here has to track fmt()'s "%.2f" below -- change one, change
# both. Ties bold every winner rather than the alphabetically-first method, which is what
# max() on (value, name) tuples silently picked before.
_BOLD_DP = 2


def _winners(cand):
    """[(value, method), ...] -> set of every method tied for the displayed maximum."""
    if not cand:
        return frozenset()
    top = max(round(v, _BOLD_DP) for v, _ in cand)
    return frozenset(m for v, m in cand if round(v, _BOLD_DP) == top)


def fmt(a, best=False, show_std=True):
    if a is None:
        return "--"
    mean, std, n = a
    s = f"{mean:.2f}"
    if show_std and n > 1:
        s += f"$\\pm${std:.2f}"
    return f"\\textbf{{{s}}}" if best else s


def table_data(outdir, row_axis, col_axis, expect_seeds=None, methods=None,
               bold_best_excluding_bf16=True, average=False):
    """Fold outputs/ into the grid both renderers draw from.

    row_axis: list of model tags, or None to discover from outputs/.
    col_axis: list of benchmark (or context-length) names, or None to discover.

    An empty outputs/ is not an error -- it yields the full skeleton with every cell None, which
    is how you check a table's shape before spending any GPU time on it.
    """
    cells = load(outdir)
    if not cells:
        print(f"[table] NOTE no results in {outdir} yet -- rendering the empty skeleton")
    check_provenance(cells, expect_seeds)

    models = row_axis or sorted({k[0] for k in cells})
    cols = col_axis or sorted({k[1] for k in cells})
    # A cell that exists but matches no column renders as nothing at all, while a column with no
    # cell renders "--" -- the two are indistinguishable in the output. Say so explicitly.
    orphan_b = sorted({k[1] for k in cells} - set(cols))
    # Models hidden on purpose. Without this every deliberate omission looks like the accident the
    # orphan check exists to catch, and the real signal gets lost in a warning nobody acts on.
    hidden_m = sorted({k[0] for k in cells} & set(HIDDEN_MODELS))
    if hidden_m:
        print(f"[table] NOTE models hidden on purpose (cells kept on disk): "
              + ", ".join(f"{m} ({sum(1 for k in cells if k[0] == m)} cells)" for m in hidden_m))
    orphan_m = sorted({k[0] for k in cells} - set(models) - set(HIDDEN_MODELS))
    if orphan_b:
        print(f"[table] WARNING {sum(1 for k in cells if k[1] in orphan_b)} cell(s) have a bench "
              f"key no column matches, so they are NOT in the table: {orphan_b}")
    if orphan_m:
        print(f"[table] WARNING {sum(1 for k in cells if k[0] in orphan_m)} cell(s) have a model "
              f"key no row matches, so they are NOT in the table: {orphan_m}")
    # Methods are selected from METHOD_ORDER rather than from the data, so an arm dropped from that
    # list disappears without a word. Say what was left out and how much of it there is.
    hidden = sorted({k[2] for k in cells} - set(METHOD_ORDER))
    if hidden:
        counts = {h: sum(1 for k in cells if k[2] == h) for h in hidden}
        print(f"[table] NOTE superseded arms present on disk but not shown: "
              + ", ".join(f"{h} ({counts[h]} cells)" for h in hidden))
    # With the axes pinned, show every method as a row even before it has run. Only in discovery
    # mode (no pinned axes) is the method list narrowed to what actually exists.
    if methods is None:
        methods = (list(METHOD_ORDER) if (row_axis and col_axis)
                   else [m for m in METHOD_ORDER if any(k[2] == m for k in cells)])
    else:
        methods = list(dict.fromkeys(canonical_method(m) for m in methods))
    if not models or not cols:
        raise SystemExit("[table] nothing to render: pin MODELS/BENCHES or produce some outputs")

    blocks = []
    for model in models:
        rows = [(m, [agg(cells.get((model, c, m), []), c) for c in cols]) for m in methods]
        best = []
        for j in range(len(cols)):
            cand = [(r[1][j][0], r[0]) for r in rows
                    if r[1][j] and (not bold_best_excluding_bf16 or r[0] != "bf16")]
            best.append(_winners(cand))
        blocks.append({"model": model, "rows": rows, "best": best})

    if average:
        suppressed = []
        for blk in blocks:
            blk["avg"] = [row_average(vals) for _, vals in blk["rows"]]
            for (method, vals), a in zip(blk["rows"], blk["avg"]):
                if a is None:
                    miss = [cols[j] for j, v in enumerate(vals) if v is None]
                    suppressed.append(f"{blk['model']}/{method} (missing: {', '.join(miss)})")
            cand = [(a[0], m) for (m, _), a in zip(blk["rows"], blk["avg"])
                    if a and (not bold_best_excluding_bf16 or m != "bf16")]
            blk["best_avg"] = _winners(cand)
        if suppressed:
            print(f"[table] NOTE average suppressed for {len(suppressed)} row(s) with an "
                  f"incomplete benchmark set: {'; '.join(suppressed)}")
    return {"cols": cols, "methods": methods, "blocks": blocks, "average": average}


def to_latex_body(data, show_std=True, multirow=None):
    r"""Emit ONLY the tabular -- no float, no caption, no label.

    Kept separate because `\begin{table}` is a float, and LaTeX raises "Not in outer par mode"
    the moment a float lands inside another float (or inside a box/minipage). A paper that
    already has its own `\begin{table}` must be able to `\input` this file directly, so the file
    it inputs cannot open one itself.

    Defaults to a form that compiles against booktabs alone -- the model name simply sits in the
    first row of its block instead of being \multirow'd, and the method column is plain text.
    Both are cosmetic; turn them on when the LaTeX preamble has the packages/macros:
        PE_MULTIROW=1          use \multirow (needs \usepackage{multirow})
        PE_OURS_MACRO=\ours    use a macro for the method name (define it yourself)
    """
    if multirow is None:
        multirow = os.environ.get("PE_MULTIROW", "0") == "1"
    cols = data["cols"]
    avg = data.get("average", False)
    heads = [PRETTY_BENCH.get(c, c.replace("_", r"\_")) for c in cols]
    if avg:
        heads.append("Avg.")
    lines = ["% tabular only -- no float. \\input this inside your own table/table* environment.",
             "% requires: \\usepackage{booktabs}, \\usepackage[table]{xcolor} and \\method"
             " (table_float.tex provides a fallback)"
             + ("  \\usepackage{multirow}" if multirow else ""),
             r"\begin{tabular}{llc" + "r" * len(heads) + "}", r"\toprule",
             "Model & Method & Bits (K/V) & " + " & ".join(heads) + r" \\"]
    for blk in data["blocks"]:
        lines.append(r"\midrule")
        for i, (method, vals) in enumerate(blk["rows"]):
            name = PRETTY_MODEL.get(blk["model"], blk["model"])
            if i != 0:
                head = ""
            elif multirow:
                head = rf"\multirow{{{len(blk['rows'])}}}{{*}}{{{name}}}"
            else:
                head = name
            cs = [fmt(v, best=(method in blk["best"][j]), show_std=show_std)
                  for j, v in enumerate(vals)]
            if avg:
                cs.append(fmt(blk["avg"][i], best=(method in (blk.get("best_avg") or ())),
                              show_std=show_std))
            meth_cell = PRETTY_METHOD.get(method, method)
            bits = bits_cell(method, blk["model"])
            body = [meth_cell, bits] + cs
            # Ours shaded from the METHOD column rightward, cell by cell. \rowcolor would be
            # shorter but it paints the whole row including column 1 -- and column 1 is the
            # model name, which is empty on every row but the block's first, so the shading
            # ran out under the model label instead of starting at the method.
            if method in OURS_METHODS:
                body = [r"\cellcolor{gray!15}" + c for c in body]
            lines.append(f"{head} & " + " & ".join(body) + r" \\")
            # A rule under the uncompressed reference: it is not one of the compared methods,
            # it is the ceiling they are measured against, and a reader should not have to
            # infer that from the numbers.
            if method == "bf16":
                lines.append(r"\cmidrule(lr){2-" + str(3 + len(heads)) + "}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def to_latex_float(body_file, caption, label):
    r"""The float wrapper, as a separate file that `\input`s the body.

    Use this one only at top level (outer par mode). If the paper already opens its own float,
    input the body file instead and put the caption there.
    """
    return "\n".join([
        "% full float. Only valid at top level -- if you already have \\begin{table},",
        f"% \\input{{{body_file}}} instead.",
        "%",
        "% \\method is the name of our method, used by the body file. \\providecommand, not",
        "% \\newcommand: if the paper already defines it, that definition stands and this line does",
        "% nothing. The body file (table.tex) is tabular-only and does NOT define it, so a document",
        "% that \\inputs the body directly must define \\method itself.",
        r"\providecommand{\method}{TaSQ}",
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\input{{{body_file}}}",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}",
    ])


def to_png(data, path, title, show_std=True, note=None):
    """A booktabs-looking preview of the same grid, for eyeballing without a LaTeX toolchain.

    This is a preview, not the paper artifact: the .tex file is what goes in the paper. Both come
    from the same `table_data`, so they cannot disagree.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols, blocks = data["cols"], data["blocks"]
    avg = data.get("average", False)
    header = ["Model", "Method"] + [PRETTY_BENCH.get(c, c) for c in cols] + (["Avg."] if avg else [])
    body, rules, bolds = [], [], []          # rules: row indices that get a rule above them
    for blk in blocks:
        rules.append(len(body))
        for i, (method, vals) in enumerate(blk["rows"]):
            name = PRETTY_MODEL.get(blk["model"], blk["model"]) if i == 0 else ""
            label = PRETTY_METHOD.get(method, method)
            if label.startswith("\\"):        # a macro was configured; the preview spells it out
                label = "TaSQ (ours)"
            cells = []
            for j, v in enumerate(vals):
                s = "--" if v is None else f"{v[0]:.2f}" + (
                    f" ±{v[1]:.2f}" if show_std and v[2] > 1 else "")
                cells.append(s)
                if v is not None and method in blk["best"][j]:
                    bolds.append((len(body), 2 + j))
            if avg:
                a = blk["avg"][i]
                cells.append("--" if a is None else f"{a[0]:.2f}" + (
                    f" \u00b1{a[1]:.2f}" if show_std and a[2] > 1 else ""))
                if a is not None and method in (blk.get("best_avg") or ()):
                    bolds.append((len(body), 2 + len(cols)))
            body.append([name, label] + cells)

    nrow, ncol = len(body), len(header)
    # Column widths from the widest entry, so a long model name is never clipped.
    widths = [max(len(header[c]), *(len(r[c]) for r in body)) + 2 for c in range(ncol)]
    total = sum(widths)
    widths = [w / total for w in widths]

    row_h, title_h = 0.30, 0.62
    note_h = 0.45 if note else 0.10
    fig_w = 0.098 * total + 1.0
    fig_h = row_h * (nrow + 1) + title_h + note_h
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=170)
    # Place the table axes exactly, rather than letting a default axes leave dead space.
    ax = fig.add_axes([0.02, note_h / fig_h, 0.96, row_h * (nrow + 1) / fig_h])
    ax.axis("off")
    fig.text(0.02, 1 - 0.22 * title_h / fig_h, title, fontsize=10.5, va="top", ha="left",
             wrap=True)

    tbl = ax.table(cellText=body, colLabels=header, colWidths=widths, cellLoc="right",
                   bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("none")
        cell.visible_edges = ""
        cell.set_facecolor("white")
        cell.get_text().set_ha("left" if c < 2 else "right")
        cell.PAD = 0.04
        if r == 0:                                     # header: rule above and below
            cell.get_text().set_fontweight("bold")
            cell.visible_edges = "TB"
            cell.set_edgecolor("#333333")
            cell.set_linewidth(1.1)
        elif (r - 1) in rules and r > 1:               # midrule between model blocks
            cell.visible_edges = "T"
            cell.set_edgecolor("#999999")
            cell.set_linewidth(0.7)
        elif r == nrow:                                # bottomrule
            cell.visible_edges = "B"
            cell.set_edgecolor("#333333")
            cell.set_linewidth(1.1)
    for r, c in bolds:
        tbl[r + 1, c].get_text().set_fontweight("bold")
    for c in range(ncol):                              # dim the empty cells
        for r in range(nrow):
            if body[r][c] == "--":
                tbl[r + 1, c].get_text().set_color("#aaaaaa")
    if note:
        fig.text(0.02, 0.02, note, fontsize=7.5, color="#555555", va="bottom", ha="left",
                 wrap=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def build(outdir, row_axis, col_axis, caption, label, expect_seeds=None, show_std=True,
          bold_best_excluding_bf16=True, methods=None, write=True, png_note=None,
          multirow=None, average=False):
    r"""Fold outputs/ and write, beside outputs/:

        table.tex        the tabular alone -- \input this inside your own float
        table_float.tex  a ready-made float that \inputs table.tex -- top level only
        table.png        preview

    Returns the body LaTeX.
    """
    data = table_data(outdir, row_axis, col_axis, expect_seeds, methods,
                      bold_best_excluding_bf16, average)
    body = to_latex_body(data, show_std, multirow)
    if write:
        base = os.path.dirname(os.path.abspath(outdir))
        paths = {k: os.path.join(base, v) for k, v in
                 (("body", "table.tex"), ("float", "table_float.tex"), ("png", "table.png"))}
        with open(paths["body"], "w") as f:
            f.write(body + "\n")
        with open(paths["float"], "w") as f:
            f.write(to_latex_float("table.tex", caption, label) + "\n")
        # PNG rendering is optional; the LaTeX files are always written.
        wrote = ["body", "float"]
        try:
            to_png(data, paths["png"], caption, show_std, png_note)
            wrote.append("png")
        except ImportError as e:
            print(f"[table] NOTE no PNG preview ({e}); the .tex files are the paper artifact")
        for k in wrote:
            print(f"[table] wrote {paths[k]}")
    return body
