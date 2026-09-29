#!/usr/bin/env python3
"""Build the RULER-NIAH long-context rows from scratch (client-ready JSONL).

Reproduces the paper's NIAH row files end-to-end: the essay haystack is
extracted from the public `simonjegou/ruler` dataset (16384 config, first 40
`niah_single_2` rows with the intro line and needle sentence stripped), the
8-subtask RULER-NIAH suite is regenerated at the requested context length with
seeded needles, and the serving model's chat template is applied to produce
rows the client (`scripts/run_cell.py`) consumes directly. Deterministic under
a fixed seed, and regenerates the paper's row files byte-identically (modulo
the current-date line some chat templates stamp at render time).

    python tasks/niah/build_niah_rows.py --model openai/gpt-oss-20b \
        --length 8192 --seed 0 --out datasets/niah_8192_gptoss_s200.jsonl
    python tasks/niah/build_niah_rows.py --model Qwen/Qwen3-8B \
        --length 32768 --seed 0 --out datasets/niah_32768_qwen_s200.jsonl
    python tasks/niah/build_niah_rows.py --model meta-llama/Llama-3.1-8B-Instruct \
        --length 131072 --seed 0 --out datasets/niah_131072_llama_s200.jsonl
"""
import argparse
import json
import random
import re
import uuid
from pathlib import Path

# --- RULER-NIAH generator (verbatim from the generator that built the paper's
# --- 8-64K sets; do not reorder CFG or reseed per task: one rng streams
# --- through all subtasks, so row i of a task depends on everything before it.

ADJ = ("solid abashed unsightly grumpy tiny ancient rough abaft lazy efficient stale calm bright "
       "fast deep red blue green gold silver long short new old happy quiet brave clever eager fancy "
       "gentle jolly kind lively proud silly witty zany bold crisp dark early fair giant huge icy").split()
NOUN = ("few geometry patty cornerstone summary orchard blueberry daily pursuit government river moon "
        "track lake fox hill tea coin key road dawn tree idea alluvium melody canyon harbor meadow "
        "anchor beacon cipher domain ember fable galaxy harvest island jungle kernel lantern marble").split()
NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.\n"

CFG = {  # haystack, value-type, n_keys, n_values(per key), n_queried
    "niah_single_1":   ("noise", "num", 1, 1, 1),
    "niah_single_2":   ("essay", "num", 1, 1, 1),
    "niah_single_3":   ("essay", "uuid", 1, 1, 1),
    "niah_multikey_1": ("essay", "num", 4, 1, 1),
    "niah_multikey_2": ("needle", "num", 1, 1, 1),
    "niah_multikey_3": ("needle", "uuid", 1, 1, 1),
    "niah_multivalue": ("essay", "num", 1, 4, 1),
    "niah_multiquery": ("essay", "num", 4, 1, 4),
}


def rval(vt, rng):
    return str(uuid.UUID(int=rng.getrandbits(128))) if vt == "uuid" else str(rng.randint(1_000_000, 9_999_999))


def rkey(rng, used):
    while True:
        k = f"{rng.choice(ADJ)}-{rng.choice(NOUN)}"
        if k not in used:
            used.add(k); return k


def needle(key, val, vt):
    return f"One of the special magic {'uuids' if vt=='uuid' else 'numbers'} for {key} is: {val}.\n"


def build(tok, ctx, task, rng, essay_ids, noise_ids):
    hs, vt, n_k, n_v, n_q = CFG[task]
    word = "uuid" if vt == "uuid" else "number"
    used = set()
    keys = [rkey(rng, used) for _ in range(n_k)]
    # target key(s): multiquery queries all n_k; others query the first key
    q_keys = keys if n_q > 1 else keys[:1]
    # values: multivalue gives n_v values to the (single) queried key
    kv = {}                                   # key -> list of values (target needles)
    for k in keys:
        kv[k] = [rval(vt, rng) for _ in range(n_v if k in q_keys else 1)]
    answers = [v for k in q_keys for v in kv[k]]
    target_needles = [needle(k, v, vt) for k in keys for v in kv[k]]

    # haystack budget
    intro = (f"Some special magic {word}s are hidden within the following text. Make sure to memorize "
             f"them. I will quiz you about the {word}s afterwards.\n" if n_q > 1 or n_v > 1 else
             f"A special magic {word} is hidden within the following text. Make sure to memorize it. "
             f"I will quiz you about the {word} afterwards.\n")
    overhead = len(tok(intro + "".join(target_needles))["input_ids"]) + 40
    budget = max(ctx - overhead, 0)

    if hs == "needle":                        # haystack = distractor needles (pre-size, no O(n^2))
        per = max(len(tok(needle("aa-bb", rval(vt, rng), vt))["input_ids"]), 1)
        tgt = set(keys)                        # distractors need only differ from the target key(s),
        def dkey():                            # not be globally unique (key space is only ~1.8k combos)
            while True:
                k = f"{rng.choice(ADJ)}-{rng.choice(NOUN)}"
                if k not in tgt:
                    return k
        parts = [needle(dkey(), rval(vt, rng), vt) for _ in range(budget // per + 8)]
        filler = tok.decode(tok("".join(parts))["input_ids"][:budget])
    else:                                     # slice pre-tokenized corpus token ids (O(budget))
        base = noise_ids if hs == "noise" else essay_ids
        reps = budget // max(len(base), 1) + 2
        filler = tok.decode((base * reps)[:budget])

    # insert target needles at random depths
    for nd in target_needles:
        cut = rng.randint(0, len(filler))
        filler = filler[:cut] + nd + filler[cut:]
    context = intro + filler + "\n"

    kq = ", ".join(q_keys[:-1]) + (", and " + q_keys[-1] if len(q_keys) > 1 else q_keys[0])
    plural = "s" if (n_q > 1 or n_v > 1) else ""
    question = f"What {'are all' if plural else 'is'} the special magic {word}{plural} for {kq} mentioned in the provided text? "
    prefix = f"The special magic {word}{plural} for {kq} mentioned in the provided text {'are' if plural else 'is'}"
    return context, question, prefix, answers


# --- Essay-corpus extraction from the public simonjegou/ruler dataset.
# --- Verified byte-identical to the corpus behind the paper's row files:
# --- the first 40 niah_single_2 contexts of the 16384 config, each with the
# --- intro line dropped (the body keeps its leading newline) and the needle
# --- sentence removed together with the space preceding it, joined by " ".

def extract_essays(cache_path):
    if cache_path.exists():
        return cache_path.read_text()
    from datasets import load_dataset
    ds = load_dataset("simonjegou/ruler", data_dir="16384", split="test")
    ds = ds.filter(lambda r: r["task"] == "niah_single_2")
    pieces = []
    for row in ds.select(range(40)):
        body = row["context"][row["context"].index("\n"):]
        pieces.append(re.sub(r" ?One of the special magic numbers for [^\s]+ is: \d+\.",
                             "", body, count=1))
    essays = " ".join(pieces)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(essays)
    return essays


# --- Prompt assembly (verbatim from the paper's export path): chat template
# --- applied to user content = context + question via the separator-split
# --- trick, then the answer_prefix as a forced assistant continuation.

def build_prompt(tokenizer, context, question, answer_prefix,
                 enable_thinking=False, model=""):
    separator = "#" * (len(context) + 10)
    templ = tokenizer.apply_chat_template(
        [{"role": "user", "content": context + separator}],
        add_generation_prompt=True, tokenize=False,
        enable_thinking=enable_thinking)
    ctx_part, question_suffix = templ.split(separator)
    # gpt-oss's harmony format requires every message to declare a channel
    # (its own system prompt states this) but apply_chat_template's
    # generation prompt is a bare "<|start|>assistant" with no channel tag.
    # A non-empty answer_prefix glued directly onto that is malformed input:
    # the model burns tokens self-recovering the channel structure before it
    # can answer, which on tight budgets (NIAH's 128) truncates before it
    # ever emits a formatted response -- go straight to the final channel.
    if "gpt-oss" in model.lower() and answer_prefix:
        question_suffix += "<|channel|>final<|message|>"
    return ctx_part + (question or "") + question_suffix + (answer_prefix or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="chat-template source (the serving model)")
    ap.add_argument("--length", type=int, required=True,
                    help="context length in tokens: 8192/16384/32768/65536/131072")
    ap.add_argument("--samples-per-task", type=int, default=25,
                    help="rows kept per subtask (default 25 x 8 tasks = 200, the s200 convention)")
    ap.add_argument("--pool-per-task", type=int, default=100,
                    help="rows generated per subtask before subsetting; the paper's files "
                         "subset a 100/task pool, and the shared rng stream makes row "
                         "content depend on the pool size -- keep 100 to reproduce them")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-model", default="Qwen/Qwen3-8B",
                    help="tokenizer that meters the haystack budget; the paper's sets all "
                         "use Qwen/Qwen3-8B regardless of the serving model -- keep it")
    ap.add_argument("--corpus", type=Path,
                    default=Path(__file__).resolve().parent / "niah_corpus_essays.txt",
                    help="essay-corpus cache; extracted from simonjegou/ruler on first run")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="open the <think> block in the chat template (Qwen3 thinking "
                         "mode); the paper's NIAH rows keep it off")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.samples_per_task > args.pool_per_task:
        ap.error("--samples-per-task cannot exceed --pool-per-task")

    from transformers import AutoTokenizer
    essay = extract_essays(args.corpus)
    budget_tok = AutoTokenizer.from_pretrained(args.budget_model)
    rng = random.Random(args.seed)
    essay_ids = budget_tok(essay)["input_ids"]          # tokenize corpora ONCE
    noise_ids = budget_tok(NOISE * 200)["input_ids"]

    rows = []
    for task in CFG:
        for i in range(args.pool_per_task):
            ctx, q, pref, ans = build(budget_tok, args.length, task, rng, essay_ids, noise_ids)
            rows.append(dict(_id=f"{task}_{args.length}_{args.seed}_{i}", context=ctx, question=q,
                             answer_prefix=pref, answer=ans, task=task, max_new_tokens=128))
        print(f"[gen] {task}: {args.pool_per_task} rows", flush=True)

    # s200 subset convention: first samples_per_task of each subtask, tasks in
    # alphabetical order, round-robin interleaved (row j serves task j % 8).
    by_task = {t: [r for r in rows if r["task"] == t] for t in sorted(CFG)}
    subset = [by_task[t][i] for i in range(args.samples_per_task) for t in sorted(CFG)]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for rid, row in enumerate(subset):
            prompt = build_prompt(tok, row["context"], row["question"], row["answer_prefix"],
                                  enable_thinking=args.enable_thinking, model=args.model)
            rec = {k: v for k, v in row.items() if k != "context"}
            rec.update({"rid": rid, "prompt": prompt, "max_new_tokens": row["max_new_tokens"],
                        "dataset": "niah", "data_dir": str(args.length)})
            fh.write(json.dumps(rec) + "\n")
    print(f"[build_niah_rows] wrote {len(subset)} rows "
          f"({len(CFG)} tasks x {args.samples_per_task}) -> {out}")


if __name__ == "__main__":
    main()
