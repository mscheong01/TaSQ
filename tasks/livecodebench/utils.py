"""LiveCodeBench v6 code-generation task for lm_eval.

The task reads the dataset's raw JSONL files because current `datasets` versions do not execute
its loading script. The evaluator supports:

  * two problem flavours -- `stdin` (feed stdin, diff stdout) and `functional` (call a method on
    `Solution`), selected per test case by its `testtype`;
  * `private_test_cases` arrive base64 -> zlib -> pickle -> json, while `public_test_cases` are
    plain json;
  * generations from reasoning models carry a `<think>...</think>` prefix that must be stripped
    before the code fence is extracted.

pass@1 with a single greedy/sampled generation per problem, every public+private test case
required to pass. Executes model-written code, so `HF_ALLOW_CODE_EVAL=1` is required.
"""
import base64
import json
import multiprocessing as mp
import os
import pickle
import re
import time
import zlib

TIMEOUT_S = float(os.environ.get("LCB_TIMEOUT", "6"))
MAX_CASES = int(os.environ.get("LCB_MAX_CASES", "0"))  # 0 = all
# Cases run concurrently; verdicts are combined with a logical AND, so execution order does not
# change the score.
WORKERS = int(os.environ.get("LCB_WORKERS", "32"))
# Backstop for a problem whose cases are individually under the per-case timeout but collectively
# enormous. Exceeding it fails the problem, which is what a real judge does with a TLE.
PROBLEM_TIMEOUT_S = float(os.environ.get("LCB_PROBLEM_TIMEOUT", "300"))

PROMPT_STDIN = """### Question:
{question}

### Format: Read the inputs from stdin and write the answer to stdout. Do not add any extra \
output. Enclose your code within delimiters as follows.
```python
# YOUR CODE HERE
```

### Answer: (use the provided format with backticks)
"""

PROMPT_FUNCTIONAL = """### Question:
{question}

### Format: You will use the following starter code to write the solution and enclose your code \
within delimiters.
```python
{starter}
```

### Answer: (use the provided format with backticks)
"""


# --------------------------------------------------------------------------- prompt
def doc_to_text(doc):
    starter = (doc.get("starter_code") or "").strip()
    if starter:
        return PROMPT_FUNCTIONAL.format(question=doc["question_content"], starter=starter)
    return PROMPT_STDIN.format(question=doc["question_content"])


def doc_to_target(doc):
    return ""


# --------------------------------------------------------------------------- extraction
_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text):
    """Last fenced block after dropping any reasoning trace.

    An unterminated final fence is kept: reasoning models regularly hit the token budget
    mid-answer, and the partial program is still worth executing.
    """
    text = _THINK.sub("", text)
    text = re.sub(r"^.*?</think>", "", text, flags=re.S)  # unterminated <think>
    blocks = _FENCE.findall(text)
    if blocks:
        return blocks[-1]
    m = re.search(r"```(?:python|py)?\s*\n(.*)\Z", text, re.S)
    return m.group(1) if m else text


# --------------------------------------------------------------------------- test cases
def _load_cases(doc):
    cases = json.loads(doc["public_test_cases"] or "[]")
    priv = doc.get("private_test_cases") or ""
    if priv:
        try:
            cases += json.loads(priv)
        except Exception:
            cases += json.loads(pickle.loads(zlib.decompress(base64.b64decode(priv.encode()))))
    return cases[:MAX_CASES] if MAX_CASES else cases


def _norm(s):
    return "\n".join(line.rstrip() for line in str(s).strip().splitlines())


# --------------------------------------------------------------------------- execution
def _run_stdin(code, stdin, q):
    import io
    import sys

    buf = io.StringIO()
    try:
        sys.stdin = io.StringIO(stdin)
        sys.stdout = buf
        exec(compile(code, "<lcb>", "exec"), {"__name__": "__main__"})
        q.put(("ok", buf.getvalue()))
    except SystemExit:
        q.put(("ok", buf.getvalue()))
    except BaseException as e:  # noqa: BLE001 - any failure is just a wrong answer
        q.put(("err", f"{type(e).__name__}: {e}"))


def _run_functional(code, fn_name, args_json, q):
    try:
        ns = {"__name__": "__main__"}
        exec(compile(code, "<lcb>", "exec"), ns)
        sol = ns["Solution"]()
        args = [json.loads(line) for line in args_json.split("\n") if line.strip()]
        q.put(("ok", json.dumps(getattr(sol, fn_name)(*args))))
    except BaseException as e:  # noqa: BLE001
        q.put(("err", f"{type(e).__name__}: {e}"))


def _exec_one(target, args):
    q = mp.Queue()
    p = mp.Process(target=target, args=(*args, q))
    p.start()
    p.join(TIMEOUT_S)
    if p.is_alive():
        p.kill()
        p.join()
        return ("err", "timeout")
    try:
        return q.get_nowait()
    except Exception:
        return ("err", "no result")


def _exec_many(jobs):
    """Run (target, args, check) jobs with bounded concurrency; stop at the first failure.

    One process per case, as before -- that is what makes a hung case killable -- but up to
    WORKERS of them at once. Yields True only if every case passed.
    """
    deadline = time.monotonic() + PROBLEM_TIMEOUT_S
    pending = list(jobs)
    running = []  # (proc, queue, start, check)
    while pending or running:
        while pending and len(running) < WORKERS:
            target, args, check = pending.pop(0)
            q = mp.Queue()
            p = mp.Process(target=target, args=(*args, q))
            p.start()
            running.append((p, q, time.monotonic(), check))
        if time.monotonic() > deadline:
            for p, _, _, _ in running:
                p.kill()
                p.join()
            return False
        still = []
        for p, q, start, check in running:
            if p.is_alive():
                if time.monotonic() - start > TIMEOUT_S:
                    p.kill()
                    p.join()
                    return _drain(running, still) or False
                still.append((p, q, start, check))
                continue
            p.join()
            try:
                status, got = q.get_nowait()
            except Exception:
                status, got = "err", "no result"
            if status != "ok" or not check(got):
                return _drain(running, still) or False
        running = still
        if running and len(running) == WORKERS:
            time.sleep(0.02)
    return True


def _drain(running, still):
    """Kill everything still in flight after a verdict is already decided."""
    for p, _, _, _ in list(running) + list(still):
        if p.is_alive():
            p.kill()
        p.join(1)
    return False


def _match_functional(expected):
    def check(got):
        try:
            return json.loads(got) == json.loads(expected)
        except Exception:
            return _norm(got) == _norm(expected)
    return check


def _match_stdin(expected):
    return lambda got: _norm(got) == _norm(expected)


def _passes(code, doc):
    if not code.strip():
        return False
    meta = json.loads(doc.get("metadata") or "{}")
    fn_name = meta.get("func_name")
    jobs = []
    for case in _load_cases(doc):
        if case.get("testtype") == "functional":
            if not fn_name:
                return False
            jobs.append((_run_functional, (code, fn_name, case["input"]),
                         _match_functional(case["output"])))
        else:
            jobs.append((_run_stdin, (code, case["input"]), _match_stdin(case["output"])))
    return _exec_many(jobs)


# --------------------------------------------------------------------------- lm_eval hooks
def process_results(doc, results):
    if os.environ.get("HF_ALLOW_CODE_EVAL") != "1":
        raise RuntimeError("LiveCodeBench executes model-written code; set HF_ALLOW_CODE_EVAL=1")
    return {"pass@1": float(_passes(extract_code(results[0]), doc))}
