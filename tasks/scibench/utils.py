"""SciBench (Wang et al. 2023) as a vendored lm_eval task.

692 open-ended college-level science problems from ten textbooks (atkins, chemmc, calculus,
class, diff, fund, matter, quan, stat, thermo). The gold answer is a NUMBER plus a unit, not a
string, so this cannot reuse MATH-500's string-equivalence checker.

Scoring follows the paper's own released evaluator: correct iff the model's number is within
a 5% RELATIVE tolerance of the gold number (`math.isclose(..., rel_tol=0.05)`). That tolerance
is the benchmark's definition, not a choice made here -- these problems carry rounded textbook
answers, and an exact-match rule would score a correct derivation wrong for reporting 50.68
against a gold of 50.7.

Two deliberate departures from a naive port, both learned from this project's BBH scorer:

  * The unit is given IN THE PROMPT and excluded from the comparison. The gold `answer_number`
    is unitless; asking for a bare number and then parsing only the number keeps the metric
    measuring physics rather than unit formatting.
  * Extraction takes the LAST \\boxed{} by scanning, then falls back to the last number in the
    response. A chat model routinely restates its conclusion, and a first-match rule scores the
    retracted answer -- exactly the defect found in bbh_chat/utils.py.
"""
import math
import re

import datasets

_NUM = re.compile(r"-?\d+(?:[\d,]*\d)?(?:\.\d+)?(?:\s*[eE]\s*[-+]?\d+)?")
# Accept both "\times" and its normalized bare-word form. Bounded digit counts prevent
# pathological backtracking on degenerate generations while covering valid answers.
_SCI = re.compile(r"(-?\d{0,20}\.?\d{1,20})\s*(?:times|x|\*|·|×)"
                  r"\s*10\s*\^?\s*\{?\s*\(?\s*(-?\d{1,6})\s*\)?\s*\}?", re.I)
_SCAN_LIMIT = 512


def doc_to_text(doc: dict) -> str:
    unit = (doc.get("unit") or "").strip()
    tail = f" The unit of the answer is {unit}." if unit else ""
    return ("Solve the following problem, reasoning step by step. Give the final answer as a "
            f"single number in \\boxed{{}}, with no unit inside the box.{tail}\n\n"
            "Problem:\n" + doc["problem_text"].strip() + "\n\nSolution:")


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _p(doc):
        return {"problem_text": doc["problem_text"],
                "answer_number": str(doc["answer_number"]).strip(),
                "unit": doc.get("unit") or "",
                "source": doc.get("source") or ""}
    return dataset.map(_p)


def _last_boxed(text: str):
    """Last \\boxed{...} with brace matching -- a regex cannot nest, and these answers do
    (\\boxed{1.5 \\times 10^{-3}})."""
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def _to_float(s: str):
    if s is None:
        return None
    s = s.replace("$", "").replace("\\", " ").replace(",", "").strip()[:_SCAN_LIMIT]
    m = _SCI.search(s)   # before the plain-number fallback: 1.5 times 10^-3 must not read as 1.5
    if m:                                   # 1.5 \times 10^{-3}
        try:
            return float(m.group(1)) * (10.0 ** int(m.group(2)))
        except ValueError:
            pass
    m = _NUM.search(s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", "").replace(" ", ""))
    except ValueError:
        return None


def extract_answer(text: str):
    b = _last_boxed(text or "")
    if b is not None:
        v = _to_float(b)
        if v is not None:
            return v
    # No box: take the LAST number the response mentions. Weaker, and deliberately last.
    nums = _NUM.findall(text or "")
    return _to_float(nums[-1]) if nums else None


def process_results(doc: dict, results: list) -> dict:
    pred = extract_answer(results[0])
    gold = _to_float(doc["answer_number"])
    if pred is None or gold is None:
        return {"acc": 0.0}
    try:
        ok = math.isclose(pred, gold, rel_tol=0.05, abs_tol=1e-12)
    except (ValueError, OverflowError):
        ok = False
    return {"acc": float(ok)}
