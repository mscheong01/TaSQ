"""AIME 2024 scoring with ordered boxed-answer, answer-phrase, and integer fallbacks."""
import re

_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_ANSWER_IS = re.compile(r"answer is[:\s]*\$?\\?\(?\s*(-?\d+)", re.I)
_INT = re.compile(r"-?\d+")


# An AIME answer is an integer in [0, 999], so nothing longer than a few digits can be one. The
# cap matters because a collapsed quantized model emits digit runs thousands of characters long:
# int() on a 26560-digit string raises ValueError under Python's 4300-digit conversion limit and
# killed a whole seed. fp16 never produces such output, so this only ever breaks the arms under
# test.
_MAX_DIGITS = 6


def _as_int(tok: str):
    """Parse a bounded AIME integer without passing an unbounded digit run to ``int``."""
    if not tok:
        return None
    neg = tok.startswith("-")
    body = tok.lstrip("-").lstrip("0") or "0"
    if len(body) > _MAX_DIGITS:
        return None
    return str(-int(body) if neg else int(body))


def _norm(s: str):
    """AIME answers are integers; normalize away $ , \\! \\text{} and leading zeros."""
    s = s.replace("$", "").replace(",", "").replace("\\!", "").replace("\\,", "").strip()
    s = re.sub(r"\\text\{[^{}]*\}", "", s).strip()
    m = _INT.search(s)
    return _as_int(m.group()) if m else None


def extract_answer(text: str):
    boxed = _BOXED.findall(text)
    if boxed:
        v = _norm(boxed[-1])
        if v is not None:
            return v
    m = list(_ANSWER_IS.finditer(text))
    if m:
        v = _as_int(m[-1].group(1))
        if v is not None:
            return v
    # last resort: the final plausible integer. Scan backwards and skip absurd digit runs rather
    # than taking ints[-1] blindly -- a degenerate generation ends in exactly such a run.
    for tok in reversed(_INT.findall(text)):
        v = _as_int(tok)
        if v is not None:
            return v
    return None


def process_results(doc: dict, results: list) -> dict:
    # The answer column is `Answer` on Maxwell-Jia/AIME_2024 and `answer` elsewhere; resolve it
    # the way the built-in task does so this scorer works on either.
    key = next(k for k in doc.keys() if k.lower() == "answer")
    gold = _norm(str(doc[key]))
    pred = extract_answer(results[0])
    return {"exact_match": float(pred is not None and pred == gold)}
