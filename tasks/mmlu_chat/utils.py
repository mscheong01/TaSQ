"""MMLU scored the way a chat-templated instruct model actually answers it.

This is the GENERATIVE variant on purpose. Plain `mmlu` is loglikelihood, and measured on this
project's harness it does not discriminate KV quantization at all: 1.25-bit arms land within
0.2 points of BF16 on all three models, because scoring never decodes. One server log for a full
mmlu run showed 4,572 prefill batches against 65 decode batches -- the score comes out of the
prompt's own forward pass, so the quantized cache is written and never read back. A CoT answer
decodes a few hundred tokens against that cache, which is the path the paper is about.

Extraction follows bbh_chat's, for the same reasons:

  * The LAST option letter wins, located by scanning. A chat model restates its conclusion, and
    upstream's `(?<=The answer is )(.*)(?=.)` takes the first -- so a model that corrects itself
    is scored on the answer it retracted.
  * Markdown is stripped. "The answer is **(B)**." captures "**(B)**" raw, which never
    string-matches the gold "(B)".
  * A bare letter counts only when nothing alphanumeric follows it, so an answer given as the
    option TEXT ("Patrick") is not read as option (P).
"""
import re

_ANSWER_AT = re.compile(r"answer is[:\s]*", re.I)
_TO_EOL = re.compile(r"[^\n]*")
_PAREN = re.compile(r"\(\s*([A-Da-d])\s*\)")
_BARE = re.compile(r"\b([A-Da-d])\b(?![A-Za-z0-9])")
_MD = re.compile(r"^[\*_`\s\$]+|[\*_`\s\$\.]+$")


def _letter(s):
    """The option letter in a fragment, preferring a parenthesised one."""
    if not s:
        return None
    s = _MD.sub("", s).strip()
    m = _PAREN.search(s)
    if m:
        return m.group(1).upper()
    m = _BARE.match(s)
    return m.group(1).upper() if m else None


def extract_answer(text):
    text = text or ""
    # 1. the last "answer is ..." sentence, to end of line
    starts = [m.end() for m in _ANSWER_AT.finditer(text)]
    if starts:
        got = _letter(_TO_EOL.match(text, starts[-1]).group(0))
        if got:
            return got
    # 2. otherwise the last parenthesised option anywhere in the reply
    hits = _PAREN.findall(text)
    if hits:
        return hits[-1].upper()
    # 3. last resort: a lone letter on the final non-empty line
    for line in reversed([l for l in text.splitlines() if l.strip()]):
        got = _letter(line)
        if got:
            return got
    return None


_NORM = re.compile(r"[^a-z0-9]+")


def _by_choice_text(text, choices):
    """Fallback: the model named the option instead of lettering it.

    "The answer is Patrick" yields no letter -- deliberately, since reading the P of Patrick as
    option (P) is how a scorer starts measuring formatting instead of knowledge. But refusing it
    outright scores a correct answer zero, so match the tail of the reply against the choices and
    accept only when EXACTLY ONE of them appears. Ambiguity stays unscored.
    """
    tail = _NORM.sub(" ", (text or "")[-400:].lower())
    hit = [i for i, c in enumerate(choices)
           if (n := _NORM.sub(" ", str(c).lower()).strip()) and len(n) > 2 and n in tail]
    return "ABCD"[hit[0]] if len(hit) == 1 else None


def process_results(doc, results):
    resp = results[0]
    choices = list(doc.get("choices") or [])
    pred = extract_answer(resp) or _by_choice_text(resp, choices)
    gold = "ABCD"[int(doc["answer"])]
    return {"exact_match": float(pred is not None and pred == gold)}
