"""BBH scored the way a chat-templated instruct model actually answers.

Upstream `bbh_cot_fewshot` is a completion-style task and two of its settings break under a chat
template, which this project applies to every benchmark:

  * `until: ["\\n\\n"]` -- in completion mode the model continues the exemplar style with no blank
    lines, so the stop never fires early. Given a chat template the model replies in markdown and
    the first blank line arrives before the conclusion, truncating the generation. Measured on
    Qwen3-8B: 16 of 27 subtasks scored exactly 0.000 because the answer sentence was never
    reached. Here `until` keeps only the EOS marker.
  * `regex "(?<=the answer is )(.*)(?=.)"` -- a chat model writes "the answer is **Yes**." and the
    raw capture is "**Yes**", which never string-matches the gold "Yes". Extraction below strips
    markdown and the trailing period, and falls back to the last option-letter or last line when
    the model omits the stock phrase.

Everything else -- dataset, 3-shot exemplars, prompts, exact-match metric -- is upstream's.
"""
import re

# "answer is" located by SCANNING, then captured to end of line -- not by findall.
# findall consumes each match, so when one line carries two of them
# ("the answer is (B) is incorrect, the correct answer is (F)") the first capture swallows
# the second and "last answer wins" silently returns the FIRST. That scored a model which
# corrected itself as though it had not.
_ANSWER_AT = re.compile(r"answer is[:\s]*", re.I)
_TO_EOL = re.compile(r"[^\n]*")
# "**Answer**: X" -- only when X is on the SAME line and non-empty, so a bare "Final Answer:"
# heading (value on the next line) falls through to the \boxed / last-line rules instead.
_ANSWER2 = re.compile(r"^[\s*_#]*answer[\s*_]*:[ \t]*(\S.*)$", re.I | re.M)
_OPTION = re.compile(r"\(([A-Z])\)")
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_JUNK = re.compile(r"^[\$\\\s\-*_`#]*$")   # "$$", "---", "**" ... never an answer


def _clean(s):
    """Peel markdown and trailing punctuation until stable -- "**Yes**." needs both, in
    whichever order they nest."""
    s = (s or "").strip()
    for _ in range(8):
        prev = s
        b = _BOXED.search(s)
        if b:
            s = b.group(1)                            # \boxed{24} -> 24
        s = s.replace("$", "").replace("\\(", "").replace("\\)", "")
        s = re.sub(r"^[\*_`\s]+|[\*_`\s]+$", "", s)   # markdown emphasis / code ticks
        s = s.strip().rstrip(".").strip()
        if s == prev:
            break
    return s


def extract_answer(text):
    """Last 'answer is X' wins -- a chat model often restates its conclusion."""
    text = text or ""
    starts = [m.end() for m in _ANSWER_AT.finditer(text)]
    if starts:
        return _clean(_TO_EOL.match(text, starts[-1]).group(0))
    b = _BOXED.findall(text or "")
    if b:
        return _clean(b[-1])
    m2 = _ANSWER2.findall(text or "")
    if m2:
        return _clean(m2[-1])
    opts = _OPTION.findall(text or "")
    if opts:
        return f"({opts[-1]})"
    lines = [l for l in (text or "").splitlines() if l.strip() and not _JUNK.match(l)]
    return _clean(lines[-1]) if lines else ""


# A gold of "(A)" is a multiple-choice answer; anything else is a literal string.
_GOLD_OPTION = re.compile(r"^\(?([A-Za-z])\)?$")
# The option letter a prediction LEADS with: "(D)", "(D) 12/26/1962", "D." -- but not "Dec",
# because a bare letter only counts when nothing alphanumeric follows it. Without that guard
# an answer given as the option TEXT ("Patrick") would be read as option (P).
_PRED_OPTION = re.compile(r"^\(\s*([A-Za-z])\s*\)|^([A-Za-z])(?![A-Za-z0-9])")


def process_results(doc, results):
    pred = extract_answer(results[0])
    gold = _clean(str(doc["target"]))
    if pred == gold:
        return {"exact_match": 1.0}
    # gold is "(A)" but the model wrote "A", or vice versa
    p, g = pred.strip("()").strip(), gold.strip("()").strip()
    if p.lower() == g.lower():
        return {"exact_match": 1.0}
    # Multiple choice: the model is allowed to name the option it picked. The exemplars end
    # "So the answer is (A).", and models that instead write "So the answer is (A) Patrick."
    # were scored 0 for the restatement alone. That is a formatting habit, not an error, and
    # it is NOT evenly distributed across the systems under test -- on Qwen3-4B it cost the
    # arms between 11 and 32 points of BBH, reordering them. Compare the option letter.
    gm = _GOLD_OPTION.match(gold)
    if gm:
        pm = _PRED_OPTION.match(pred)
        if pm:
            letter = pm.group(1) or pm.group(2)
            return {"exact_match": float(letter.upper() == gm.group(1).upper())}
    return {"exact_match": 0.0}
