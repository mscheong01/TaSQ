"""MATH-500 task. Reuses lm_eval's own hendrycks_math string-normalization equivalence checker
(is_equiv/remove_boxed/last_boxed_only_string -- no extra deps like sympy/math_verify/antlr4,
which aren't installed in this env), but with our own process_results: hendrycks_math's shipped
process_results extracts the model's answer by slicing between the first/last '$' in the response,
which silently leaves a `\boxed{...}` wrapper in place (and so never matches the ground truth,
which IS boxed-stripped) whenever the model naturally answers with `... $\boxed{X}$` rather than
the original MATH few-shot completion style `... $X$`. We extract via the same brace-matching
`last_boxed_only_string`/`remove_boxed` used for the ground truth, falling back to the original
$-slicing heuristic only if no \boxed is found.

Few-shot prompt/exemplars copied verbatim from lm_eval's own minerva_math task (its `utils.py`
can't be imported directly here -- it unconditionally requires sympy/math_verify/antlr4==4.11 at
import time, none of which are installed in this env -- so the plain exemplar list and doc_to_text
string are inlined instead of imported).
"""
from typing import Dict, List

import datasets
from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed


def doc_to_text(doc: dict) -> str:
    return ("Solve the following problem, reasoning step by step, and put your final "
            "answer in \\boxed{}.\n\nProblem:\n" + doc["problem"] + "\n\nSolution:")


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        return {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": remove_boxed(last_boxed_only_string(doc["solution"])),
        }
    return dataset.map(_process_doc)


def list_fewshot_samples() -> list:
    return [
        {
            "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
            "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
            "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
            "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
            "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
            "few_shot": "1",
        },
    ]


def _extract_answer(resp: str) -> str:
    boxed = last_boxed_only_string(resp)
    if boxed is not None:
        try:
            return remove_boxed(boxed)
        except AssertionError:
            pass
    indices = [pos for pos, char in enumerate(resp) if char == "$"]
    if len(indices) <= 1:
        return resp
    return resp[indices[0] + 1 : indices[-1]]


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    answer = _extract_answer(results[0])
    return {"exact_match": 1 if is_equiv(answer, doc["answer"]) else 0}
