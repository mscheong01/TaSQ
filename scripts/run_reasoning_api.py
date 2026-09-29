"""Reasoning-mode (thinking-on) lm_eval driver against a running SGLang server. Used for every
arm of the reasoning tables; run_sweep.sh calls it once per seed.

gen_kwargs is passed as a real dict (not a CLI string) specifically so the nested
`chat_template_kwargs={"enable_thinking": True}` key survives -- lm_eval's local-chat-completions
_create_payload() spreads any leftover gen_kwargs keys straight into the JSON request body, and
SGLang's OpenAI-compatible endpoint reads chat_template_kwargs.enable_thinking from exactly there
(verified manually via curl during the SGLang validation experiment).
"""
import faulthandler
import signal
import argparse
import json
import os

import lm_eval
import lm_eval.tasks
from lm_eval.utils import make_table


def _template_has_enable_thinking(model_name):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return "enable_thinking" in (tok.chat_template or "")
    except Exception as e:  # offline, gated, no template -- fall back to sending it
        print(f"[run_reasoning_api] could not inspect chat template ({e}); assuming hybrid")
        return True


# On-demand stack dump: `kill -USR1 <pid>` prints where this process is, into its own log.
# Added after three scorers wedged for an hour in a cubic regex and py-spy could not
# attach when kernel.yama.ptrace_scope=1 sits on a read-only /proc/sys, so an external
# profiler is not an option and the only way in is from inside the process.
faulthandler.register(signal.SIGUSR1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--base_url", type=str, default="http://localhost:30000/v1/chat/completions")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_concurrent", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./reasoning_output")
    parser.add_argument("--save_postfix", type=str, default="nova")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_gen_toks", type=int, default=32768)
    parser.add_argument("--include_path", type=str, default=None,
                        help="extra directory of custom lm_eval task yamls (e.g. tasks/livecodebench)")
    parser.add_argument("--timeout", type=int, default=3600,
                         help="per-request HTTP client timeout in seconds -- the lm_eval default "
                              "(300s) is far too short for a full 32768-token thinking-mode "
                              "completion at ~50 tok/s/request (~11 min), causing spurious "
                              "TimeoutError retries that never actually succeed.")
    args = parser.parse_args()

    task_manager = lm_eval.tasks.TaskManager(include_path=args.include_path)
    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_gen_toks": args.max_gen_toks,
    }
    # Only hybrid models take the switch. Qwen3's original releases gate reasoning on
    # `enable_thinking` in the chat template; the 2507 Thinking releases always reason and their
    # template has no such variable (it just pre-fills `<think>`), so sending the key there is at
    # best inert. Detect it rather than hardcode, so a model swap cannot silently change the mode.
    if _template_has_enable_thinking(args.model_name):
        gen_kwargs["chat_template_kwargs"] = {"enable_thinking": True}
        print("[run_reasoning_api] hybrid template: sending enable_thinking=True")
    else:
        print("[run_reasoning_api] template has no enable_thinking switch (always-reasoning model)")

    results = lm_eval.simple_evaluate(
        model="local-chat-completions",
        model_args={
            "model": args.model_name,
            "base_url": args.base_url,
            "num_concurrent": args.num_concurrent,
            "max_retries": 3,
            "tokenized_requests": False,
            "timeout": args.timeout,
        },
        tasks=[args.task],
        task_manager=task_manager,
        log_samples=True,
        limit=args.limit,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
        apply_chat_template=True,
        gen_kwargs=gen_kwargs,
        confirm_run_unsafe_code=True,
    )

    print(make_table(results))

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.save_postfix}_{args.seed}_{args.task}.json")
    with open(output_file, "w") as f:
        json.dump(results["results"], f, indent=4)
    print(f"Results saved to {output_file}")

    samples = results.get("samples")
    if samples:
        samples_file = output_file[:-5] + "_samples.jsonl"
        with open(samples_file, "w") as f:
            for task, recs in samples.items():
                for r in recs:
                    # `resps` is the RAW generation; `filtered_resps` is only the extracted answer.
                    # Reasoning arms fail by not terminating at least as often as by reasoning
                    # wrongly, and the two are indistinguishable from the extracted answer alone
                    # (see the AIME'24 cell that scores 0.00). Keeping the raw text is what makes
                    # a finish-rate column possible without re-running every cell.
                    row = {"task": task, "doc_id": r.get("doc_id"), "filter": r.get("filter"),
                           "target": r.get("target"), "filtered_resps": r.get("filtered_resps"),
                           "resps": r.get("resps")}
                    row.update({k: v for k, v in r.items()
                                if k in ("exact_match", "em", "f1", "acc", "finish_reason")})
                    f.write(json.dumps(row, default=str) + "\n")
        print(f"Per-example samples saved to {samples_file}")


if __name__ == "__main__":
    main()
