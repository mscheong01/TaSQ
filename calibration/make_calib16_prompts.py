#!/usr/bin/env python3
"""Export gpqa_code calibration windows for ``dump_qkv.py --no-chat-template``.

This keeps NovaKV calibration aligned with the CQ and TaSQ calibration corpus.

    PYTHONPATH=third_party/kvquant python3 calibration/make_calib16_prompts.py \
        --model Qwen/Qwen3-4B --nsamples 16 --seqlen 2048 --seed 0 --out prompts.jsonl
"""
import argparse
import json

from kvquant.datautils import get_gpqa_code
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--nsamples", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mix-code-n", type=int, default=4,
                    help="how many of --nsamples windows come from code rather than GPQA. A COUNT, "
                         "not a ratio: raising --nsamples alone dilutes the code share. Must match "
                         "what the Fisher and simquant steps were given, or nova's dump and the "
                         "CQ/TaSQ centroids are fit on different corpora.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    loader, _ = get_gpqa_code(args.nsamples, args.seed, args.seqlen, args.model,
                              mix_code_n=args.mix_code_n)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    with open(args.out, "w") as f:
        for inp, _t in loader:
            f.write(json.dumps({"prompt": tok.decode(inp[0], skip_special_tokens=False)}) + "\n")
    print(f"wrote {args.out}: {args.nsamples} prompts x {args.seqlen} tokens")


if __name__ == "__main__":
    main()
