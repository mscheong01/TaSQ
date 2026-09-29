#!/usr/bin/env python3
"""Write the GPQA-diamond calibration CSV that dump_qkv.py consumes.

Calibration inputs are deliberately NOT shipped with the repo -- every node regenerates them, so
a stale or mismatched codebook can never be mistaken for a fresh one. This produces the exact
columns dump_qkv.py's GPQA branch looks for.

    python calibration/make_calib_csv.py --out work/gpqa_diamond.csv
"""
import argparse

COLS = [
    "Question",
    "Correct Answer",
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="Idavidrein/gpqa")
    ap.add_argument("--config", default="gpqa_diamond")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    import pandas as pd
    from datasets import load_dataset

    d = load_dataset(args.dataset, args.config, split=args.split)
    missing = [c for c in COLS if c not in d.column_names]
    if missing:
        raise SystemExit(f"{args.dataset}/{args.config} lacks columns: {missing}")
    pd.DataFrame({c: d[c] for c in COLS}).to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(d)} rows)")


if __name__ == "__main__":
    main()
