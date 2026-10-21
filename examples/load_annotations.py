"""Load one judge's annotations, preserve row order, optionally export a CSV."""
import argparse
import json
from softelo_release.data import load_annotations
from softelo_release.suite import DATASET_ID, DATASET_REVISION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=DATASET_ID)
    p.add_argument("--corpus", choices=["lmarena100k", "lmarena140k"], default="lmarena100k")
    p.add_argument("--judge", default="GPT-OSS-120B")
    p.add_argument("--csv", help="Optional output CSV for run_demo.py")
    args = p.parse_args()
    frame = load_annotations(args.data, corpus=args.corpus, judge=args.judge, revision=DATASET_REVISION)
    print(f"{args.corpus} / {args.judge}: {len(frame):,} valid judge-battle rows")
    print(frame[["model_a", "model_b", "human_pref", "judge_pref"]].head().to_string(index=False))
    if args.csv:
        for column in ("scores_a", "scores_b"):
            frame[column] = frame[column].map(json.dumps)
        frame.to_csv(args.csv, index=False)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
