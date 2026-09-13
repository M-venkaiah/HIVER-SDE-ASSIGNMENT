"""
Creates CANDIDATE rows for the golden evaluation set. This does NOT label
anything for you. You open the output CSV and fill in gold_intent,
gold_escalate, and gold_reason yourself, by reading each message.
That manual step is not optional -- it's the part of the assignment
that proves the evaluation isn't circular.
"""
import argparse
import os
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    n = min(args.n, len(df))
    sample = df.sample(n=n, random_state=42)[["tweet_id", "text"]].copy()
    sample["gold_intent"] = ""
    sample["gold_escalate"] = ""
    sample["gold_reason"] = ""
    sample["gold_reply_notes"] = ""

    os.makedirs("evaluation", exist_ok=True)
    sample.to_csv("evaluation/golden_set.csv", index=False)
    print(f"Created {n} candidate rows in evaluation/golden_set.csv.")
    print("Open the file and manually fill gold_intent / gold_escalate / gold_reason for every row.")


if __name__ == "__main__":
    main()
