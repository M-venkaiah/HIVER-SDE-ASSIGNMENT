"""
Evaluation harness. Deliberately refuses to produce numbers until the
golden set has actually been hand-labelled -- this is a guardrail against
accidentally reporting a metric computed on blank/weak labels.
"""
import os
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report


def main():
    path = "evaluation/golden_set.csv"
    if not os.path.exists(path):
        raise SystemExit("Run: python -m evaluation.make_golden --input data/processed/inbound.csv --n 200")

    df = pd.read_csv(path)
    if df["gold_intent"].fillna("").eq("").any():
        n_blank = df["gold_intent"].fillna("").eq("").sum()
        raise SystemExit(f"{n_blank} rows still have a blank gold_intent. Label every row before evaluating.")

    model = joblib.load("artifacts/intent_model.joblib")
    pred = model.predict(df["text"].fillna(""))
    y = df["gold_intent"].astype(str)

    result = {
        "n": int(len(df)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "weighted_f1": float(f1_score(y, pred, average="weighted")),
        "classification_report": classification_report(y, pred, output_dict=True, zero_division=0),
    }

    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/intent_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps({k: v for k, v in result.items() if k != "classification_report"}, indent=2))
    print("\nFull per-intent breakdown saved to evaluation/results/intent_metrics.json")


if __name__ == "__main__":
    main()
