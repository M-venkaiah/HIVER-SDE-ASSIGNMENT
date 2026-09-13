"""
Train a TF-IDF + Logistic Regression intent classifier.

IMPORTANT: The keyword-based `weak_label` function below is a BOOTSTRAP label,
not the required human gold label. It exists only so you have something to
smoke-test the pipeline with before you've labelled the real golden set.
Do NOT report accuracy/F1 numbers based on weak labels as your headline result.
"""
import argparse
import os
import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Placeholder taxonomy. YOU must revise this after inspecting the real,
# selected brand's actual conversations (Step 4 in the sequence).
INTENTS = [
    "technical_issue",
    "account_or_login",
    "billing_or_payment",
    "refund_or_cancellation",
    "delivery_or_status",
    "feature_or_how_to",
    "subscription_or_plan",
    "complaint_or_feedback",
    "unknown",
]

KEYWORDS = {
    "account_or_login": ["login", "log in", "password", "account", "sign in", "locked"],
    "billing_or_payment": ["charged", "charge", "payment", "billing", "bill", "card"],
    "refund_or_cancellation": ["refund", "cancel", "cancellation", "money back"],
    "delivery_or_status": ["delivery", "shipping", "arrived", "tracking", "order", "delay", "slot"],
    "subscription_or_plan": ["premium", "subscription", "plan", "renew"],
    "feature_or_how_to": ["how do i", "how can i", "where can i", "setting", "enable"],
    "technical_issue": ["error", "crash", "broken", "not working", "slow", "update", "bug", "freeze"],
}


def weak_label(text):
    t = str(text).lower()
    for label, words in KEYWORDS.items():
        if any(w in t for w in words):
            return label
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df["weak_intent"] = df["text"].fillna("").map(weak_label)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30000)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(df["text"].fillna(""), df["weak_intent"])

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/intent_model.joblib")
    with open("artifacts/intents.json", "w") as f:
        json.dump(INTENTS, f, indent=2)

    print(f"Saved artifacts/intent_model.joblib ({len(df)} training rows)")
    print("WARNING: trained on weak/keyword labels for smoke testing only.")
    print("Replace with real data + human-labelled golden set before reporting any metric.")


if __name__ == "__main__":
    main()
