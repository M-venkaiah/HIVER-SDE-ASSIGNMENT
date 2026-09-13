"""
The core agent: classify -> retrieve -> decide -> draft reply.

This is intentionally simple (lexical TF-IDF retrieval, rule-based escalation,
a conservative extractive-style reply). You should be able to explain and
modify every function here without looking anything up.
"""
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Words that signal a sensitive / account-specific request. Tune this list
# after you see real examples from your chosen brand.
ESCALATE_WORDS = ["account number", "password", "card number", "refund", "fraud", "hacked"]


def retrieve(query, df, k=3):
    """Find the k most textually similar historical messages to `query`."""
    texts = df["text"].fillna("").tolist()
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=30000)
    X = vec.fit_transform(texts + [query])
    sims = cosine_similarity(X[-1], X[:-1]).ravel()
    idx = np.argsort(-sims)[:k]
    return [(texts[i], float(sims[i])) for i in idx]


def decide(query, confidence, evidence):
    """Return (auto_handle: bool, reason: str)."""
    sensitive = any(w in query.lower() for w in ESCALATE_WORDS)
    if sensitive:
        return False, "Sensitive/account-specific request"
    if confidence < 0.60:
        return False, "Low intent confidence"
    if not evidence or evidence[0][1] < 0.20:
        return False, "Weak historical evidence"
    return True, "Intent confidence and historical evidence are sufficient"


def draft_reply(query, evidence):
    """
    Conservative extractive-style reply. Deliberately does NOT invent policy,
    price, timeline, or account-specific detail. Swap in an LLM call here later,
    but keep the same constraint: only use what's in `evidence`.
    """
    if not evidence:
        return "I'm sorry you're experiencing this. I'd like to get this checked by our support team."
    return (
        "Thanks for reaching out. Based on similar support cases, the next step is to "
        "share the relevant details with support so they can check the issue. "
        "Please avoid sending passwords or payment-card details in a public message."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--message", required=True)
    ap.add_argument("--data", default="data/processed/inbound.csv")
    args = ap.parse_args()

    model = joblib.load("artifacts/intent_model.joblib")
    df = pd.read_csv(args.data)

    pred = model.predict([args.message])[0]
    conf = float(model.predict_proba([args.message]).max()) if hasattr(model, "predict_proba") else 0.0

    evidence = retrieve(args.message, df, k=3)
    auto, reason = decide(args.message, conf, evidence)
    reply = draft_reply(args.message, evidence)

    print({
        "intent": pred,
        "confidence": round(conf, 3),
        "auto_handle": auto,
        "reason": reason,
        "reply": reply,
        "evidence": evidence,
    })


if __name__ == "__main__":
    main()
