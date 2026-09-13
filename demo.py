from src.agent import retrieve, decide, draft_reply
import pandas as pd
import joblib

df = pd.read_csv("data/processed/inbound.csv")
model = joblib.load("artifacts/intent_model.joblib")

examples = [
    "My app keeps crashing after the latest update.",
    "I need a refund for this charge.",
    "I cannot log in to my account.",
]

for message in examples:
    pred = model.predict([message])[0]
    conf = float(model.predict_proba([message]).max())
    evidence = retrieve(message, df)
    auto, reason = decide(message, conf, evidence)
    print("\nMESSAGE:", message)
    print("INTENT:", pred, "CONF:", round(conf, 3))
    print("DECISION:", "AUTO-HANDLE" if auto else "ESCALATE", "-", reason)
    print("DRAFT:", draft_reply(message, evidence))
