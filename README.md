# Hiver SDE Intern — AI Customer Support Agent

## Status — read this first

This is a **working scaffold**, tested end-to-end against a small smoke-test
fixture (`data/smoke_sample.csv`). It is **not yet a finished submission**.

Specifically, these steps still require real work with the actual dataset,
by you, before this can be submitted:

- [ ] Download the real Kaggle dataset and run `src/prepare.py` on it
- [ ] Inspect brand conversation counts and pick ONE brand with real justification
- [ ] Revise the intent taxonomy in `src/train.py` based on what that brand's
      messages actually look like
- [ ] Generate and **manually label** the 150–250 example golden set
      (`evaluation/make_golden.py`, then edit the CSV by hand — do not skip this)
- [ ] Run `evaluation/run.py` to get real accuracy/F1 numbers
- [ ] Wire up `evaluation/judge.py` to a real LLM API and run a human-vs-LLM
      agreement check on 30–50 examples
- [ ] Write the failure analysis from your actual run's mistakes, not hypothetical ones
- [ ] Write the decision log from decisions you actually made, with your real reasons
- [ ] Fill in `reports/final_report.md` with real numbers

## What's already done and verified

- Repo structure, working pipeline: prepare → train → agent → evaluate
- Escalation logic with passing unit tests (`pytest tests/` — 4/4 pass)
- Golden-set generator that creates blank rows for you to label (never
  auto-fills labels)
- Evaluation harness that **refuses to run** if any gold label is still blank —
  this is intentional, to stop you from accidentally reporting numbers computed
  on incomplete data
- LLM-judge rubric defined, API call left as a stub for you to wire up

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Step-by-step from here

### 1. Smoke test (already verified working — do this first to confirm your environment)

```bash
cp data/smoke_sample.csv data/processed/inbound.csv
python -m src.train --input data/processed/inbound.csv
python -m src.demo
python -m pytest tests/ -v
```

You should see 3 example messages get classified and either auto-handled or
escalated, and 4 passing tests.

### 2. Get the real dataset

Download `thoughtvector/customer-support-on-twitter` from Kaggle (see
`kaggle datasets download` in the Kaggle CLI docs). Put the CSV at `data/twcs.csv`.

### 3. Inspect and pick a brand

Before running `prepare.py`, actually look at the data:

```python
import pandas as pd
df = pd.read_csv("data/twcs.csv")
print(df["author_id"].value_counts().head(20))
```

Pick a brand with enough volume and clear resolved conversations. Write down
*why* you picked it — that's a real decision-log entry.

### 4. Process your chosen brand

```bash
python -m src.prepare --input data/twcs.csv --brand AppleSupport --limit 5000
```

### 5. Revise the intent taxonomy

Open `src/train.py`, look at real messages from your processed data, and
adjust `INTENTS` and `KEYWORDS` to match what you actually see — the current
list is a starting guess, not a final answer.

### 6. Retrain

```bash
python -m src.train --input data/processed/inbound.csv
```

### 7. Build and label the golden set

```bash
python -m evaluation.make_golden --input data/processed/inbound.csv --n 200
```

Open `evaluation/golden_set.csv` and fill in `gold_intent`, `gold_escalate`,
`gold_reason` for every row, by reading the message yourself. This is the part
of the assignment that proves your evaluation isn't circular — there's no
shortcut here.

### 8. Run evaluation

```bash
python -m evaluation.run
```

### 9. LLM-as-judge + human agreement

Implement the API call in `evaluation/judge.py`, score 30–50 examples yourself
using the same rubric, and compute agreement between your scores and the LLM's.

### 10. Write it up

Fill in `reports/final_report.md` with your real numbers, real failure
examples, and real decision log.

## Baselines

- **Majority-class baseline**: predict the single most frequent intent for
  every message. Compute this from your labelled golden set — it's a couple
  lines of pandas (`df["gold_intent"].value_counts().idxmax()`), not included
  as a separate script since it's meant to be trivial and quick to write yourself.
- **TF-IDF + Logistic Regression**: implemented in `src/train.py`.

## Design principles

- Retrieval happens before generation — the reply is only allowed to use what
  the retrieval step finds, not invented policy.
- Escalation is rule-based and conservative: sensitive keywords, low
  confidence, or weak evidence all trigger escalation over guessing.
- Macro-F1 is reported alongside accuracy because intent classes are likely
  imbalanced.
