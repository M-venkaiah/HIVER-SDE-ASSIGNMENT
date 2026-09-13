# Hiver SDE Intern — AI Customer Support Agent

**Candidate:** Manda Venkaiah
**Role:** SDE Intern

> ⚠️ This is a TEMPLATE. Every `[TODO: ...]` below needs a real answer from
> your actual run on the actual dataset. Do not submit this file with any
> `[TODO]` still in it — an unfilled template is worse than a short but real
> report.

## 1. Problem framing

Brand chosen: `[TODO: which brand, and why — cite actual conversation counts you observed]`

What "good" means for this brand: `[TODO: e.g., for a technical-support brand, groundedness and safe escalation matter more than raw accuracy]`

What I chose not to build: `[TODO: e.g., no embedding-based retrieval, no fine-tuning, no multi-brand support — and why that was the right call for this scope]`

## 2. Results vs. baselines

| System | Accuracy | Macro F1 |
|---|---|---|
| Majority-class baseline | `[TODO]` | `[TODO]` |
| TF-IDF + Logistic Regression | `[TODO]` | `[TODO]` |
| Full agent | `[TODO]` | `[TODO]` |

Reply-quality (LLM-judge, 0-2 scale, averaged over golden set):

| Dimension | Score |
|---|---|
| Groundedness | `[TODO]` |
| Relevance | `[TODO]` |
| Helpfulness | `[TODO]` |
| Brand consistency | `[TODO]` |
| Safety | `[TODO]` |

Human-vs-LLM judge agreement: `[TODO: exact-match % and/or correlation, from a real 30-50 example comparison]`

## 3. Failure analysis — top 5

For each: real example, what happened, why, proposed fix.

1. `[TODO]`
2. `[TODO]`
3. `[TODO]`
4. `[TODO]`
5. `[TODO]`

## 4. What is misleading about my headline number?

`[TODO: write this yourself once you have a real number — think about
evaluation-set size, single-brand scope, class imbalance, judge-generator
shared blind spots, whether reply safety is captured by intent accuracy]`

## 5. What I'd do with one more week

`[TODO]`

## 6. Decision log (10-15 items)

Format: decision — real reason, grounded in something you actually observed.

1. `[TODO]`
2. `[TODO]`
...

## Sources

- Customer Support on Twitter — Kaggle / Thought Vector:
  https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
- `[TODO: cite anything else you borrowed — code, prompts, articles]`
