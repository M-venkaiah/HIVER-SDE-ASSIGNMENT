"""
LLM-as-judge rubric.

To actually use this for the submission:
1. Pick a stratified 30-50 example subset of your golden set.
2. Score each example YOURSELF (or a second human) on the 5 dimensions below, 0-2.
3. Send the same examples + rubric to an LLM API and get its scores.
4. Compare the two score sets (exact agreement %, and/or Spearman correlation).
5. Report the real number, whatever it is. A lower number is more credible
   than a suspiciously perfect one, because it shows you actually ran the check.

This file is a rubric definition only -- it does not call an LLM yet.
Wire up your own API call in the `score_with_llm` stub once you have a key.
"""

RUBRIC = {
    "groundedness": "Does the reply stay supported by retrieved historical evidence? (0=unsupported, 1=partial, 2=fully supported)",
    "relevance": "Does it address the customer's actual request? (0=misses issue, 1=partial, 2=direct)",
    "helpfulness": "Does it give a useful next step without overpromising? (0=not useful, 1=some help, 2=clear next step)",
    "brand_consistency": "Does the tone resemble the brand's historical style? (0=poor, 1=acceptable, 2=strong)",
    "safety": "Does it avoid unsupported policies, prices, timelines, or sensitive actions? (0=unsafe claim, 1=minor concern, 2=clean)",
}


def score_with_llm(customer_message, evidence, reply):
    """
    STUB. Replace with a real call to your chosen LLM API, passing RUBRIC,
    customer_message, evidence, and reply, and asking for a 0-2 score per
    dimension plus a one-line justification. Return a dict like:
    {"groundedness": 2, "relevance": 1, ...}
    """
    raise NotImplementedError("Wire up your LLM API call here before running the judge.")


if __name__ == "__main__":
    print("Rubric dimensions:")
    for k, v in RUBRIC.items():
        print(f"  {k}: {v}")
