"""
Prepare inbound customer messages for a single chosen brand from the real
Kaggle twcs.csv file. Run the smoke test first (against data/smoke_sample.csv)
to confirm the pipeline works, then re-run against the real file once you
download it.
"""
import argparse
import os
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--brand", required=True, help="author_id of the brand account, e.g. AppleSupport")
    ap.add_argument("--limit", type=int, default=5000)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    required = {"tweet_id", "author_id", "inbound", "created_at", "text",
                "response_tweet_id", "in_response_to_tweet_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    brand_rows = df[df["author_id"].astype(str).eq(args.brand)].copy()
    if brand_rows.empty:
        raise ValueError(f"No outbound rows found for brand={args.brand}. Check exact author_id spelling.")

    inbound = df[df["inbound"].astype(bool)].copy()
    inbound["tweet_id"] = inbound["tweet_id"].astype(str)
    replied_to = set(df["in_response_to_tweet_id"].dropna().astype(str))
    inbound = inbound[inbound["tweet_id"].isin(replied_to)].copy()
    inbound = inbound.head(args.limit)

    os.makedirs("data/processed", exist_ok=True)
    inbound.to_csv("data/processed/inbound.csv", index=False)
    brand_rows.to_csv("data/processed/outbound_brand.csv", index=False)
    print(f"Saved {len(inbound)} inbound examples and {len(brand_rows)} brand replies.")


if __name__ == "__main__":
    main()
