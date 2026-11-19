"""
FactIndic - Step 1: load Hindi articles from XL-Sum
===================================================
Source: csebuetnlp/xlsum (BBC Hindi news articles + one-sentence summaries).

We take the *test* split, keep articles of 150-500 words (long enough to have
several facts, short enough to fit comfortably in an LLM prompt), sample 300
with a fixed seed (42) so the sample is reproducible, and save them to
data/raw/hindi_300.jsonl.

Run:  python -m src.data.load_xlsum
"""

import json
import random
from pathlib import Path

import pandas as pd

from src.text_utils import normalize

OUT_PATH = Path("data/raw/hindi_300.jsonl")
MIN_WORDS, MAX_WORDS = 150, 500
N_SAMPLES = 300
SEED = 42


def load_test_split() -> pd.DataFrame:
    """Load the Hindi test split as a pandas DataFrame.

    XL-Sum is published with a Python loading script. Newer versions of the
    `datasets` library (>= 4.0) no longer run loading scripts, so if that fails
    we read the Parquet copy that Hugging Face auto-generates on the
    refs/convert/parquet branch of the dataset repo.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("csebuetnlp/xlsum", "hindi", split="test")
        return ds.to_pandas()
    except Exception as e:
        print(f"[info] load_dataset failed ({type(e).__name__}); using parquet fallback")
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="csebuetnlp/xlsum",
            repo_type="dataset",
            revision="refs/convert/parquet",
            filename="hindi/test/0000.parquet",
        )
        return pd.read_parquet(path)


def main():
    df = load_test_split()
    print(f"Hindi test split: {len(df)} articles")

    # Build cleaned records and keep only articles in the word-count range.
    records = []
    for _, row in df.iterrows():
        text = normalize(row["text"])
        n_words = len(text.split())
        if MIN_WORDS <= n_words <= MAX_WORDS:
            records.append({
                "id": row["id"],
                "title": normalize(row["title"]),
                "text_hi": text,
                "ref_summary_hi": normalize(row["summary"]),
            })
    print(f"Articles with {MIN_WORDS}-{MAX_WORDS} words: {len(records)}")

    # Reproducible random sample.
    random.seed(SEED)
    sample = random.sample(records, N_SAMPLES)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for rec in sample:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(sample)} articles to {OUT_PATH}")


if __name__ == "__main__":
    main()
