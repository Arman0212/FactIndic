"""
FactIndic - gpt-oss-120b vs gemma3:4b on the same articles
==========================================================
The Groq model was run on only 50 articles and two conditions (hi, hinglish),
so we compare it with gemma3:4b on exactly those (article, condition, prompt)
combinations.

  - wrong-language rate: over ALL summaries of the 50 articles (per model);
  - every other column: mean over the rows where BOTH models passed the
    language/script check (so both models are scored on identical inputs).
    num_p is undefined for summaries without numbers, so its mean uses fewer
    rows (n_num). NLI scores are comparable here because both models are
    compared within the same condition.
  - latency is NOT a fair speed comparison: gemma runs locally on an 8 GB Mac,
    gpt-oss-120b on Groq's servers (network time included).

Writes results/model_comparison.csv.

Run:  python -m src.eval.compare_models
"""

from pathlib import Path

import pandas as pd

from src.data.make_hinglish import read_jsonl
from src.evaluate import scores_path
from src.summarize import output_path

MODELS = ["gemma3:4b", "openai/gpt-oss-120b"]
BASE = "openai/gpt-oss-120b"  # the model that defines the article subset
KEY = ["id", "condition", "prompt_type"]
METRICS = ["num_p", "num_r", "ent_p", "ent_r", "nli_chunk_score", "nli_window_score",
           "rougeL_src", "bertscore_src", "summary_words", "latency_s"]
OUT_PATH = Path("results/model_comparison.csv")


def main():
    summaries = {m: pd.DataFrame(read_jsonl(output_path(m))) for m in MODELS}
    scores = {m: pd.read_csv(scores_path(m)) for m in MODELS}
    subset = summaries[BASE][KEY].drop_duplicates()  # 50 articles x hi/hinglish x 2 prompts

    # rows where both models produced a summary in the right language
    both_ok = subset
    for m in MODELS:
        ok = summaries[m].loc[summaries[m]["script_ok"], KEY]
        both_ok = both_ok.merge(ok, on=KEY)

    rows = []
    for (cond, ptype), cell in subset.groupby(["condition", "prompt_type"]):
        common = both_ok[(both_ok["condition"] == cond) & (both_ok["prompt_type"] == ptype)]
        for m in MODELS:
            s = summaries[m].merge(cell, on=KEY)
            sc = scores[m].merge(common, on=KEY)
            row = {"condition": cond, "prompt_type": ptype, "model": m,
                   "n_articles": len(cell), "wrong_language_pct": 100 * (~s["script_ok"]).mean(),
                   "n_scored": len(sc), "n_num": int(sc["num_p"].notna().sum())}
            row.update({k: sc[k].mean() for k in METRICS})
            rows.append(row)

    table = pd.DataFrame(rows).round(3)
    table.to_csv(OUT_PATH, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.to_string(index=False))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
