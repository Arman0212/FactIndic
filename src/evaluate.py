"""
FactIndic - Phase 4: evaluate all summaries
===========================================
Runs every metric on every summary that passed the language/script check
(script_ok) and writes:

  results/scores_<model>.csv         one row per summary (all metrics)
  results/summary_table_<model>.csv  mean and 95% bootstrap CI per
                                     condition x prompt
  results/cache/gold_entities.json   Stanza entities per article (cache)

Metrics (see src/eval/*.py):
  numbers   num_p, num_r, num_f1               (number precision/recall/F1)
  entities  ent_p, ent_r, ent_f1               (named entities, across scripts)
  NLI       nli_chunk_* (primary for now) and nli_window_* (2-sentence
            premises): score and unsupported sentences (mDeBERTa, SummaC-style)
  overlap   rouge1/2/L vs source, vs reference (hi only), bertscore_src
  other     summary_words, compression_ratio, latency_s, wrong-language rate

Resumable: summaries already in scores_<model>.csv are skipped, and results
are appended in small batches, so a crash loses at most one batch.

Run:
    python -m src.evaluate --model gemma3:4b --limit 20   # quick test
    python -m src.evaluate --model gemma3:4b              # everything
    python -m src.evaluate --model openai/gpt-oss-120b
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.fix_flagged import FINAL_PATH
from src.data.make_hinglish import read_jsonl
from src.eval.entities import entity_scores, stanza_entities
from src.eval.nli import NLIScorer
from src.eval.numbers import number_scores
from src.eval.overlap import BertScorer, rouge_f1
from src.summarize import CONDITIONS, output_path

CACHE_DIR = Path("results/cache")
GOLD_PATH = CACHE_DIR / "gold_entities.json"
BATCH = 16          # summaries per batch (BERTScore runs per batch)
N_BOOTSTRAP = 1000  # resamples for the confidence intervals
SEED = 42

# Column order of the per-summary CSV
COLUMNS = [
    "id", "condition", "prompt_type", "model",
    "num_p", "num_r", "num_f1", "num_unsupported",
    "ent_p", "ent_r", "ent_f1", "ent_n_gold", "ent_n_summary", "ent_unsupported",
    "nli_n_sent",
    "nli_chunk_score", "nli_chunk_n_unsupported", "nli_chunk_pct_unsupported",
    "nli_window_score", "nli_window_n_unsupported", "nli_window_pct_unsupported",
    "rouge1_src", "rouge2_src", "rougeL_src",
    "rouge1_ref", "rouge2_ref", "rougeL_ref",
    "bertscore_src",
    "summary_words", "source_words", "compression_ratio", "latency_s",
]

# Metrics shown in the aggregate table
TABLE_METRICS = [
    "num_p", "num_r", "num_f1", "ent_p", "ent_r", "ent_f1",
    "nli_chunk_score", "nli_chunk_pct_unsupported",
    "nli_window_score", "nli_window_pct_unsupported",
    "rouge1_src", "rouge2_src", "rougeL_src", "rouge1_ref", "rouge2_ref", "rougeL_ref",
    "bertscore_src", "summary_words", "compression_ratio", "latency_s",
]


def scores_path(model: str) -> Path:
    return output_path(model).with_name(output_path(model).name.replace("summaries_", "scores_")).with_suffix(".csv")


def table_path(model: str) -> Path:
    return scores_path(model).with_name(scores_path(model).name.replace("scores_", "summary_table_"))


def load_gold_entities(articles: dict, ids: set) -> dict:
    """Stanza entities of text_hi per article, cached in a JSON file."""
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8")) if GOLD_PATH.exists() else {}
    missing = [i for i in ids if i not in gold]
    for i in tqdm(missing, desc="Gold entities (Stanza)"):
        gold[i] = stanza_entities(articles[i]["text_hi"])
    if missing:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        GOLD_PATH.write_text(json.dumps(gold, ensure_ascii=False, indent=1), encoding="utf-8")
    return gold


def score_batch(batch: list, articles: dict, gold: dict, nli, bert) -> list:
    """All metrics for a list of summaries; returns one dict per summary."""
    sources = [articles[s["id"]][CONDITIONS[s["condition"]]] for s in batch]
    bert_f1 = bert.f1([s["summary"] for s in batch], sources)
    rows = []
    for s, source, bs in zip(batch, sources, bert_f1):
        art = articles[s["id"]]
        row = {k: s[k] for k in ("id", "condition", "prompt_type", "model",
                                 "summary_words", "source_words",
                                 "compression_ratio", "latency_s")}
        row.update(number_scores(source, s["summary"]))
        row.update(entity_scores(gold[s["id"]], source, s["summary"]))
        row.update(nli.score(source, s["summary"]))
        row.update(rouge_f1(source, s["summary"], "src"))
        if s["condition"] == "hi":  # the reference summary is Devanagari Hindi
            row.update(rouge_f1(art["ref_summary_hi"], s["summary"], "ref"))
        row["bertscore_src"] = bs
        rows.append(row)
    return rows


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple:
    """Mean and 95% percentile-bootstrap confidence interval (NaNs dropped)."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (math.nan, math.nan, math.nan)
    means = rng.choice(values, size=(N_BOOTSTRAP, len(values)), replace=True).mean(axis=1)
    return (values.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5))


def aggregate(model: str, summaries: list):
    """Write the summary table: one row per condition x prompt_type."""
    scores = pd.read_csv(scores_path(model))
    rng = np.random.default_rng(SEED)
    rows = []
    for (cond, ptype), group in scores.groupby(["condition", "prompt_type"], sort=True):
        all_summ = [s for s in summaries if s["condition"] == cond and s["prompt_type"] == ptype]
        row = {
            "condition": cond, "prompt_type": ptype, "n_scored": len(group),
            "n_total": len(all_summ),
            "wrong_language_pct": 100 * sum(not s["script_ok"] for s in all_summ) / max(1, len(all_summ)),
        }
        for m in TABLE_METRICS:
            mean, lo, hi = bootstrap_ci(group[m].to_numpy(dtype=float), rng)
            row[m], row[f"{m}_lo"], row[f"{m}_hi"] = mean, lo, hi
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(table_path(model), index=False, float_format="%.4f")

    # Short printout of the main metrics
    show = ["condition", "prompt_type", "n_scored", "wrong_language_pct",
            "num_p", "num_f1", "ent_p", "ent_r", "nli_chunk_score", "nli_window_score",
            "rougeL_src", "bertscore_src"]
    print(table[show].round(3).to_string(index=False))
    print(f"Saved {table_path(model)}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate summaries with all factuality metrics.")
    parser.add_argument("--model", default="gemma3:4b", help="model whose summaries to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="only the first N script_ok summaries")
    args = parser.parse_args()

    articles = {a["id"]: a for a in read_jsonl(FINAL_PATH)}
    summaries = read_jsonl(output_path(args.model))
    todo = [s for s in summaries if s["script_ok"]]  # wrong-language summaries are excluded
    if args.limit:
        todo = todo[: args.limit]

    out = scores_path(args.model)
    done = set()
    if out.exists():
        done = {(r["id"], r["condition"], r["prompt_type"]) for r in csv.DictReader(open(out, encoding="utf-8"))}
    todo = [s for s in todo if (s["id"], s["condition"], s["prompt_type"]) not in done]
    print(f"{args.model}: {len(done)} summaries already scored, {len(todo)} to go -> {out}")

    if todo:
        gold = load_gold_entities(articles, {s["id"] for s in todo})
        nli, bert = NLIScorer(), BertScorer()

        new_file = not out.exists()
        with open(out, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if new_file:
                writer.writeheader()

            for start in tqdm(range(0, len(todo), BATCH), desc="Scoring batches"):
                batch = todo[start:start + BATCH]
                writer.writerows(score_batch(batch, articles, gold, nli, bert))
                f.flush()  # make each batch durable

    aggregate(args.model, summaries)


if __name__ == "__main__":
    main()
