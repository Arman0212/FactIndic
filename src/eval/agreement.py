"""
FactIndic - Phase 4: agreement of automatic metrics with human labels
=====================================================================
Run this AFTER merging the five annotators' labels:
    python -m src.eval.merge_annotations
which writes the FINAL label of every summary (majority vote on the 10
shared rows) into the `label` column of results/manual_annotation.csv.
It answers: which automatic score
best separates faithful summaries from summaries with errors? We use this to
choose the primary NLI variant (nli_chunk vs nli_window).

Human label as a binary variable:  faithful = 1,  minor_error or major_error = 0.

For each metric:
  - point-biserial correlation r between the metric and the binary label
    (= Pearson correlation with a 0/1 variable). r > 0: higher score goes with
    "faithful", which is what a good faithfulness metric should show.
  - AUC: the probability that a randomly chosen faithful summary gets a
    HIGHER score than a randomly chosen erroneous one (0.5 = chance, 1 = perfect).
    Computed from the Mann-Whitney U statistic: AUC = U / (n_faithful * n_error).

With only 60 labelled summaries the differences between metrics are
uncertain; we print n per metric and the p-value of r.

Run:  python -m src.eval.agreement [--model gemma3:4b]
"""

import argparse

import pandas as pd
from scipy.stats import mannwhitneyu, pointbiserialr

from src.eval.make_annotation_set import OUT_PATH as ANNOTATION_PATH
from src.evaluate import scores_path

# Metrics to compare: both NLI variants first, the others for context.
METRICS = [
    "nli_chunk_score", "nli_window_score",
    "num_p", "ent_p", "rougeL_src", "bertscore_src",
]
LABELS = {"faithful": 1, "minor_error": 0, "major_error": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma3:4b")
    args = parser.parse_args()

    ann = pd.read_csv(ANNOTATION_PATH, encoding="utf-8-sig")
    if "annotator" not in ann.columns:
        print("Labels not merged yet - run: python -m src.eval.merge_annotations")
        return
    ann["label"] = ann["label"].astype(str).str.strip().str.lower()
    ann = ann[ann["label"].isin(LABELS)]
    if ann.empty:
        print(f"No final labels yet in {ANNOTATION_PATH} - run src.eval.merge_annotations first.")
        return
    ann["faithful"] = ann["label"].map(LABELS)

    scores = pd.read_csv(scores_path(args.model))
    data = ann.merge(scores, on=["id", "condition", "prompt_type"], how="inner")
    print(f"{len(data)} labelled summaries: {int(data['faithful'].sum())} faithful, "
          f"{int((data['faithful'] == 0).sum())} with errors\n")

    print(f"{'metric':<18} {'n':>4} {'r_pb':>7} {'p':>7} {'AUC':>6}")
    for m in METRICS:
        d = data[["faithful", m]].dropna()  # e.g. num_p is NaN without numbers
        n_pos, n_neg = int(d["faithful"].sum()), int((d["faithful"] == 0).sum())
        if n_pos == 0 or n_neg == 0:
            print(f"{m:<18} {len(d):>4}   (needs both faithful and error labels)")
            continue
        r, p = pointbiserialr(d["faithful"], d[m])
        u = mannwhitneyu(d.loc[d["faithful"] == 1, m], d.loc[d["faithful"] == 0, m]).statistic
        auc = u / (n_pos * n_neg)
        print(f"{m:<18} {len(d):>4} {r:>7.3f} {p:>7.3f} {auc:>6.3f}")


if __name__ == "__main__":
    main()
