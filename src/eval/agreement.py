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
uncertain, so every r and AUC gets a 95% percentile-bootstrap confidence
interval: resample the summaries with replacement 1,000 times (seed 42, as for
the other CIs in the paper), recompute r and AUC, and take the 2.5th and 97.5th
percentiles. A metric whose AUC interval includes 0.5 is not reliably better
than chance on this data.

Writes results/metric_agreement.csv (used by src/analysis/make_figures.py).

Sensitivity check:  --without A1  rebuilds the final labels from
results/annotation_packs/all_labels.csv WITHOUT that annotator (their own
10 rows drop out; shared rows take the majority of the other four) and
reports the label distribution and the AUCs on that set. It does not write
results/metric_agreement.csv.

Run:  python -m src.eval.agreement [--model gemma3:4b] [--without A1]
"""

import argparse
from pathlib import Path

import numpy as np
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
OUT_PATH = Path("results/metric_agreement.csv")
N_BOOT, SEED = 1000, 42


def r_and_auc(y: np.ndarray, x: np.ndarray) -> tuple:
    """Point-biserial r and AUC (= Mann-Whitney U / (n_pos * n_neg)).
    If the metric has the same value everywhere, r is undefined (NaN) and the
    AUC is exactly 0.5 (every faithful/error pair is a tie): no signal."""
    pos, neg = x[y == 1], x[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, np.nan
    if np.all(x == x[0]):
        return np.nan, 0.5
    r = pointbiserialr(y, x)[0]
    auc = mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))
    return r, auc


def bootstrap_ci(y: np.ndarray, x: np.ndarray) -> tuple:
    """95% percentile-bootstrap CIs for r and AUC (resampling summaries)."""
    rng = np.random.default_rng(SEED)
    rs, aucs = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        r, auc = r_and_auc(y[idx], x[idx])
        if not np.isnan(auc):  # skip resamples with only one class
            aucs.append(auc)
        if not np.isnan(r):    # ... or with a constant metric
            rs.append(r)
    r_ci = np.percentile(rs, [2.5, 97.5]) if rs else [np.nan, np.nan]
    return r_ci, np.percentile(aucs, [2.5, 97.5])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--without", default="", help="leave out one annotator, e.g. A1")
    args = parser.parse_args()
    if args.without:
        return sensitivity(args.model, args.without)

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

    print(f"{'metric':<18} {'n':>4} {'r_pb':>7} {'95% CI':>17} {'p':>7} {'AUC':>6} {'95% CI':>15}")
    rows = []
    for m in METRICS:
        d = data[["faithful", m]].dropna()  # e.g. num_p is NaN without numbers
        y, x = d["faithful"].to_numpy(int), d[m].to_numpy(float)
        n_pos, n_neg = int(y.sum()), int((y == 0).sum())
        if n_pos == 0 or n_neg == 0:
            print(f"{m:<18} {len(d):>4}   (needs both faithful and error labels)")
            continue
        r, auc = r_and_auc(y, x)
        if np.isnan(r):
            print(f"{m:<18} {len(d):>4}   constant ({x[0]:g}) on all labelled summaries: no signal, AUC = 0.5")
            rows.append({"metric": m, "n": len(d), "n_faithful": n_pos, "n_error": n_neg, "auc": 0.5,
                         "auc_lo": 0.5, "auc_hi": 0.5, "note": "constant"})
            continue
        p = pointbiserialr(y, x)[1]
        (r_lo, r_hi), (a_lo, a_hi) = bootstrap_ci(y, x)
        print(f"{m:<18} {len(d):>4} {r:>7.3f} [{r_lo:6.3f}, {r_hi:6.3f}] {p:>7.3f} {auc:>6.3f} [{a_lo:.3f}, {a_hi:.3f}]")
        rows.append({"metric": m, "n": len(d), "n_faithful": n_pos, "n_error": n_neg,
                     "r": r, "r_lo": r_lo, "r_hi": r_hi, "p": p, "auc": auc, "auc_lo": a_lo, "auc_hi": a_hi,
                     "note": f"{int((x != np.median(x)).sum())} values differ from the median"})
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")

    # extra check: does the metric separate MAJOR errors from faithful?
    print("\nAUC faithful vs MAJOR errors only (minor errors left out):")
    for m in METRICS:
        d = data.loc[data["label"] != "minor_error", ["faithful", m]].dropna()
        r, auc = r_and_auc(d["faithful"].to_numpy(int), d[m].to_numpy(float))
        print(f"  {m:<18} n={len(d):>3}  AUC {auc:.3f}")


def sensitivity(model: str, left_out: str):
    """Final labels without one annotator, then the same r / AUC table."""
    from src.eval.merge_annotations import ALL_LABELS_PATH, majority_label
    labels = pd.read_csv(ALL_LABELS_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    labels = labels[labels["annotator"] != left_out]
    keys = ["id", "condition", "prompt_type"]
    finals = labels.groupby(keys)["label"].apply(lambda g: majority_label(list(g))[0]).reset_index()
    finals["faithful"] = finals["label"].map(LABELS)
    print(f"Without {left_out}: {len(finals)} summaries, "
          + ", ".join(f"{lab} {(finals['label'] == lab).sum()}" for lab in LABELS)
          + f"; error rate {(finals['faithful'] == 0).mean():.0%}")
    data = finals.merge(pd.read_csv(scores_path(model)), on=keys)
    for m in METRICS:
        d = data[["faithful", m]].dropna()
        y, x = d["faithful"].to_numpy(int), d[m].to_numpy(float)
        r, auc = r_and_auc(y, x)
        _, (lo, hi) = bootstrap_ci(y, x)
        print(f"  {m:<18} n={len(d):>3}  AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
