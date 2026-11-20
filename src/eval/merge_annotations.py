"""
FactIndic - merge the five annotators' labels and measure agreement
===================================================================
Reads from results/annotation_packs/:
  labels_A1.csv            A1's labels (from src/eval/label_server.py)
  labels_A2.csv .. A5.csv  teammates' downloads from their offline pages
  assignment.csv           who labels which row (make_annotation_packs.py)

Writes:
  results/annotation_packs/all_labels.csv  every individual label (one row
      per annotator per summary), so nothing is lost by the merge;
  results/manual_annotation.csv            one FINAL label per summary, plus
      the columns  set (shared / single)  and  annotator  (who labelled it).
      Single rows: the label of the one annotator.
      Shared rows: the MAJORITY label of the annotators; a tie goes to the
      more severe label (major_error > minor_error > faithful), because a
      summary that some annotators find wrong should not count as clean.
      The error type is the most common one among annotators who chose the
      final label; notes are joined as "A1: ... | A3: ...".
  src/eval/agreement.py then uses these final labels.

Agreement on the 10 shared rows (3 labels: faithful / minor / major):
  - Fleiss' kappa: agreement of ALL raters together, corrected for chance.
      P_i   = (sum_j n_ij^2 - n) / (n (n - 1))   agreement on item i
      p_j   = share of all ratings that are label j
      kappa = (mean P_i - sum p_j^2) / (1 - sum p_j^2)
  - Cohen's kappa for every PAIR of annotators:
      kappa = (p_observed - p_expected) / (1 - p_expected),
      p_expected = sum over labels of p_A(label) * p_B(label).
  - Krippendorff's alpha (nominal): like Fleiss' kappa but it allows MISSING
    ratings (A3 left one shared row empty), so it uses all 10 shared rows.
      every item with m >= 2 ratings adds each ordered pair of its ratings
      (c, k) to a coincidence table o_ck with weight 1 / (m - 1);
      n_c = sum_k o_ck,  n = sum_c n_c
      alpha = 1 - (n - 1) * sum_{c != k} o_ck / sum_{c != k} n_c n_k
    Fleiss' kappa needs the same raters on every item, so it is computed on
    the rows labelled by all five (a missing rating is never guessed).
  Both are also printed for the binary version (faithful vs any error), which
  is what agreement.py uses. Rough reading (Landis & Koch 1977): < 0.2 slight,
  0.2-0.4 fair, 0.4-0.6 moderate, 0.6-0.8 substantial, > 0.8 almost perfect.
  With only 10 shared items these numbers are very uncertain.

Missing label files are reported and skipped, so the script can be run while
teammates are still labelling. It never changes the label files themselves,
so it can be re-run any time.

Run:  python -m src.eval.merge_annotations
"""

from collections import Counter
from itertools import combinations

import pandas as pd

from src.eval.make_annotation_packs import (
    ANNOTATION_PATH, ANNOTATORS, KEY, PACK_DIR, labels_path, load_assignment, read_csv,
    write_csv,
)

ALL_LABELS_PATH = PACK_DIR / "all_labels.csv"
SEVERITY = {"faithful": 0, "minor_error": 1, "major_error": 2}
ERROR_TYPES = {"", "entity", "number", "added_info", "meaning", "omission", "other"}


# ---------------------------------------------------------------- statistics
def majority_label(labels: list) -> tuple:
    """Most frequent label; a tie goes to the most severe tied label.
    Returns (label, was_tie)."""
    counts = Counter(labels)
    best = max(counts.values())
    tied = [lab for lab, c in counts.items() if c == best]
    return max(tied, key=SEVERITY.get), len(tied) > 1


def fleiss_kappa(items: list, categories: list) -> float:
    """items: one list of labels per item, every item rated by the same n raters."""
    n = len(items[0])
    counts = [[item.count(c) for c in categories] for item in items]          # n_ij
    p_item = [(sum(x * x for x in row) - n) / (n * (n - 1)) for row in counts]  # P_i
    p_cat = [sum(row[j] for row in counts) / (len(items) * n) for j in range(len(categories))]
    p_bar, p_e = sum(p_item) / len(items), sum(p * p for p in p_cat)
    return float("nan") if p_e == 1 else (p_bar - p_e) / (1 - p_e)


def krippendorff_alpha(items: list) -> float:
    """Nominal Krippendorff's alpha. items: one list of labels per item;
    items may have different numbers of ratings (missing ones left out)."""
    o = Counter()  # coincidence table o[(c, k)]
    for item in items:
        m = len(item)
        if m < 2:
            continue  # a single rating cannot be compared with anything
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[(item[i], item[j])] += 1 / (m - 1)
    n_c = Counter()
    for (c, _), v in o.items():
        n_c[c] += v
    n = sum(n_c.values())
    disagree = sum(v for (c, k), v in o.items() if c != k)
    expected = sum(n_c[c] * n_c[k] for c in n_c for k in n_c if c != k)
    return float("nan") if expected == 0 else 1 - (n - 1) * disagree / expected


def cohen_kappa(a: list, b: list) -> float:
    """Agreement of two raters on the same items, corrected for chance."""
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    p_e = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b))
    return float("nan") if p_e == 1 else (p_o - p_e) / (1 - p_e)


def binary(labels: list) -> list:
    return ["faithful" if lab == "faithful" else "error" for lab in labels]


# ---------------------------------------------------------------- reading
def read_labels(assign: pd.DataFrame) -> pd.DataFrame:
    """All annotators' labels in one long table (only non-empty labels)."""
    parts = []
    for a in ANNOTATORS:
        path = labels_path(a)
        if not path.exists():
            print(f"  {a}: {path.name} not found yet - skipped")
            continue
        df = read_csv(path)
        df["annotator"] = a  # trust the file name, not the column
        df["label"] = df["label"].str.strip().str.lower()
        df["error_type"] = df["error_type"].str.strip().str.lower()
        if "label_source" not in df.columns:
            df["label_source"] = "blind_pack"  # teammates never see drafts
        bad = df[~df["label"].isin(set(SEVERITY) | {""}) | ~df["error_type"].isin(ERROR_TYPES)]
        for _, r in bad.iterrows():
            print(f"  {a}: unknown label/error type {r['label']!r}/{r['error_type']!r} for {r['id']} - skipped")
        df = df.drop(bad.index)
        # keep only rows that were assigned to this annotator
        mine = assign[assign["annotators"].str.split(";").apply(lambda x: a in x)]
        df = df.merge(mine[KEY], on=KEY, how="inner")
        done = df[df["label"] != ""]
        print(f"  {a}: {len(done)}/{len(mine)} labelled")
        parts.append(done)
    cols = KEY + ["annotator", "label", "error_type", "notes", "label_source"]
    return pd.concat(parts)[cols] if parts else pd.DataFrame(columns=cols)


# ---------------------------------------------------------------- merging
def final_row(group: pd.DataFrame) -> dict:
    """Final label for one summary from its annotators' labels."""
    if len(group) == 1:
        r = group.iloc[0]
        return {"annotator": r["annotator"], "label": r["label"], "error_type": r["error_type"],
                "notes": r["notes"], "label_source": r["label_source"], "tie": False}
    label, tie = majority_label(list(group["label"]))
    types = [t for t in group.loc[group["label"] == label, "error_type"] if t]
    etype = Counter(types).most_common(1)[0][0] if types and label != "faithful" else ""
    notes = " | ".join(f"{r['annotator']}: {r['notes']}" for _, r in group.iterrows() if r["notes"])
    return {"annotator": ";".join(group["annotator"]), "label": label, "error_type": etype,
            "notes": notes, "label_source": "majority_tie" if tie else "majority", "tie": tie}


def report_agreement(long: pd.DataFrame, assign: pd.DataFrame):
    shared = long.merge(assign.loc[assign["set"] == "shared", KEY], on=KEY)
    raters = sorted(shared["annotator"].unique())
    table = shared.pivot_table(index=KEY, columns="annotator", values="label", aggfunc="first")
    complete = table.dropna()
    print(f"\nAgreement on shared rows: {len(raters)} annotators ({', '.join(raters)}), "
          f"{len(table)} rows, {len(complete)} labelled by all of them, "
          f"{int(table.isna().sum().sum())} rating(s) missing")
    if len(raters) >= 2 and len(complete) >= 2:
        items = complete.values.tolist()
        print(f"  Fleiss' kappa, 3 labels         (n={len(items)} complete rows): "
              f"{fleiss_kappa(items, list(SEVERITY)):.3f}")
        print(f"  Fleiss' kappa, faithful vs error (n={len(items)} complete rows): "
              f"{fleiss_kappa([binary(i) for i in items], ['faithful', 'error']):.3f}")
        all_items = [[x for x in row if isinstance(x, str)] for row in table.values.tolist()]
        print(f"  Krippendorff's alpha, 3 labels         (all {len(all_items)} rows): "
              f"{krippendorff_alpha(all_items):.3f}")
        print(f"  Krippendorff's alpha, faithful vs error (all {len(all_items)} rows): "
              f"{krippendorff_alpha([binary(i) for i in all_items]):.3f}")
    if len(raters) < 5:
        print("  (not all five annotators yet - these are not the final numbers)")

    print("\n  Pairwise Cohen's kappa (3 labels / binary), on rows both labelled:")
    for a, b in combinations(raters, 2):
        both = table[[a, b]].dropna()
        if len(both) < 2:
            continue
        k3 = cohen_kappa(list(both[a]), list(both[b]))
        k2 = cohen_kappa(binary(list(both[a])), binary(list(both[b])))
        print(f"    {a}-{b}: {k3:6.3f} / {k2:6.3f}   (n={len(both)})")


def report_error_types(long: pd.DataFrame, out: pd.DataFrame):
    """Error-type statistics. A2 and A3 gave labels but no error types, so
    the types come only from annotators who gave them; n is always printed."""
    typed_by = sorted(a for a, g in long.groupby("annotator")
                      if (g.loc[g["label"] != "faithful", "error_type"] != "").any())
    print(f"\nError types (only annotators who gave types: {', '.join(typed_by)})")
    for a in typed_by:
        g = long[(long["annotator"] == a) & (long["label"] != "faithful")]
        print(f"  {a} ({len(g)} error labels): " + ", ".join(f"{t} {c}" for t, c in
                                                         g["error_type"].value_counts().items()))
    err = out[out["label"].isin(["minor_error", "major_error"])]
    typed = err[err["error_type"] != ""]
    print(f"  FINAL labels: {len(typed)} of {len(err)} error summaries have a type: "
          + ", ".join(f"{t} {c}" for t, c in typed["error_type"].value_counts().items()))

    print("\nFinal labels by condition x prompt (faithful / minor / major ; typed errors)")
    for (c, p), g in out[out["label"] != ""].groupby(["condition", "prompt_type"]):
        e = g[g["label"] != "faithful"]
        types = e.loc[e["error_type"] != "", "error_type"].value_counts()
        print(f"  {c:<9} {p:<8} " + " / ".join(str((g['label'] == lab).sum()) for lab in SEVERITY)
              + f" ; {int(types.sum())}/{len(e)} typed: "
              + ", ".join(f"{t} {n}" for t, n in types.items()))


def main():
    ann = read_csv(ANNOTATION_PATH)
    assign = load_assignment(ann)
    print("Label files:")
    long = read_labels(assign)
    long = long.merge(assign[KEY + ["set"]], on=KEY)
    write_csv(long[KEY + ["set", "annotator", "label", "error_type", "notes", "label_source"]], ALL_LABELS_PATH)

    report_agreement(long, assign)

    finals = {k: final_row(g) for k, g in long.groupby(KEY, sort=False)}
    ties = [k for k, f in finals.items() if f["tie"]]
    if ties:
        print(f"\n  {len(ties)} shared row(s) had a tie (more severe label used): "
              + ", ".join(f"{k[0]} ({k[1]}/{k[2]})" for k in ties))

    # rebuild manual_annotation.csv: same rows and drafts, final labels
    out = ann.drop(columns=[c for c in ["set", "annotator", "label", "error_type", "notes", "label_source"]
                            if c in ann.columns])
    out = out.merge(assign[KEY + ["set"]], on=KEY, how="left")
    for col in ["annotator", "label", "error_type", "notes", "label_source"]:
        out[col] = [finals.get(tuple(r[k] for k in KEY), {}).get(col, "") for _, r in out.iterrows()]
    order = KEY + ["source", "summary", "set", "annotator", "label", "error_type", "notes",
                   "draft_label", "draft_error_type", "draft_note", "label_source"]
    write_csv(out[order], ANNOTATION_PATH)

    done = out[out["label"] != ""]
    print(f"\nFinal labels: {len(done)}/{len(out)} summaries  "
          + "  ".join(f"{lab}: {(done['label'] == lab).sum()}" for lab in SEVERITY))
    print(f"Wrote {ALL_LABELS_PATH} ({len(long)} individual labels) and {ANNOTATION_PATH}")
    report_error_types(long, out)
    if len(done) < len(out):
        print("Some summaries have no label yet; agreement.py uses only the labelled ones.")


if __name__ == "__main__":
    main()
