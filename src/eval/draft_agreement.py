"""
FactIndic - how far do A1's labels agree with the LLM drafts?
=============================================================
A1 had LLM-written suggestion labels ("drafts") for the 60 summaries. To
report how much they could have influenced A1, we compare with the drafts:

  BLIND pass    results/annotation_packs/labels_A1.csv
                label / error_type locked BEFORE the draft was revealed
                (enforced by label_server.py), plus optional revisions made
                after seeing the draft (revised_label).
  ASSISTED pass results/annotation_packs/labels_A1_assisted.csv  (if present)
                A1's first pass, made with the draft visible; discarded and
                kept only for this comparison.

For each pass: exact label agreement, Cohen's kappa (3 labels and faithful vs
error), the confusion matrix, error-type agreement on rows where both say
"error", and (blind pass) how many labels A1 revised after seeing the draft
and whether the revision moved towards the draft.

Run:  python -m src.eval.draft_agreement
"""

import pandas as pd

from src.eval.make_annotation_packs import ANNOTATION_PATH, KEY, PACK_DIR, OWNER, labels_path, read_csv
from src.eval.merge_annotations import binary, cohen_kappa

ASSISTED_PATH = PACK_DIR / "labels_A1_assisted.csv"
ORDER = ["faithful", "minor_error", "major_error"]


def compare(name: str, labels: pd.DataFrame, drafts: pd.DataFrame):
    d = labels[labels["label"] != ""].merge(drafts, on=KEY)
    print(f"\n== {name}: {len(d)} labelled rows")
    if len(d) < 2:
        print("   (not enough labels yet)")
        return
    mine, draft = list(d["label"]), list(d["draft_label"])
    agree = sum(a == b for a, b in zip(mine, draft))
    print(f"   label agreement with draft:   {agree}/{len(d)} ({100 * agree / len(d):.0f}%)")
    print(f"   Cohen's kappa, 3 labels:      {cohen_kappa(mine, draft):.3f}")
    print(f"   Cohen's kappa, binary:        {cohen_kappa(binary(mine), binary(draft)):.3f}")
    both_err = d[(d["label"] != "faithful") & (d["draft_label"] != "faithful")]
    if len(both_err):
        same_type = int((both_err["error_type"] == both_err["draft_error_type"]).sum())
        print(f"   error type agreement (both say error): {same_type}/{len(both_err)}")
    table = pd.crosstab(pd.Categorical(d["draft_label"], ORDER), pd.Categorical(d["label"], ORDER),
                        rownames=["draft"], colnames=["A1"], dropna=False)
    print("   " + table.to_string().replace("\n", "\n   "))
    print("   by condition (agree / n): " + ", ".join(
        f"{c} {int((g['label'] == g['draft_label']).sum())}/{len(g)}" for c, g in d.groupby("condition")))

    if "revised_label" in d.columns:
        revised = d[d["revised_label"] != ""]
        towards = int((revised["revised_label"] == revised["draft_label"]).sum())
        print(f"   revised after seeing the draft: {len(revised)}/{len(d)}"
              f" ({towards} of them now equal to the draft)")


def main():
    drafts = read_csv(ANNOTATION_PATH)[KEY + ["draft_label", "draft_error_type"]]
    blind = read_csv(labels_path(OWNER))
    if "locked_at" in blind.columns:
        blind = blind[blind["locked_at"] != ""]
    compare("BLIND pass (label locked before the draft was shown)", blind, drafts)
    if ASSISTED_PATH.exists():
        compare("ASSISTED pass (draft visible, discarded)", read_csv(ASSISTED_PATH), drafts)


if __name__ == "__main__":
    main()
