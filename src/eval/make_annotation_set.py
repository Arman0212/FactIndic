"""
FactIndic - Phase 4: manual annotation set
==========================================
Writes results/manual_annotation.csv: 60 summaries for human checking,
stratified as 10 per condition x prompt_type (random, seed 42), drawn from
the summaries that passed the language/script check.

Columns: id, condition, prompt_type, source, summary, and three EMPTY columns
to fill in by hand:
  label       faithful / minor_error / major_error
  error_type  entity / number / added_info / meaning / omission / other
  notes       free text

The file is never overwritten (it will contain hand labels); delete it
yourself if you really want a new sample.

Run:  python -m src.eval.make_annotation_set [--model gemma3:4b]
"""

import argparse
import csv
import random
from pathlib import Path

from src.data.fix_flagged import FINAL_PATH
from src.data.make_hinglish import read_jsonl
from src.summarize import CONDITIONS, PROMPTS, output_path

OUT_PATH = Path("results/manual_annotation.csv")
PER_CELL, SEED = 10, 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma3:4b")
    args = parser.parse_args()

    if OUT_PATH.exists():
        print(f"{OUT_PATH} already exists - not overwriting (it may contain your labels).")
        return

    articles = {a["id"]: a for a in read_jsonl(FINAL_PATH)}
    summaries = [s for s in read_jsonl(output_path(args.model)) if s["script_ok"]]

    random.seed(SEED)
    rows = []
    for cond in CONDITIONS:
        for ptype in PROMPTS:
            cell = [s for s in summaries if s["condition"] == cond and s["prompt_type"] == ptype]
            for s in random.sample(cell, PER_CELL):
                rows.append({
                    "id": s["id"], "condition": cond, "prompt_type": ptype,
                    "source": articles[s["id"]][CONDITIONS[cond]], "summary": s["summary"],
                    "label": "", "error_type": "", "notes": "",
                })

    with open(OUT_PATH, "w", encoding="utf-8-sig", newline="") as f:  # -sig: Excel shows Hindi correctly
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} summaries to {OUT_PATH}")


if __name__ == "__main__":
    main()
