"""
FactIndic - Step 4: re-generate flagged pairs once, drop the ones still failing
==============================================================================
Reads the first-pass output (data/processed/factindic_300.jsonl), re-runs the
quality check, and for every flagged pair asks the LLM once more. Because we
use temperature 0, simply repeating the same prompt would give the same
output, so the second attempt includes the checker's complaint as feedback
(e.g. "missing numbers: 1995"). If the new version passes it replaces the old
one; if it still fails the pair is dropped.

The final dataset (with a `cmi` field per pair) is written to
data/processed/factindic.jsonl; dropped ids go to data/processed/dropped.csv.

Run:  python -m src.data.fix_flagged
"""

import csv
import time
from pathlib import Path

from tqdm import tqdm

from src.data.check_hinglish import check_all, check_pair, compute_cmi
from src.data.make_hinglish import OUT_PATH, read_jsonl, to_hinglish, write_jsonl
from src.llm_client import _is_rate_limit_error

FINAL_PATH = Path("data/processed/factindic.jsonl")
DROPPED_PATH = Path("data/processed/dropped.csv")


def to_hinglish_patient(text_hi: str, feedback: str) -> str:
    """Like to_hinglish(), but if the daily token quota is used up it waits
    (checking every 2 minutes) instead of crashing, so a retry is never lost."""
    while True:
        try:
            return to_hinglish(text_hi, feedback=feedback)
        except Exception as e:
            if not _is_rate_limit_error(e):
                raise
            tqdm.write("[quota] rate limit reached, waiting 2 minutes...")
            time.sleep(120)


def main():
    records = read_jsonl(OUT_PATH)
    flagged = check_all(records)
    print(f"{len(records)} pairs, {len(flagged)} flagged -> re-generating once")

    final, dropped, fixed = [], [], 0
    for rec in tqdm(records, desc="Fixing flagged"):
        if rec["id"] in flagged:
            feedback = "; ".join(flagged[rec["id"]])
            new_text = to_hinglish_patient(rec["text_hi"], feedback)
            problems = check_pair(rec["text_hi"], new_text)
            if problems:
                dropped.append((rec["id"], "; ".join(problems)))
                continue
            rec = {**rec, "text_hinglish": new_text}
            fixed += 1
        rec["cmi"] = compute_cmi(rec["text_hinglish"])
        final.append(rec)

    write_jsonl(FINAL_PATH, final)
    with open(DROPPED_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "reasons_after_retry"])
        writer.writerows(dropped)

    mean_cmi = sum(r["cmi"] for r in final) / len(final)
    print(f"Fixed on retry: {fixed}, dropped: {len(dropped)}")
    print(f"Final dataset: {len(final)} pairs -> {FINAL_PATH} (mean CMI {mean_cmi:.2f})")


if __name__ == "__main__":
    main()
