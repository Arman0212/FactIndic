"""
FactIndic - Step 5: sample 30 pairs for manual checking
=======================================================
Writes data/processed/sample_for_manual_check.md: 30 random pairs (seed 42)
from the final dataset, each shown as a table with the Hindi and Hinglish
sentences side by side. Because the prompt requires the same number of
sentences in the same order, row i of Hindi should match row i of Hinglish,
which makes missing or changed facts easy to spot.

Run:  python -m src.data.make_manual_sample
"""

import random
import re
from pathlib import Path

from src.data.fix_flagged import FINAL_PATH
from src.data.make_hinglish import read_jsonl

SAMPLE_PATH = Path("data/processed/sample_for_manual_check.md")
N, SEED = 30, 42

# Split after . ? ! or the Devanagari danda (।) when followed by a space.
# ("9.15" is not split because there is no space after the dot.)
SENTENCE_RE = re.compile(r"(?<=[.?!।])\s+")


def split_sentences(text: str) -> list:
    return [s for s in SENTENCE_RE.split(text) if s.strip()]


def cell(text: str) -> str:
    """Escape characters that would break a markdown table cell."""
    return text.replace("|", "\\|")


def main():
    records = read_jsonl(FINAL_PATH)
    random.seed(SEED)
    sample = random.sample(records, min(N, len(records)))

    lines = [
        "# Manual check sample",
        "",
        f"{len(sample)} random pairs (seed {SEED}) from `{FINAL_PATH}`, split into sentences.",
        "For each pair, check that every name, number, date and fact in the Hindi",
        "column appears unchanged in the Hinglish column, and nothing is added.",
        "",
    ]
    for i, rec in enumerate(sample, 1):
        hi, hing = split_sentences(rec["text_hi"]), split_sentences(rec["text_hinglish"])
        lines += [
            f"## {i}. {rec['title']}",
            "",
            f"id: `{rec['id']}` | CMI: {rec['cmi']} | sentences: {len(hi)} Hindi / {len(hing)} Hinglish",
            "",
            "| # | Hindi | Hinglish |",
            "|---|---|---|",
        ]
        for j in range(max(len(hi), len(hing))):
            a = hi[j] if j < len(hi) else ""
            b = hing[j] if j < len(hing) else ""
            lines.append(f"| {j + 1} | {cell(a)} | {cell(b)} |")
        lines.append("")

    SAMPLE_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(sample)} pairs to {SAMPLE_PATH}")


if __name__ == "__main__":
    main()
