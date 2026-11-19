"""
FactIndic - Step 3: automatic quality check of Hindi/Hinglish pairs
===================================================================
Two cheap, deterministic checks that catch the most common LLM rewrite errors:

1. Numbers: every number in the Hindi text must appear in the Hinglish text,
   and no new numbers may appear. Devanagari digits (०-९) are converted to
   ASCII (0-9) first, and thousands separators are dropped (1,200 == 1200).
   We compare *multisets* (Counter), so a number that appears twice in Hindi
   must also appear twice in Hinglish.
2. Length ratio: Hinglish word count / Hindi word count must be in 0.7-1.5.
   Much shorter suggests content was dropped or summarized; much longer
   suggests content was added.

We also measure *how* code-mixed each Hinglish text is with the Code-Mixing
Index (CMI, Das & Gambäck 2014). This is not a pass/fail check, it is saved
as a field for the analysis.

Flagged ids + reasons are saved to data/processed/flagged.csv.

Run:  python -m src.data.check_hinglish
"""

import csv
import re
from collections import Counter
from pathlib import Path

from wordfreq import zipf_frequency

from src.data.make_hinglish import OUT_PATH, read_jsonl

FLAGGED_PATH = Path("data/processed/flagged.csv")
MIN_RATIO, MAX_RATIO = 0.7, 1.5

# Map Devanagari digits ० १ २ ... ९ to 0 1 2 ... 9
DEVANAGARI_TO_ASCII = str.maketrans("०१२३४५६७८९", "0123456789")

# A number: digits, optionally with comma groups (1,20,000) and a decimal part (9.15)
NUMBER_RE = re.compile(r"\d+(?:,\d+)*(?:\.\d+)?")


def extract_numbers(text: str) -> Counter:
    """Return a Counter of all numbers in `text`, e.g. Counter({'2014': 2, '9.15': 1})."""
    text = text.translate(DEVANAGARI_TO_ASCII)
    return Counter(n.replace(",", "") for n in NUMBER_RE.findall(text))


# ---------------------------------------------------------------------------
# Code-Mixing Index (CMI)
# ---------------------------------------------------------------------------
# Das & Gambäck (2014):  CMI = 100 * (1 - max(w_i) / (n - u))
#   n   = number of tokens, u = language-independent tokens (numbers,
#         punctuation, named entities), w_i = tokens of language i.
# Since n - u is just the number of language-tagged tokens, for two languages:
#   CMI = 100 * (1 - max(n_en, n_hi) / (n_en + n_hi))
# 0 = monolingual; 50 = maximum mixing (half English, half Hindi).
#
# Word-level language tagging (simple, dictionary based):
#   1. token is capitalized in the middle of a sentence -> named entity (ignored)
#   2. token is in our romanized-Hindi list           -> Hindi
#   3. English Zipf frequency >= 3.0 (wordfreq)        -> English
#      (Zipf 3 = about once per million words, i.e. a reasonably common word)
#   4. otherwise                                       -> Hindi (e.g. "viman")
# Step 2 comes before step 3 because romanized Hindi is so common online that
# words like "hai", "ka", "se" get high English frequencies, and some Hindi
# words are also English words ("the" = were, "do" = two, "teen" = three).

HINDI_WORDS_PATH = Path("data/lexicons/romanized_hindi_words.txt")
HINDI_WORDS = {
    w
    for line in HINDI_WORDS_PATH.read_text(encoding="utf-8").splitlines()
    if not line.startswith("#")
    for w in line.split()
}
EN_ZIPF_THRESHOLD = 3.0

# Words (letters, optionally with an apostrophe) and sentence-ending punctuation.
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.!?]")


def tag_tokens(text: str) -> list:
    """Return a list of (word, tag) with tag in {'en', 'hi', 'ne'}."""
    tagged = []
    sentence_start = True
    for tok in TOKEN_RE.findall(text):
        if tok in ".!?":
            sentence_start = True
            continue
        low = tok.lower()
        if tok[0].isupper() and not sentence_start and low not in HINDI_WORDS:
            tag = "ne"
        elif low in HINDI_WORDS:
            tag = "hi"
        elif zipf_frequency(low, "en") >= EN_ZIPF_THRESHOLD:
            tag = "en"
        else:
            tag = "hi"
        tagged.append((tok, tag))
        sentence_start = False
    return tagged


def compute_cmi(text: str) -> float:
    """Code-Mixing Index of a Roman-script Hinglish text (0-50)."""
    tags = [t for _, t in tag_tokens(text)]
    n_en, n_hi = tags.count("en"), tags.count("hi")
    if n_en + n_hi == 0:
        return 0.0
    return round(100 * (1 - max(n_en, n_hi) / (n_en + n_hi)), 2)


def check_pair(text_hi: str, text_hinglish: str) -> list:
    """Return a list of problems (empty list = pair passed)."""
    problems = []

    nums_hi = extract_numbers(text_hi)
    nums_hing = extract_numbers(text_hinglish)
    missing = nums_hi - nums_hing   # in Hindi but not in Hinglish
    added = nums_hing - nums_hi     # in Hinglish but not in Hindi
    if missing:
        problems.append("missing numbers: " + ", ".join(sorted(missing.elements())))
    if added:
        problems.append("added numbers: " + ", ".join(sorted(added.elements())))

    ratio = len(text_hinglish.split()) / max(1, len(text_hi.split()))
    if not MIN_RATIO <= ratio <= MAX_RATIO:
        problems.append(f"length ratio {ratio:.2f} outside {MIN_RATIO}-{MAX_RATIO}")

    return problems


def check_all(records: list) -> dict:
    """Check every record. Returns {id: [problems]} for the flagged ones only."""
    flagged = {}
    for rec in records:
        problems = check_pair(rec["text_hi"], rec["text_hinglish"])
        if problems:
            flagged[rec["id"]] = problems
    return flagged


def print_summary(n_total: int, flagged: dict):
    """Print how many passed / were flagged, and a breakdown by reason."""
    reasons = Counter()
    for problems in flagged.values():
        for p in problems:
            reasons[p.split(":")[0].split(" ratio")[0]] += 1  # 'missing numbers', 'added numbers', 'length'
    print(f"Checked {n_total} pairs: {n_total - len(flagged)} passed, {len(flagged)} flagged")
    for reason, count in reasons.most_common():
        print(f"  {reason}: {count}")


def main():
    records = read_jsonl(OUT_PATH)
    flagged = check_all(records)

    FLAGGED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FLAGGED_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "reasons"])
        for id_, problems in flagged.items():
            writer.writerow([id_, "; ".join(problems)])

    print_summary(len(records), flagged)
    cmis = [compute_cmi(r["text_hinglish"]) for r in records]
    if cmis:
        print(f"Mean CMI: {sum(cmis) / len(cmis):.2f} (min {min(cmis)}, max {max(cmis)})")
    print(f"Flagged list saved to {FLAGGED_PATH}")


if __name__ == "__main__":
    main()
