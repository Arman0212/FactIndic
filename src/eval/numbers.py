"""
FactIndic - Phase 4: number precision / recall / F1
===================================================
Numbers are the easiest facts to check automatically, and a changed number is
almost always a real factual error.

1. Extract every number from the source and from the summary:
   - digits: "1,200" -> 1200, "9.15" -> 9.15, Devanagari digits "२०१४" -> 2014;
   - number words: "teen hazaar" / "तीन हज़ार" -> 3000, "do lakh" -> 200000,
     "derh crore" -> 15000000. Devanagari text is first romanized with the same
     transliteration as add_roman.py, so ONE dictionary of Roman number words
     covers Hindi, Romanized Hindi and Hinglish.
2. Compare the SETS of values:
   precision = |summary values that are also in the source| / |summary values|
   recall    = |source values that are also in the summary| / |source values|
   F1        = harmonic mean of precision and recall.
   precision < 1  -> the summary contains a number that is not in the source
                     (hallucinated or altered number).
   If the summary has no numbers, precision is undefined (NaN); if the source
   has none, recall is undefined. NaN values are skipped when averaging.

Limitations (simple on purpose):
- A lone "ek"/"एक" (value 1) is ignored, because it is mostly the article "a"
  ("ek mahila" = "a woman"); "ek lakh" (100000) is still counted.
- Times like "9:15" give two numbers (9 and 15); "9.15" gives one (9.15).
- Values are compared exactly, so "1.2 lakh" vs "1,20,000" match (both 120000)
  but rounding ("lagbhag 1200" vs "1,187") counts as an error.
- Hindi "do" can also mean "give" (imperative); this is rare in news summaries.
- Some spellings are left out on purpose because they collide with common
  words: साठ (60) romanizes to "sath" like साथ ("with"), and "chah" is also चाह
  ("wish"). The surname "Das" is read as 10 (das) - a known false positive.
"""

import math
import re

from src.data.add_roman import romanize

# Plain number words (Roman spellings as written by people AND as produced by
# our romanizer, e.g. तीन -> "tin", सात -> "sat"), plus English words in case a
# Hinglish summary uses them.
UNITS = {
    "ek": 1, "do": 2, "teen": 3, "tin": 3, "char": 4, "chaar": 4,
    "panch": 5, "paanch": 5, "chhah": 6, "chhe": 6,
    "saat": 7, "sat": 7, "aath": 8, "ath": 8, "nau": 9, "das": 10,
    "gyarah": 11, "barah": 12, "terah": 13, "chaudah": 14, "pandrah": 15,
    "solah": 16, "satrah": 17, "atharah": 18, "unnis": 19, "bees": 20, "bis": 20,
    "tees": 30, "tis": 30, "chalis": 40, "chaalis": 40, "pachas": 50, "pachaas": 50,
    "sattar": 70, "assi": 80, "nabbe": 90,
    "derh": 1.5, "dedh": 1.5, "dhai": 2.5, "dhaai": 2.5,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "twenty": 20, "fifty": 50,
    "tventi": 20,  # ट्वेंटी as romanized ("ट्वेंटी ट्वेंटी" = T20)
}
# Multipliers: "teen hazaar" = 3 x 1000
MULTIPLIERS = {
    "sau": 100, "hundred": 100,
    "hazaar": 1000, "hazar": 1000, "hajar": 1000, "thousand": 1000,
    "lakh": 100_000, "lac": 100_000,
    "crore": 10_000_000, "karod": 10_000_000, "karor": 10_000_000,
    "million": 1_000_000, "billion": 1_000_000_000,
}

# A token is either a digit number (with optional 1,20,000-style commas and a
# decimal part) or a word made of letters.
TOKEN_RE = re.compile(r"\d+(?:,\d+)*(?:\.\d+)?|[a-z]+")


def _clean(value: float) -> float:
    """Round away float noise (1.5 * 100000 = 150000.0)."""
    return round(value, 4)


def extract_numbers(text: str) -> set:
    """Return the set of numeric values mentioned in `text`."""
    # Devanagari -> Roman (this also turns Devanagari digits into 0-9), lowercase
    text = romanize(text).lower()

    values = set()
    total, current, in_number = 0.0, 0.0, False
    only_word_ek = False  # True while the current number is just the word "ek"

    def finish():
        """Close the number being built and store it."""
        nonlocal total, current, in_number, only_word_ek
        if in_number and not only_word_ek:
            values.add(_clean(total + current))
        total, current, in_number, only_word_ek = 0.0, 0.0, False, False

    for tok in TOKEN_RE.findall(text):
        if tok[0].isdigit():
            if in_number and current:   # two digit numbers in a row: separate numbers
                finish()
            current += float(tok.replace(",", ""))
            in_number, only_word_ek = True, False
        elif tok in UNITS:
            if in_number and current:   # "do teen" (two three) -> separate numbers
                finish()
            only_word_ek = (tok == "ek") and not in_number
            current += UNITS[tok]
            in_number = True
        elif tok in MULTIPLIERS and in_number:
            mult = MULTIPLIERS[tok]
            current = (current or 1) * mult
            if mult >= 1000:            # "nau hazaar char sau" = 9000 + 400
                total += current
                current = 0.0
            only_word_ek = False
        else:
            finish()
    finish()
    return values


def prf(summary_values: set, source_values: set) -> dict:
    """Precision, recall and F1 between two sets of values (NaN if undefined)."""
    nan = float("nan")
    p = len(summary_values & source_values) / len(summary_values) if summary_values else nan
    r = len(source_values & summary_values) / len(source_values) if source_values else nan
    if math.isnan(p) or math.isnan(r):
        f = nan
    else:
        f = 2 * p * r / (p + r) if p + r else 0.0
    return {"p": p, "r": r, "f1": f}


def number_scores(source: str, summary: str) -> dict:
    """Number P/R/F1 for one summary, plus the unsupported values (for error analysis)."""
    src, summ = extract_numbers(source), extract_numbers(summary)
    s = prf(summ, src)
    return {
        "num_p": s["p"], "num_r": s["r"], "num_f1": s["f1"],
        "num_unsupported": " ".join(str(v).rstrip("0").rstrip(".") for v in sorted(summ - src)),
    }
