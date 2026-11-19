"""
FactIndic - language / script check for summaries
=================================================
Small models sometimes answer in the wrong language or script (e.g. an English
summary of a Hinglish article). We do NOT regenerate those: the wrong-language
rate per condition and prompt is itself a finding. But such summaries are
excluded from the factuality metrics, because comparing e.g. an English
summary with a Roman-script source would measure translation, not factuality.

Two simple measures per summary:
  deva_share = Devanagari letters / (Devanagari + Latin letters)
  en_ratio   = share of words tagged English by the CMI word tagger
               (src/data/check_hinglish.tag_tokens; names are ignored)

Rules (script_ok = True if all rules for the condition hold):
  hi        : deva_share >= 0.8   (mostly Devanagari; a few Latin acronyms are fine)
  hi_roman  : deva_share <= 0.2   and en_ratio <= 0.6
  hinglish  : deva_share <= 0.2   and en_ratio <= 0.6
The 0.6 threshold was calibrated on our data: every Hinglish and Romanized
Hindi source article has en_ratio <= 0.33, while pure-English summaries score
0.81-0.86.

Run on an existing results file (adds/updates the fields and prints the
wrong-language rate per condition and prompt):
    python -m src.script_check results/summaries_gemma3_4b.jsonl
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

from src.data.check_hinglish import tag_tokens
from src.data.make_hinglish import read_jsonl, write_jsonl

DEVANAGARI_RE = re.compile("[" + chr(0x0900) + "-" + chr(0x097F) + "]")  # Devanagari block U+0900-U+097F
LATIN_RE = re.compile(r"[A-Za-z]")

MIN_DEVA_HI = 0.8      # hi summaries must be at least 80% Devanagari letters
MAX_DEVA_ROMAN = 0.2   # Roman-script summaries may be at most 20% Devanagari
MAX_EN_RATIO = 0.6     # above this the summary is (almost) pure English


def check_script(summary: str, condition: str) -> dict:
    """Return {"script_ok": bool, "deva_share": float, "en_ratio": float}."""
    n_deva = len(DEVANAGARI_RE.findall(summary))
    n_latin = len(LATIN_RE.findall(summary))
    deva_share = n_deva / (n_deva + n_latin) if n_deva + n_latin else 0.0

    tags = [t for _, t in tag_tokens(summary) if t != "ne"]  # ignore names
    en_ratio = tags.count("en") / len(tags) if tags else 0.0

    if not summary.strip():
        ok = False
    elif condition == "hi":
        ok = deva_share >= MIN_DEVA_HI
    else:  # hi_roman and hinglish: Roman script, but not plain English
        ok = deva_share <= MAX_DEVA_ROMAN and en_ratio <= MAX_EN_RATIO

    return {"script_ok": ok, "deva_share": round(deva_share, 3), "en_ratio": round(en_ratio, 3)}


def report(records: list):
    """Print the wrong-language rate for every (condition, prompt_type)."""
    groups = defaultdict(list)
    for r in records:
        groups[(r["condition"], r["prompt_type"])].append(r["script_ok"])
    print(f"{'condition':<10} {'prompt':<8} {'n':>4} {'wrong':>6} {'rate':>7}")
    for (cond, ptype), oks in sorted(groups.items()):
        wrong = oks.count(False)
        print(f"{cond:<10} {ptype:<8} {len(oks):>4} {wrong:>6} {100 * wrong / len(oks):>6.1f}%")
    total_wrong = sum(not r["script_ok"] for r in records)
    print(f"Total: {total_wrong}/{len(records)} wrong language/script "
          f"({100 * total_wrong / max(1, len(records)):.1f}%)")


def main():
    path = Path(sys.argv[1])
    records = read_jsonl(path)
    for r in records:
        r.update(check_script(r["summary"], r["condition"]))
    write_jsonl(path, records)
    report(records)


if __name__ == "__main__":
    main()
