"""
FactIndic - highlighting for the labelling pages
================================================
Splits a text into segments so the page can colour names (blue) and numbers
(amber). Used by the local labelling server (src/eval/label_server.py) and by
the offline teammate pages (src/eval/make_annotation_packs.py).

  names   - Stanza gold entities of the article (Devanagari and their
            romanized form) and capitalized word runs in Roman text;
  numbers - digits and number words (same lists as src/eval/numbers.py).

In the SUMMARY, a name or digit-number that the automatic matcher cannot find
in the source gets the extra kind "bad" (red dashed outline). It is only a
reading aid: the check can be wrong.
"""

import json
import re
from pathlib import Path

from src.data.add_roman import romanize
from src.eval.entities import is_supported, roman_candidates
from src.eval.numbers import MULTIPLIERS, UNITS, extract_numbers

GOLD_PATH = Path("results/cache/gold_entities.json")  # written by src/evaluate.py
NUMBER_WORDS = set(UNITS) | set(MULTIPLIERS)


def is_word_char(c: str) -> bool:
    return c.isalnum() or 0x0900 <= ord(c) <= 0x097F


def find_phrase(text: str, phrase: str):
    """Yield (start, end) of whole-word, case-insensitive matches of phrase."""
    low, p = text.lower(), phrase.lower()
    start = low.find(p)
    while start != -1 and p:
        end = start + len(p)
        before_ok = start == 0 or not is_word_char(text[start - 1])
        after_ok = end == len(text) or not is_word_char(text[end])
        if before_ok and after_ok:
            yield start, end
        start = low.find(p, start + 1)


def segments(text: str, names: list, source: str = None) -> list:
    """Split text into [{"t": text, "k": kind}] with kind "", "name" or "num",
    plus " bad" when (summary only) the item is not found in the source."""
    marks = [""] * len(text)
    src_values = extract_numbers(source) if source is not None else None

    # names: longest phrases first so "Pakistan International Airlines" wins
    for phrase in sorted(set(names), key=len, reverse=True):
        for s, e in find_phrase(text, phrase):
            if any(marks[s:e]):
                continue
            kind = "name"
            if source is not None and not is_supported(phrase, source):
                kind += " bad"
            marks[s:e] = [kind] * (e - s)

    # numbers: tokens with digits, or number words (after romanizing)
    for m in re.finditer(r"\S+", text):
        core = m.group().strip(".,;:!?\"'()[]-।")
        if not core or any(marks[m.start():m.end()]):
            continue
        has_digit = any(ch.isdigit() for ch in core)
        if not has_digit and romanize(core).lower() not in NUMBER_WORDS:
            continue
        s = m.start() + m.group().find(core)
        kind = "num"
        if source is not None and has_digit:
            values = extract_numbers(core)
            if values and not values <= src_values:
                kind += " bad"
        marks[s:s + len(core)] = [kind] * len(core)

    out = []
    for ch, k in zip(text, marks):
        if out and out[-1]["k"] == k:
            out[-1]["t"] += ch
        else:
            out.append({"t": ch, "k": k})
    return out


def entity_phrases(gold: list, text: str) -> list:
    """Names to highlight in one text."""
    phrases = list(gold) + [romanize(g).lower() for g in gold]
    if re.search(r"[A-Za-z]", text):
        phrases += roman_candidates(text)
    return phrases


def load_gold() -> dict:
    """Article id -> list of gold entities (empty if evaluate.py has not run)."""
    return json.loads(GOLD_PATH.read_text(encoding="utf-8")) if GOLD_PATH.exists() else {}


def highlight(source: str, summary: str, gold: list) -> tuple:
    """(source segments, summary segments) for one row."""
    return (segments(source, entity_phrases(gold, source)),
            segments(summary, entity_phrases(gold, summary), source=source))
