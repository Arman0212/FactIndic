"""
FactIndic - Text normalization utilities
========================================
Every piece of text we save goes through normalize(), so that later steps
(number matching, entity matching, ROUGE) are not confused by invisible
differences such as two Unicode encodings of the same Devanagari letter, or
an en-dash vs. a normal hyphen.
"""

import re
import unicodedata

# All dash / hyphen variants: U+2010..U+2015 (hyphen, non-breaking hyphen,
# figure dash, en dash, em dash, horizontal bar) and U+2212 (minus sign).
_DASHES = re.compile("[\u2010-\u2015\u2212]")

# Curly / angled double quotes -> "   and   curly single quotes / primes -> '
_DOUBLE_QUOTES = re.compile("[\u201c\u201d\u201e\u201f\u00ab\u00bb\u2033]")
_SINGLE_QUOTES = re.compile("[\u2018\u2019\u201a\u201b\u2032]")

# Any run of whitespace (spaces, tabs, newlines, non-breaking spaces ...)
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Return a cleaned version of `text`:
    1. Unicode NFC (so the same character always has the same code points),
    2. all dash variants -> "-",
    3. all fancy quotes -> plain " or ',
    4. all whitespace runs -> one space, trimmed at both ends.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _DASHES.sub("-", text)
    text = _DOUBLE_QUOTES.sub('"', text)
    text = _SINGLE_QUOTES.sub("'", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text
