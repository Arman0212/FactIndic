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
_DASHES = re.compile("[‐-―−]")

# Curly / angled double quotes -> "   and   curly single quotes / primes -> '
_DOUBLE_QUOTES = re.compile("[“”„‟«»″]")
_SINGLE_QUOTES = re.compile("[‘’‚‛′]")

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
