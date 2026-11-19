"""
FactIndic - Phase 3 input: Romanized Hindi (no English mixing)
==============================================================
Adds the field `text_hi_roman` to data/processed/factindic.jsonl: the Hindi
article converted letter-by-letter to Roman script with the rule-based
`indic-transliteration` library (ITRANS scheme). No LLM is involved, so the
words are exactly the Hindi words - only the script changes.

This gives a third experimental condition that separates two effects:
  hi        (Devanagari Hindi)   -> hi_roman : effect of changing the SCRIPT
  hi_roman  (Roman Hindi)        -> hinglish : effect of CODE-MIXING

ITRANS output like "vimAna ke bAda meM" is made more readable:
  1. Hindi drops the silent final "a" of a word ending in a consonant
     (विमान = "vimaan", not "vimaana"), so we remove that final "a".
  2. Anusvara (M) and chandrabindu (.N) are written as "n" (में -> "men").
  3. Everything is lowercased (no capital letters for long vowels, so
     "A" and "a" both become "a"; this loses vowel length, like casual Hinglish).

Run:  python -m src.data.add_roman
"""

import re

from indic_transliteration import sanscript

from src.data.fix_flagged import FINAL_PATH
from src.data.make_hinglish import read_jsonl, write_jsonl
from src.text_utils import normalize

# A run of Devanagari letters/signs/digits (U+0900-U+097F), but NOT the danda
# punctuation marks । and ॥ (U+0964, U+0965), which we handle separately.
DEVANAGARI_WORD = re.compile(
    "[" + chr(0x0900) + "-" + chr(0x0963) + chr(0x0966) + "-" + chr(0x097F) + "]+"
)

# Consonants: क..ह (U+0915-U+0939) and the nukta letters क़..य़ (U+0958-U+095F).
CONSONANTS = set(chr(c) for c in range(0x0915, 0x093A)) | set(chr(c) for c in range(0x0958, 0x0960))
NUKTA = chr(0x093C)  # the dot below a letter, e.g. in ज़

# Fixes applied to the Devanagari *before* transliteration:
#  - consonant + separate nukta sign -> the single precomposed letter
#    (ज + nukta -> ज़), because ITRANS only knows the precomposed letters;
#  - the English-loan vowels ऑ/ॉ ("office") and ऍ/ॅ, which ITRANS does not
#    support, -> nearest Hindi vowels ओ/ो and ए/े;
#  - zero-width (non-)joiners and spaces, which are invisible but break words.
# (Code points are written as chr(0x...) so that no invisible character
#  appears in this source file.)
PRE_FIXES = {
    chr(0x0915) + NUKTA: chr(0x0958),  # क + nukta -> क़
    chr(0x0916) + NUKTA: chr(0x0959),  # ख + nukta -> ख़
    chr(0x0917) + NUKTA: chr(0x095A),  # ग + nukta -> ग़
    chr(0x091C) + NUKTA: chr(0x095B),  # ज + nukta -> ज़
    chr(0x0921) + NUKTA: chr(0x095C),  # ड + nukta -> ड़
    chr(0x0922) + NUKTA: chr(0x095D),  # ढ + nukta -> ढ़
    chr(0x092B) + NUKTA: chr(0x095E),  # फ + nukta -> फ़
    chr(0x092F) + NUKTA: chr(0x095F),  # य + nukta -> य़
    NUKTA: "",                 # any other (rare) nukta is simply dropped
    chr(0x0911): chr(0x0913),  # ऑ -> ओ
    chr(0x0949): chr(0x094B),  # ॉ (vowel sign) -> ो
    chr(0x090D): chr(0x090F),  # ऍ -> ए
    chr(0x0945): chr(0x0947),  # ॅ (vowel sign) -> े
    chr(0x0929): chr(0x0928),  # ऩ (rare) -> न
    chr(0x200B): "",           # zero-width space
    chr(0x200C): "",           # zero-width non-joiner
    chr(0x200D): "",           # zero-width joiner
}


def ends_in_bare_consonant(word: str) -> bool:
    """True if the Devanagari word ends with a consonant that has no vowel
    sign or virama after it, i.e. it ends in the silent inherent 'a'."""
    last = word[-1]
    return last in CONSONANTS


def romanize_word(word: str) -> str:
    """Transliterate one Devanagari word to readable lowercase Roman."""
    roman = sanscript.transliterate(word, sanscript.DEVANAGARI, sanscript.ITRANS)
    roman = roman.replace(".N", "n").replace("M", "n")   # chandrabindu, anusvara -> n
    roman = roman.replace(chr(0x095D), "rh")               # ढ़ (not mapped by ITRANS)
    roman = roman.replace("R^i", "ri").replace("RRi", "ri").replace(".h", "")  # ऋ, virama
    roman = roman.replace("K", "kh").replace("G", "gh")   # ख़, ग़ (ITRANS capitals)
    roman = roman.replace("^", "").replace("~", "").replace(".", "")  # leftover ITRANS marks
    roman = roman.lower()
    # Silent final schwa: "vimana" -> "viman" (but keep 1-letter words like "na")
    if ends_in_bare_consonant(word) and len(word) > 1 and roman.endswith("a"):
        roman = roman[:-1]
    return roman


def romanize(text: str) -> str:
    """Romanize all Devanagari words in `text`; digits, Latin letters and
    punctuation are left as they are. The danda । becomes a full stop."""
    for old, new in PRE_FIXES.items():
        text = text.replace(old, new)
    text = DEVANAGARI_WORD.sub(lambda m: romanize_word(m.group()), text)
    text = text.replace("।", ".").replace("॥", ".")
    return normalize(text)


def main():
    records = read_jsonl(FINAL_PATH)
    for rec in records:
        rec["text_hi_roman"] = romanize(rec["text_hi"])
    write_jsonl(FINAL_PATH, records)
    print(f"Added text_hi_roman to {len(records)} records in {FINAL_PATH}")


if __name__ == "__main__":
    main()
