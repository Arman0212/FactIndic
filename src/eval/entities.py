"""
FactIndic - Phase 4: named-entity precision / recall / F1 across scripts
========================================================================
The same entity can be written three ways in our data:
    Devanagari   स्टैनस्टेड      (hi)
    ITRANS-Roman stainasted       (hi_roman, made by add_roman.py)
    English      Stansted         (hinglish)
Exact string matching would fail, so both sides are reduced to a rough
PHONETIC SKELETON and compared with fuzzy matching (rapidfuzz).

Skeleton (per word):
  1. romanize Devanagari with the same ITRANS rules as add_roman.py, lowercase;
  2. merge similar sounds: English "-tion" -> "shan", soft c (ce/ci/cy) -> s,
     ph->f, sh->s, aspirated consonants lose their h (kh->k, bh->b ...),
     w->v, z->j, q->k, c->k, x->ks;
  3. keep the first letter (a leading vowel is always written "a"), drop all
     later vowels (a e i o u, and y, which acts as a vowel inside words);
  4. collapse doubled letters.
  "Stansted" -> "stnstd", स्टैनस्टेड -> "stainasted" -> "stnstd";
  "Manchester" and मैनचेस्टर -> "mnkstr".

Gold entities: Stanza Hindi NER on the Devanagari source (text_hi), keeping
name-like types only (person, location, organization, abbreviation, brand,
titles). Numbers and dates are left to numbers.py.

Recall    = gold entities whose skeleton fuzzy-matches the summary skeleton,
            counted only over gold entities that can be found in THIS
            condition's source text. (The Hinglish rewrite sometimes
            translates names, e.g. रक्षा मंत्रालय -> "Defence Ministry"; a summary
            cannot be blamed for not containing a spelling its source lacks.)
Precision = summary entities supported by the source text:
  - Devanagari summaries: entities found by Stanza in the summary;
  - Roman summaries: runs of Capitalized words (not a common word at the very
    start of a sentence), plus gold entities found in the summary;
  an entity is unsupported if its skeleton does not fuzzy-match anywhere in
  the source text of the same condition. A multi-word entity also counts as
  supported if EACH of its words is found in the source, because Stanza often
  merges neighbouring names into one span ("डेनियल क्रेग जेम्स बॉन्ड").
Match = rapidfuzz.partial_ratio(entity_skeleton, text_skeleton) >= THRESHOLD
(threshold tuned on 20 hand-checked pairs, results/entity_threshold_tuning.csv).

Limitations:
- Stanza misses some entities and sometimes merges words into one entity
  ("पीके709 लाहौर हवाई अड्डे"); errors in the gold set carry into recall.
- Roman-script summaries have no NER: lowercase Romanized Hindi summaries
  (hi_roman) only get candidates from capitalization and from gold entities,
  so a hallucinated lowercase name in hi_roman is NOT detected -> hi_roman
  entity precision is optimistic.
- Skeletons are rough: different names with the same consonants (e.g. "Ram" /
  "Rome") can match, and very different spellings of one name can fail.
- Because of the word-by-word rule above, a mix-up of two real names
  (e.g. first name of one person + surname of another) is not detected.
- Very short skeletons (< 3 letters, e.g. acronyms like "PIA") are compared
  on the full romanized form with vowels instead, to avoid matching everything.
"""

import math
import re
from functools import lru_cache

from rapidfuzz import fuzz
from wordfreq import zipf_frequency

from src.data.add_roman import romanize
from src.data.check_hinglish import HINDI_WORDS

THRESHOLD = 85  # see tune_threshold() / README

# Stanza Hindi NER tags (IJCNLP-08 tag set) that are names
NAME_TYPES = {"NEP", "NEL", "NEO", "NEA", "NEB", "NETP", "NETO"}

DEVANAGARI_RE = re.compile("[" + chr(0x0900) + "-" + chr(0x097F) + "]")


# ---------------------------------------------------------------------------
# Phonetic skeleton
# ---------------------------------------------------------------------------
def word_skeleton(word: str) -> str:
    """Skeleton of one lowercase Roman word, e.g. "manchester" -> "mnkstr"."""
    w = word.replace("tion", "shan")
    w = re.sub(r"c(?=[eiy])", "s", w)               # soft c: police, defence
    w = w.replace("ph", "f").replace("sh", "s")
    w = re.sub(r"([bcdgjklmnprtvz])h", r"\1", w)  # drop aspiration: kh->k, ch->c
    w = w.replace("w", "v").replace("z", "j").replace("q", "k")
    w = w.replace("c", "k").replace("x", "ks")
    first = "a" if w[0] in "aeiouy" else w[0]        # any leading vowel -> "a"
    w = first + re.sub(r"[aeiouy]", "", w[1:])       # drop later vowels (and y)
    return re.sub(r"(.)\1+", r"\1", w)               # "ss" -> "s"


def to_roman(text: str) -> str:
    """Romanize any Devanagari (ITRANS, as in add_roman.py) and lowercase."""
    return romanize(text).lower()


@lru_cache(maxsize=100_000)
def skeleton(text: str) -> str:
    """Skeleton of a whole text: word skeletons joined by spaces."""
    words = re.findall(r"[a-z0-9]+", to_roman(text))
    return " ".join(word_skeleton(w) for w in words)


def match_score(entity: str, text: str) -> float:
    """How well `entity` occurs somewhere in `text` (0-100)."""
    ent_skel = skeleton(entity)
    if len(ent_skel.replace(" ", "")) < 3:
        # too short to be informative as a skeleton -> compare romanized forms
        ent = " ".join(re.findall(r"[a-z0-9]+", to_roman(entity)))
        return fuzz.partial_ratio(ent, " ".join(re.findall(r"[a-z0-9]+", to_roman(text))))
    return fuzz.partial_ratio(ent_skel, skeleton(text))


def is_present(entity: str, text: str, threshold: float = THRESHOLD) -> bool:
    return match_score(entity, text) >= threshold


def is_supported(entity: str, source: str) -> bool:
    """Summary entity supported by the source: the whole entity matches, or
    (for multi-word entities) every word matches somewhere in the source."""
    if is_present(entity, source):
        return True
    words = entity.split()
    return len(words) > 1 and all(is_present(w, source) for w in words)


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------
_NLP = None


def stanza_pipeline():
    """Load the Stanza Hindi NER pipeline once (CPU)."""
    global _NLP
    if _NLP is None:
        import stanza
        stanza.download("hi", processors="tokenize,ner", verbose=False)
        _NLP = stanza.Pipeline("hi", processors="tokenize,ner", verbose=False, use_gpu=False)
    return _NLP


def dedupe(entities: list) -> list:
    """Remove duplicates (same skeleton), keeping the first spelling."""
    seen, out = set(), []
    for e in entities:
        key = skeleton(e)
        if key and key not in seen:
            seen.add(key)
            out.append(e)
    return out


def stanza_entities(text: str) -> list:
    """Name-like entities found by Stanza in a Devanagari text."""
    doc = stanza_pipeline()(text)
    return dedupe([e.text for e in doc.ents if e.type in NAME_TYPES])


def roman_candidates(summary: str) -> list:
    """Entity candidates in a Roman-script summary: runs of Capitalized words.
    A single capitalized word at the start of a sentence is skipped if it is
    a common word ("Company ne ...", "Police ..."), since capitalization there
    says nothing."""
    candidates = []
    for sentence in re.split(r"(?<=[.!?])\s+", summary):
        # words start with a letter and may contain digits ("PK709", "A-380")
        words = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", sentence)
        run, run_start = [], 0
        for i, w in enumerate(words + [""]):           # "" flushes the last run
            if w and w[0].isupper():
                if not run:
                    run_start = i
                run.append(w)
                continue
            if run:
                low = run[0].lower()
                common = low in HINDI_WORDS or zipf_frequency(low, "en") >= 3.0
                if not (len(run) == 1 and run_start == 0 and common):
                    candidates.append(" ".join(run))
                run = []
    return dedupe(candidates)


# ---------------------------------------------------------------------------
# Scores for one summary
# ---------------------------------------------------------------------------
def entity_scores(gold: list, source: str, summary: str) -> dict:
    """Entity P/R/F1 for one summary.

    gold    : gold entities of the article (Stanza on text_hi, Devanagari)
    source  : source text of this condition (text_hi / text_hi_roman / text_hinglish)
    summary : the summary
    """
    nan = float("nan")

    # Recall: which gold entities appear in the summary? Only gold entities
    # that are recognizable in this condition's source can be expected.
    gold = [e for e in gold if is_present(e, source)]
    found = [e for e in gold if is_present(e, summary)]
    recall = len(found) / len(gold) if gold else nan

    # Precision: are the summary's entities supported by the source?
    is_devanagari = len(DEVANAGARI_RE.findall(summary)) > len(re.findall(r"[A-Za-z]", summary))
    if is_devanagari:
        candidates = stanza_entities(summary)
        supported = [c for c in candidates if is_supported(c, source)]
    else:
        caps = roman_candidates(summary)
        supported = [c for c in caps if is_supported(c, source)]
        # gold entities found in the summary that no capitalized run covers
        extra = [e for e in found if not any(match_score(e, c) >= THRESHOLD for c in caps)]
        candidates = caps + extra
        supported = supported + extra
    unsupported = [c for c in candidates if c not in supported]
    precision = len(supported) / len(candidates) if candidates else nan

    if math.isnan(precision) or math.isnan(recall):
        f1 = nan
    else:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "ent_p": precision, "ent_r": recall, "ent_f1": f1,
        "ent_n_gold": len(gold), "ent_n_summary": len(candidates),
        "ent_unsupported": " | ".join(unsupported),
    }


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------
def tune_threshold(path: str = "results/entity_threshold_tuning.csv"):
    """Print the accuracy of each threshold on the hand-labelled pairs.

    The CSV has 20 (gold entity, summary) pairs sampled across the score range
    (4 pairs per score bin 55-70, 70-80, 80-88, 88-95, 95-100) and labelled by
    reading the summary: present_by_hand = 1 if the entity is mentioned.
    Run:  python -c "from src.eval.entities import tune_threshold; tune_threshold()"
    """
    import csv
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    scored = [(match_score(r["entity"], r["summary"]), int(r["present_by_hand"])) for r in rows]
    for s, label in sorted(scored):
        print(f"  score {s:5.1f}  hand label {label}")
    for t in range(60, 101, 5):
        correct = sum((s >= t) == bool(label) for s, label in scored)
        print(f"threshold {t:3d}: {correct}/{len(scored)} correct")
