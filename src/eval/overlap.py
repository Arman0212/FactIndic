"""
FactIndic - Phase 4: lexical and semantic overlap (ROUGE, BERTScore)
====================================================================
ROUGE-1/2/L F1 between the summary and
  - the SOURCE article (all conditions): how much of the summary's wording
    comes from the source. F1 is low in absolute terms because the source is
    much longer than the summary; compare conditions, not absolute values.
  - the REFERENCE summary (ref_summary_hi): only for the hi condition, because
    the reference is Devanagari Hindi; comparing it with Roman-script
    summaries would give ~0 for reasons that have nothing to do with quality.

Tokenizer: the default rouge_score tokenizer keeps only a-z0-9 and would delete
all Devanagari text. Python's \\w also splits Devanagari words at vowel signs
(matras). So we split on whitespace and only strip punctuation from the ends.

BERTScore (bert-base-multilingual-cased) between summary and source: cosine
similarity of contextual token embeddings, so it rewards paraphrases.
Limitation: mBERT reads at most 512 tokens, so for long articles only the
beginning of the source is used.
"""

import string

from rouge_score import rouge_scorer

from src.eval.nli import pick_device

PUNCT = string.punctuation + "।॥“”‘’"  # + danda, curly quotes
# bert-base-multilingual-cased, under its current Hugging Face id (the old id
# without "google-bert/" no longer downloads). Layer 9 is bert_score's own
# default for this model; it has to be given explicitly for the new id.
BERTSCORE_MODEL = "google-bert/bert-base-multilingual-cased"
BERTSCORE_LAYER = 9


class WhitespaceTokenizer:
    """Lowercase, split on whitespace, strip punctuation at word ends."""

    def tokenize(self, text: str) -> list:
        tokens = (t.strip(PUNCT) for t in text.lower().split())
        return [t for t in tokens if t]


_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=WhitespaceTokenizer())


def rouge_f1(reference: str, summary: str, suffix: str) -> dict:
    """ROUGE-1/2/L F1, keys like rouge1_src."""
    scores = _ROUGE.score(reference, summary)  # (target, prediction)
    return {f"{k}_{suffix}": v.fmeasure for k, v in scores.items()}


class BertScorer:
    """Thin wrapper around bert_score that loads the model once."""

    def __init__(self):
        from bert_score import BERTScorer
        self.scorer = BERTScorer(model_type=BERTSCORE_MODEL, num_layers=BERTSCORE_LAYER,
                                 device=pick_device(), batch_size=8)

    def f1(self, summaries: list, sources: list) -> list:
        _, _, f = self.scorer.score(summaries, sources)
        return f.tolist()
