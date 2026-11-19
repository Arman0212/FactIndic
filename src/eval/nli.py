"""
FactIndic - Phase 4: NLI-based faithfulness (SummaC-style)
==========================================================
Natural Language Inference: given a PREMISE and a HYPOTHESIS, a model predicts
entailment / neutral / contradiction. A faithful summary sentence should be
ENTAILED by the source.

Model: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
(multilingual, trained on NLI data in 100+ languages incl. Hindi).

Method (like SummaC-ZS, Laban et al. 2022):
  1. Split the source into PREMISES (two variants, see below).
  2. Every summary sentence is scored against every premise; its score is the
     MAX entailment probability over premises (the fact may be anywhere).
  3. Summary score = mean over its sentences. Sentences with score < 0.5
     are counted as "unsupported".

Two premise variants are computed and saved:
  chunk  (primary for now): whole source sentences packed into chunks of at
         most CHUNK_TOKENS tokens (the source is longer than the model's 512
         token limit). Tends to UNDER-rate faithful paraphrases: NLI models are
         trained on short premises.
  window: every window of 2 consecutive source sentences. Tends to
         OVER-rate: in our tests it also gave high scores to real errors.
Which one agrees better with human labels is decided later by
src/eval/agreement.py on the manual annotation set.

Limitation: Romanized Hindi (hi_roman) is not in the model's training data
(Hindi there is Devanagari), so hi_roman NLI scores are less reliable.
Hinglish is also mostly out of distribution, though its English words help.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.make_manual_sample import split_sentences

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
CHUNK_TOKENS = 400       # 512 minus room for a summary sentence
BATCH_SIZE = 8           # small enough for an 8 GB Mac
UNSUPPORTED_BELOW = 0.5  # entailment probability threshold
WINDOW_SENTENCES = 2     # sentences per premise in the window variant


def sentence_windows(source: str, size: int = WINDOW_SENTENCES) -> list:
    """All windows of `size` consecutive sentences (the whole text if shorter)."""
    sents = split_sentences(source)
    if len(sents) <= size:
        return [" ".join(sents)] if sents else []
    return [" ".join(sents[i:i + size]) for i in range(len(sents) - size + 1)]


def pick_device() -> str:
    """Apple GPU (MPS) if available, else CPU."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


class NLIScorer:
    def __init__(self):
        self.device = pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.to(self.device).eval()
        # label order of this model: {0: entailment, 1: neutral, 2: contradiction}
        self.entail_id = self.model.config.label2id.get("entailment", 0)

    def chunk_source(self, source: str) -> list:
        """Pack whole source sentences into chunks of <= CHUNK_TOKENS tokens."""
        chunks, current, current_len = [], [], 0
        for sent in split_sentences(source):
            n = len(self.tokenizer.tokenize(sent))
            if current and current_len + n > CHUNK_TOKENS:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            current.append(sent)
            current_len += n
        if current:
            chunks.append(" ".join(current))
        return chunks

    @torch.no_grad()
    def entailment_probs(self, premises: list, hypotheses: list) -> list:
        """P(entailment) for each (premise, hypothesis) pair, in small batches."""
        probs = []
        for i in range(0, len(premises), BATCH_SIZE):
            enc = self.tokenizer(
                premises[i:i + BATCH_SIZE], hypotheses[i:i + BATCH_SIZE],
                return_tensors="pt", padding=True,
                truncation="only_first", max_length=512,  # never cut the hypothesis
            ).to(self.device)
            logits = self.model(**enc).logits.float()
            probs += logits.softmax(-1)[:, self.entail_id].cpu().tolist()
        return probs

    def sentence_scores(self, premises: list, sentences: list) -> list:
        """For each summary sentence: max P(entailment) over all premises."""
        pairs = [(p, s) for s in sentences for p in premises]
        probs = self.entailment_probs([p for p, _ in pairs], [h for _, h in pairs])
        n = len(premises)
        return [max(probs[i * n:(i + 1) * n]) for i in range(len(sentences))]

    def score(self, source: str, summary: str) -> dict:
        """NLI scores for one summary, for both premise variants.
        Keys: nli_n_sent, nli_{chunk,window}_{score,n_unsupported,pct_unsupported}."""
        sentences = split_sentences(summary)
        out = {"nli_n_sent": len(sentences)}
        for name, premises in [("chunk", self.chunk_source(source)),
                               ("window", sentence_windows(source))]:
            if not sentences or not premises:
                out.update({f"nli_{name}_score": float("nan"), f"nli_{name}_n_unsupported": 0,
                            f"nli_{name}_pct_unsupported": float("nan")})
                continue
            scores = self.sentence_scores(premises, sentences)
            n_unsupported = sum(x < UNSUPPORTED_BELOW for x in scores)
            out.update({
                f"nli_{name}_score": sum(scores) / len(scores),
                f"nli_{name}_n_unsupported": n_unsupported,
                f"nli_{name}_pct_unsupported": 100 * n_unsupported / len(scores),
            })
        return out
