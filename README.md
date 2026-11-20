# FactIndic 🇮🇳
**Factuality-Aware Summarization for Hindi and Hindi-English Code-Mixed (Hinglish) Text**

*CSET 346: Natural Language Processing — Course Project*

---

## 📌 Project Overview

**FactIndic** investigates the factual consistency of abstractive text summarization on **Hindi** and **Hinglish (Hindi-English code-mixed)** text. 

A critical challenge with modern Large Language Models (LLMs) is **hallucination**—inventing facts, misattributing numerical figures, or swapping named entities. In low-resource or code-mixed settings like Hinglish, these errors are even more prevalent due to tokenization quirks and cross-lingual drift.

**Key Design Principle:** We do **NOT** train models. Instead, we:
1. Query foundation models (via Google Gemini or Groq APIs) using baseline (plain) prompts vs. factuality-aware prompts.
2. Rigorously evaluate factual integrity using entity/number precision-recall-F1, Natural Language Inference (NLI) cross-lingual models (`mDeBERTa`), ROUGE, and BERTScore.

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 (Python 3.11 recommended for PyTorch/Transformers compatibility on Apple Silicon / Linux).
- An active API key from either:
  - [Google AI Studio](https://aistudio.google.com/) (for Gemini), or
  - [Groq Console](https://console.groq.com/) (for LLaMA 3.3 / Groq models).

### 2. Set Up Virtual Environment

Create and activate an isolated Python virtual environment:

```bash
# Navigate to the project root
cd /path/to/FactIndic

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS / Linux:
source venv/bin/activate

# On Windows:
# .\venv\Scripts\activate
```

### 3. Install Dependencies

Install required libraries (NLP packages, API clients, evaluation tools):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables (`.env`)

Copy the provided `.env.example` template to `.env`:

```bash
cp .env.example .env
```

Open `.env` in your editor and configure your provider, model, and API keys:

```env
# Provider choice: "gemini" or "groq"
LLM_PROVIDER=gemini

# Model name
LLM_MODEL=gemini-2.5-flash

# API Credentials (never commit this file to git!)
GEMINI_API_KEY=AIzaSy...
GROQ_API_KEY=gsk_...
```

> **Security Note:** The `.env` file is explicitly ignored in `.gitignore`. Never share or commit your API keys.

### 5. Run the Smoke Test

Verify that your environment, API keys, and model generation work for both Hindi and Hinglish:

```bash
python -m src.smoke_test
```

Expected output:
- Connects to your configured LLM provider.
- Summarizes a sample Hindi news paragraph in one sentence.
- Summarizes a sample Hinglish cricket paragraph in one sentence.
- Displays generation latency and prints confirmation.

---

## 📊 Dataset

**Final file:** `data/processed/factindic.jsonl`: **186 Hindi-Hinglish pairs**. Fields: `id`, `title`, `text_hi`, `text_hinglish`, `ref_summary_hi`, `cmi`, `text_hi_roman` (added in Phase 3, see below).

1. **Source.** The Hindi test split of [XL-Sum](https://huggingface.co/datasets/csebuetnlp/xlsum) (BBC Hindi articles with one-sentence reference summaries; 8,847 articles). It is loaded from the Hugging Face parquet branch because newer `datasets` versions no longer run the XL-Sum loading script.
2. **Filtering.** We kept articles of 150-500 words (5,610 articles) and sampled 300 with seed 42 (`data/raw/hindi_300.jsonl`). All text is cleaned with `src/text_utils.normalize()`: Unicode NFC, all dash variants become `-`, quotes are unified, and whitespace is collapsed.
3. **Hinglish generation.** Each article was rewritten into Roman-script Hinglish by `openai/gpt-oss-120b` (Groq) at temperature 0, using the prompt `prompts/to_hinglish.txt`. The prompt does three things:
   - It asks for Hindi grammar, with the everyday nouns, verbs and adjectives that urban speakers say in English (4 few-shot examples).
   - It requires every name, number, date and unit to stay the same, with number words kept as words and digits kept as digits.
   - It forbids adding, removing or summarizing anything, and requires the same sentences in the same order.
4. **Size.** Groq's free tier allows 200k tokens per day (about 50 articles), so generation stopped after the first 189 of the 300. The sample order is random, so these 189 are still a random sample.
5. **Automatic quality check** (`src/data/check_hinglish.py`):
   - **Numbers:** all numbers are extracted from both versions (Devanagari digits converted to ASCII, thousands separators dropped) and compared as multisets. The check flags any number that is missing or added.
   - **Length:** the ratio of Hinglish words to Hindi words must be between 0.7 and 1.5.
   - **Results:** 180 of 189 pairs passed. All 9 flagged pairs had added numbers, where the model turned a number word into digits (for example "थ्री डी" became "3D").
   - **Retry:** each flagged pair was re-generated once, with the checker's complaint added to the prompt. 6 were fixed and 3 were dropped (`data/processed/dropped.csv`).
6. **Code-mixing.** We measure the Code-Mixing Index (CMI, Das & Gambäck 2014): CMI = 100 × (1 − max(n_en, n_hi) / (n_en + n_hi)). It runs from 0 (one language only) to 50 (half and half).
   - Each word is tagged as follows:
     - A capitalized word in the middle of a sentence is treated as a named entity and ignored.
     - A word on a list of romanized Hindi words (`data/lexicons/romanized_hindi_words.txt`) is Hindi.
     - A word with an English Zipf frequency of at least 3 in `wordfreq` is English.
     - Anything else is Hindi.
   - **Mean CMI is 17.55** (median 17.04). On a 5-article pilot, the few-shot prompt raised mean CMI from 6.2 to 13.4 compared with a first prompt that had no few-shot examples.
   - On average the Hinglish version has 1.005× as many words as the Hindi, for a mean of 276 Hindi words per article.
7. **Manual check.** `data/processed/sample_for_manual_check.md` shows 30 random pairs, matched sentence by sentence. In 18 of the 186 pairs the naive sentence splitter finds a different number of sentences in the two versions (often a quotation split differently). These pairs are worth checking by hand.

To rebuild the dataset, run these in order:

```bash
python -m src.data.load_xlsum        # 300 Hindi articles
python -m src.data.make_hinglish     # Hinglish generation (resumable)
python -m src.data.check_hinglish    # flags -> data/processed/flagged.csv
python -m src.data.fix_flagged       # retry once, drop failures -> factindic.jsonl
python -m src.data.make_manual_sample
python -m src.data.add_roman         # adds text_hi_roman (Phase 3)
```

---

## 📝 Summarization (Phase 3)

Every article is summarized **6 times**: 3 input conditions × 2 prompts.

| Condition | Source field | What changes compared with the row above |
|---|---|---|
| `hi` | `text_hi` | Devanagari Hindi (baseline) |
| `hi_roman` | `text_hi_roman` | Only the **script** changes (same Hindi words) |
| `hinglish` | `text_hinglish` | **Code-mixing** added (English words, same script) |

**`text_hi_roman`** is a rule-based transliteration of the Hindi article made by `src/data/add_roman.py`. No LLM is involved.
- It uses the ITRANS scheme from `indic-transliteration`, lowercased.
- The silent final "a" is dropped (विमान becomes "viman", not "vimana"), and anusvara becomes "n" (में becomes "men").
- Nukta letters, ऑ and zero-width characters are normalized before transliterating.
- Only the final silent "a" is handled; the silent "a" inside words is not deleted, so उतरने becomes "utarane", not "utarne".

**Prompts** (`prompts/summ_plain.txt` and `prompts/summ_factual.txt`):
- **Plain:** "Summarize the following article in 2-3 sentences."
- **Factual:** the same task plus strict grounding rules:
  - use only information in the article;
  - copy names, numbers, dates and amounts exactly;
  - add no background knowledge, opinions or guesses;
  - don't translate between English and Hindi in code-mixed text;
  - leave out any fact you are unsure about.
- **Output language:** both prompts name it explicitly for each condition and repeat it after the article. With only "same language and script as the article", gemma3:4b answered in the wrong language or script in 8 of 18 test summaries, often English. With the explicit language, 17 of 18 were correct.

**Models:**
- **Main run:** `gemma3:4b`, run locally through [Ollama](https://ollama.com) (`num_ctx=8192`, temperature 0, at most 300 output tokens). The longest article needs only about 800 prompt tokens. It takes about 7 s per summary on an 8 GB Mac.
- **Large-model comparison:** `openai/gpt-oss-120b` via Groq on the first 50 articles. Groq's free-tier daily token limit makes a full run impractical.

```bash
ollama pull gemma3:4b
python -m src.data.add_roman                                   # once
python -m src.summarize --provider ollama --model gemma3:4b    # all articles
python -m src.summarize --provider groq --model openai/gpt-oss-120b --limit 50
```

**Output:** `results/summaries_<model>.jsonl`, one line per summary. Fields: `id`, `condition`, `prompt_type`, `model`, `summary`, `latency_s`, `source_words`, `summary_words`, `compression_ratio` (summary words / source words).

The script is resumable: combinations of `id`, `condition` and `prompt_type` that are already done are skipped, and each summary is appended to the file as soon as it is made. If the API's rate limit is reached, the script stops cleanly; running the same command again resumes.

**Language/script check** (`src/script_check.py`). Every saved summary gets three fields: `script_ok`, `deva_share` (Devanagari letters / all letters) and `en_ratio` (share of words tagged English by the CMI tagger).
- `hi` summaries must be at least 80% Devanagari.
- `hi_roman` and `hinglish` summaries must be at most 20% Devanagari and have `en_ratio` ≤ 0.6.
- The 0.6 threshold is calibrated on our data. All Romanized and Hinglish source articles score at most 0.33, and English summaries score 0.74–0.96. The highest-scoring accepted summary is at 0.45.
- Wrong-language summaries are **kept, not regenerated**. The wrong-language rate is reported as a finding, and these summaries are excluded from the factuality metrics.
- To re-check a whole file: `python -m src.script_check results/summaries_<model>.jsonl`.

### Results so far: gemma3:4b, all 186 articles (1,116 summaries)

Wrong language or script:

| Condition | Plain | Factual |
|---|---|---|
| `hi` | 0 / 186 (0.0%) | 0 / 186 (0.0%) |
| `hi_roman` | 0 / 186 (0.0%) | 6 / 186 (3.2%) |
| `hinglish` | 10 / 186 (5.4%) | 17 / 186 (9.1%) |

- In total, 33 of 1,116 summaries (3.0%) are in the wrong language or script.
- 27 are English summaries of Hinglish articles; 6 are Devanagari summaries of Romanized Hindi articles.
- The factuality-aware prompt causes more language switching than the plain prompt.

Mean summary length (summaries that passed the script check; compression = summary words / source words):

| Condition | Plain | Factual |
|---|---|---|
| `hi` | 73.9 words (0.288) | 78.6 words (0.308) |
| `hi_roman` | 80.9 words (0.315) | 84.7 words (0.327) |
| `hinglish` | 83.3 words (0.322) | 87.5 words (0.344) |

- The mean latency is 10.1 s per summary on an 8 GB Mac, for a total of 3.1 h.
- The `openai/gpt-oss-120b` comparison (50 articles, `hi` + `hinglish` only) is **complete**: 200 of 200 summaries, with **0** in the wrong language or script. Results are in *Evaluation → Model comparison* below.

---

## 🔍 Evaluation (Phase 4)

Everything runs locally (CPU/Apple MPS); no API calls. Only summaries that passed the language/script check are scored: **1,083 of 1,116** gemma3:4b summaries.

```bash
python -m src.evaluate --model gemma3:4b            # resumable; ~30 min on an 8 GB Mac
python -m src.eval.make_annotation_set              # 60 summaries for hand labelling (never overwritten)
python -m src.eval.make_annotation_packs            # split across 5 annotators + offline pages A2-A5
python -m src.eval.label_server                     # A1's labelling page at http://127.0.0.1:8765
python -m src.eval.merge_annotations                # merge 5 label files, Fleiss/Cohen kappa
python -m src.eval.agreement                        # then: metrics vs final human labels
python -m src.evaluate --model openai/gpt-oss-120b  # score the 200 Groq summaries
python -m src.eval.compare_models                   # gpt-oss-120b vs gemma3:4b -> results/model_comparison.csv
python -c "from src.eval.entities import tune_threshold; tune_threshold()"
```

**Manual labels (5 annotators).** The 60 summaries are labelled by a team of five (A1-A5), following `results/annotation_packs/GUIDELINES.md`.
- **Split** (`results/annotation_packs/assignment.csv`): 10 *shared* rows labelled by all five (for agreement) and 50 rows split 10 per annotator. It is stratified, so every annotator and the shared set get rows from all six condition × prompt cells. Each annotator sees 20 rows in their own random order and does not know which rows are shared.
- **A1** labels with `label_server.py` into `labels_A1.csv`, in blind mode by default.
- **A2-A5** label offline in a self-contained page (`A2.html` etc., opened by double-click in Chrome). Labels autosave in the browser, and a button exports `labels_A2.csv`. These pages contain **no LLM draft labels**.
- **Merge** (`merge_annotations.py`): writes every individual label to `all_labels.csv` and one *final* label per summary to `results/manual_annotation.csv` (columns `set`, `annotator`, `label`, `error_type`, `notes`). Shared rows take the majority label; a tie goes to the more severe label. The script prints Fleiss' kappa (all five) and pairwise Cohen's kappa on the shared rows, for 3 labels and for faithful vs error. `agreement.py` uses these final labels.
- **Drafts and anchoring.** `draft_label`, `draft_error_type` and `draft_note` are suggestions written by an LLM (Claude), shown only to A1. `label_source` records how A1's labels relate to the draft (`same_as_draft`, `changed_from_draft`, `_blind` if the draft was hidden), `blind_pack` for teammates, and `majority` / `majority_tie` for merged shared rows. Labels copied from a draft are not independent human judgments, so report these counts alongside the agreement numbers.

**Outputs**
- `results/scores_<model>.csv`: one row per summary.
- `results/summary_table_<model>.csv`: mean and 95% bootstrap CI (1,000 resamples) per condition × prompt.
- `results/cache/gold_entities.json`: cached Stanza NER output.

### Metrics

| Metric | File | What it measures |
|---|---|---|
| Number P/R/F1 | `src/eval/numbers.py` | Numbers from digits ("1,200", "9.15", Devanagari digits) and number words (teen hazaar / तीन हज़ार → 3000, ek lakh, डेढ़ करोड़). Values are compared as sets. Precision < 1 means a hallucinated or altered number. |
| Entity P/R/F1 | `src/eval/entities.py` | Names matched across scripts with a phonetic skeleton (see below). |
| NLI | `src/eval/nli.py` | Each summary sentence is scored by mDeBERTa-v3 XNLI; we take its max entailment over source premises (SummaC-style), then the mean over sentences, and count sentences with entailment < 0.5. |
| ROUGE-1/2/L | `src/eval/overlap.py` | Overlap with the source, and with the reference summary (`hi` only, because the reference is Devanagari). Uses a whitespace tokenizer that keeps Devanagari words intact. |
| BERTScore | `src/eval/overlap.py` | mBERT similarity between summary and source. mBERT reads at most 512 tokens, so only the start of long sources is used. |

**How the entity metric works**
- **Skeleton:** each word is romanized with the same ITRANS rules as `add_roman.py`, similar sounds are merged (ph/f, sh/s, w/v, z/j, aspiration dropped), and later vowels are removed. So "Stansted", स्टैनस्टेड and "stainasted" all become `stnstd`.
- **Gold entities:** Stanza Hindi NER on `text_hi`.
- **Recall:** counted only over the gold entities that can be found in that condition's source text, because the Hinglish rewrite sometimes translates names (रक्षा मंत्रालय → "Defence Ministry").
- **Precision:** for Devanagari summaries, the candidates are Stanza entities; for Roman summaries, runs of capitalized words. A candidate is unsupported if it doesn't match anywhere in the source.

**Entity threshold: 85** (`rapidfuzz.partial_ratio` on skeletons).
- **How it was tuned:** on 20 (gold entity, summary) pairs from real summaries, sampled 4 per score band (55–70, 70–80, 80–88, 88–95, 95–100). Each pair was labelled by hand ("is this entity mentioned in the summary?"); the labels and notes are in `results/entity_threshold_tuning.csv`.
- **Result:** a threshold of 85 classifies 19 of 20 correctly (80: 17/20; 90: 13/20). The one error is a Stanza span that merges a person's name with other words.
- **Caveat:** the margin is narrow. The highest-scoring absent entity is at 83.3 and the lowest-scoring present one at 85.7.

### Results: gemma3:4b (means; CIs in the CSV)

| Condition | Prompt | n | Num P | Num R | Ent P | Ent R | NLI chunk | NLI window | ROUGE-2 src | BERTScore |
|---|---|---|---|---|---|---|---|---|---|---|
| hi | plain | 186 | 0.979 | 0.230 | 0.974 | 0.475 | 0.742 | 0.745 | 0.141 | 0.749 |
| hi | factual | 186 | 0.970 | 0.556 | 0.994 | 0.661 | 0.613 | 0.727 | 0.259 | 0.787 |
| hi_roman | plain | 186 | 0.999 | 0.390 | 0.995 | 0.553 | 0.475 | 0.713 | 0.388 | 0.788 |
| hi_roman | factual | 180 | 1.000 | 0.513 | 1.000 | 0.624 | 0.454 | 0.708 | 0.440 | 0.792 |
| hinglish | plain | 176 | 0.981 | 0.352 | 0.979 | 0.534 | 0.511 | 0.632 | 0.248 | 0.746 |
| hinglish | factual | 169 | 1.000 | 0.523 | 0.997 | 0.675 | 0.397 | 0.647 | 0.419 | 0.786 |

**Main observations**
- **Precision is at the ceiling.** Number and entity precision is at least 0.97 in every cell: gemma3:4b rarely invents numbers or names. The cases we found by hand are a rounded number ("char lakh" for 4 lakh 40 hazaar) and an added first name ("जेम्स एंडरसन" where the source has only "एंडरसन").
- **The factual prompt mainly raises recall.** Number recall rises from 0.23 to 0.56 for `hi`, from 0.35 to 0.52 for `hinglish`, and from 0.39 to 0.51 for `hi_roman`. Entity recall rises in the same way. The factual prompt also makes summaries more extractive: ROUGE-2 against the source goes from 0.14 to 0.26 for `hi`.
- **The two NLI variants disagree about the prompt effect.**
  - With large chunks, the factual prompt scores *lower* than the plain prompt in every condition (`hi`: 0.74 → 0.61).
  - With 2-sentence windows, the two prompts score about the same.
  - One possible explanation (not tested) is that factual summaries pack more facts into each sentence, which long-premise NLI penalizes.
  - The primary NLI variant will be chosen by agreement with the manual labels (`src/eval/agreement.py`, point-biserial r and AUC against faithful-vs-error labels).

### Model comparison: gpt-oss-120b vs gemma3:4b (same 50 articles)

`openai/gpt-oss-120b` (Groq) was run on only 50 articles and two conditions, so both models are compared on exactly those (article, condition, prompt) rows (`src/eval/compare_models.py` → `results/model_comparison.csv`).
- The wrong-language rate is computed over all 50 summaries per cell.
- All other columns are means over the rows where **both** models passed the language check (n), so both models are scored on the same inputs.
- Num P averages only the summaries that contain numbers.

| Condition | Prompt | Model | Wrong lang | n | Num P | Num R | Ent P | Ent R | NLI chunk | NLI window | ROUGE-L src | BERTScore | Words | Latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hi | plain | gemma3:4b | 0% | 50 | 0.926 | 0.197 | 0.985 | 0.461 | 0.729 | 0.739 | 0.212 | 0.752 | 75.3 | 7.4 |
| hi | plain | gpt-oss-120b | 0% | 50 | 0.958 | 0.465 | 0.987 | 0.543 | 0.669 | 0.624 | 0.223 | 0.764 | 76.2 | 5.0 |
| hi | factual | gemma3:4b | 0% | 50 | 0.987 | 0.499 | 1.000 | 0.657 | 0.625 | 0.721 | 0.297 | 0.788 | 78.2 | 8.2 |
| hi | factual | gpt-oss-120b | 0% | 50 | 0.996 | 0.651 | 0.980 | 0.729 | 0.566 | 0.643 | 0.382 | 0.799 | 96.2 | 5.6 |
| hinglish | plain | gemma3:4b | 10% | 45 | 0.994 | 0.311 | 0.962 | 0.590 | 0.576 | 0.623 | 0.312 | 0.743 | 82.7 | 8.8 |
| hinglish | plain | gpt-oss-120b | 0% | 45 | 1.000 | 0.457 | 0.987 | 0.605 | 0.510 | 0.545 | 0.309 | 0.740 | 82.8 | 5.6 |
| hinglish | factual | gemma3:4b | 14% | 43 | 1.000 | 0.504 | 0.996 | 0.675 | 0.445 | 0.642 | 0.428 | 0.777 | 87.0 | 9.7 |
| hinglish | factual | gpt-oss-120b | 0% | 43 | 1.000 | 0.660 | 1.000 | 0.742 | 0.422 | 0.569 | 0.429 | 0.779 | 99.1 | 6.2 |

**Observations**
- **Language control:** gpt-oss-120b never answered in the wrong language or script. gemma3:4b did so for 10-14% of Hinglish summaries.
- **gpt-oss-120b covers more facts.** Number recall is higher in every cell (`hi` / plain: 0.20 → 0.47), and entity recall is higher too.
- **Precision is at the ceiling for both models** (≥ 0.93 numbers, ≥ 0.96 entities), so these metrics barely separate the models.
- **NLI favours gemma3:4b in every cell and in both variants** (e.g. window NLI for `hi` / plain: 0.74 vs 0.62). This does not by itself mean gpt-oss-120b hallucinates more:
  - its summaries carry more facts per sentence, and for `factual` they are longer (96 vs 78 words for `hi`);
  - the plain-vs-factual comparison above shows that NLI penalises denser summaries.
  - Whether NLI or the number and entity metrics is right here is a question for the manual labels (`agreement.py`).
- **Latency is not a speed comparison.** gemma3:4b ran locally on an 8 GB Mac. gpt-oss-120b ran on Groq's servers, and its time includes the network and the model's hidden reasoning tokens.
- These are 43-50 summaries per cell with no confidence intervals, so read small differences (ROUGE, BERTScore, precision) as ties.

### Limitations: read before comparing numbers

- **NLI scores are only comparable *within* a condition (plain vs. factual), not *across* conditions.**
  - The NLI model was trained on NLI data where Hindi appears in Devanagari. Romanized Hindi (`hi_roman`) is out of distribution, and code-mixed Hinglish mostly is too. Lower NLI scores for those conditions partly measure the model's unfamiliarity with the script, not hallucination.
  - **Cross-condition comparisons (Hindi vs. Romanized vs. Hinglish) should rely on the number and entity metrics.** Both work on the phonetic skeleton and the number value, so they don't depend on the script.
- **Relation errors are not caught by any automatic metric.** When a summary uses the right entities and numbers in the wrong relation, every metric passes it. Example: the Dreamliner `hi` / `plain` summary says Boeing launched *the Dreamliner* in 1995, when the source says 1995 was the launch of the *Boeing-777*. That sentence gets number precision 1.0, entity precision 1.0 and NLI 0.86. This is why we hand-label 60 summaries (`results/manual_annotation.csv`: faithful / minor_error / major_error plus the error type).
- **`hi_roman` entity precision is optimistic (1.000).** Romanized Hindi summaries are all lowercase, so the only candidates are gold entities found in the summary; a hallucinated lowercase name can't be detected.
- **Other approximations:**
  - Stanza sometimes merges or misses entities. A merged multi-word entity counts as supported if each of its words is found in the source, so a mix-up of two real names isn't detected.
  - The number parser ignores a lone "ek"/"एक" (usually the article "a"), and it reads the surname "Das" as 10. It handles arab/अरब (10⁹) only after a number, so "Saudi Arab" isn't read as a number.
  - Rounding ("दो लाख से अधिक" for 2 lakh 30 hazaar) counts as an altered number.
  - ROUGE F1 against the source is low in absolute terms because the source is much longer than the summary; compare conditions, not absolute values.

---

## 📂 Project Structure

```
FactIndic/
├── data/
│   ├── raw/             # Raw downloaded articles (gitignored)
│   └── processed/       # Paired Hindi and synthetic Hinglish datasets
├── docs/                # Project briefs and documentation
├── notebooks/           # Interactive exploration and data analysis
├── paper/               # IEEE formatted research paper LaTeX files
├── prompts/             # Plain and factuality-aware prompt templates
├── results/             # Evaluation scores, CSV files, and charts
├── src/
│   ├── __init__.py
│   ├── llm_client.py    # Unified API client (Gemini/Groq) with backoff & latency
│   └── smoke_test.py    # End-to-end smoke test script
├── tests/               # Automated unit tests (pytest)
├── .env.example         # Template environment configuration
├── .gitignore           # Git ignore rules for Python, datasets, and secrets
├── PROJECT.md           # Master roadmap and viva explanation guide
├── README.md            # Setup guide and documentation (this file)
└── requirements.txt     # Python dependencies
```

---

## 🗺️ Project Roadmap

- **Phase 1: Setup & Foundations** *(Current)* — Directory layout, unified LLM client, rate-limit exponential backoff, smoke tests.
- **Phase 2: Data Acquisition** — Curate ~300 Hindi articles from XL-Sum and generate aligned Hinglish code-mixed variants.
- **Phase 3: Summarization Experiments** — Plain prompts vs. Factuality-aware prompts across Hindi and Hinglish.
- **Phase 4: Fact-Checking Evaluation** — Entity/numerical precision-recall-F1 (Stanza + regex) and NLI entailment scores (`mDeBERTa-v3-base-xnli`).
- **Phase 5: Results Analysis** — Comprehensive comparative tables, error analysis, and hallucination breakdown visualizations.
- **Phase 6: Deliverables** — Final IEEE conference paper and viva presentation slides.

For detailed descriptions of each phase, see [PROJECT.md](file:///Users/as/Desktop/FactIndic/PROJECT.md).

---

## 💡 Viva Talking Points

1. **Why not fine-tune a model?**
   Fine-tuning small multilingual models (like mBART or mT5) requires substantial domain-specific hallucination datasets and high-compute training. Evaluating zero-shot / few-shot factuality-aware prompting on modern foundation LLMs reflects real-world API-based deployments and tests whether prompt engineering can enforce grounding.
2. **How does code-mixing (Hinglish) affect factuality?**
   Hinglish lacks standardized spelling and grammar. Subword tokenizers often fragment Hinglish words into unnatural sub-tokens, increasing the risk of entity corruption and hallucination.
3. **What makes the evaluation pipeline robust?**
   Traditional ROUGE and BLEU only measure n-gram overlap with reference summaries (which themselves may omit source facts). Our pipeline combines **symbolic verification** (exact entity & numerical matching) with **deep semantic verification** (cross-lingual NLI entailment).

