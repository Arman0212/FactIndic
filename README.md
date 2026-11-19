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

**Final file:** `data/processed/factindic.jsonl`: **186 Hindi-Hinglish pairs**. Fields: `id`, `title`, `text_hi`, `text_hinglish`, `ref_summary_hi`, `cmi`.

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
```

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

