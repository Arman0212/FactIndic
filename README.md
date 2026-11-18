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

