# FactIndic: Factuality-Aware Summarization for Hindi & Hinglish Text

**Course Project:** CSET 346 - Natural Language Processing  
**Project Name:** FactIndic  
**Assignment Reference:** `docs/brief.pdf`

---

## 1. Project Overview & Core Mission

**FactIndic** focuses on studying and improving **factual consistency** in abstractive text summarization for **Hindi** and **Hindi-English code-mixed (Hinglish)** text. 

A persistent vulnerability of Large Language Models (LLMs) in low-resource and multilingual/code-mixed scenarios is **hallucination**—generating information that contradicts, distorts, or is unsupported by the source article. FactIndic systematically evaluates whether targeted **factuality-aware prompting** can suppress hallucinations and preserve numerical/named entity factual fidelity compared to vanilla prompting.

### Key Architectural Constraint:
> **Zero Model Training:** We will **NOT** train or fine-tune models from scratch. Instead, we interact with state-of-the-art LLMs (Gemini, Groq-hosted open models like LLaMA 3.3) purely via APIs to generate summaries, followed by automated and metric-driven factual consistency evaluation.

---

## 2. Complete Project Roadmap

### Phase 1: Setup & Foundations *(Current Phase)*
- [x] Establish standardized repository layout (`data/`, `src/`, `prompts/`, `results/`, `notebooks/`, `paper/`, `tests/`).
- [x] Configure dependency management (`requirements.txt`) covering API clients, evaluation libraries, NLP tooling, and data manipulation.
- [x] Implement robust, unified LLM client (`src/llm_client.py`) supporting Groq and Gemini with exponential backoff on rate limits and latency tracking.
- [x] Build smoke test suite (`src/smoke_test.py`) validating both Hindi and Hinglish summarization flows.
- [x] Configure `.gitignore` and `.env.example` templates ensuring security compliance.
- [x] Initialize version control with baseline project commit.

### Phase 2: Data Acquisition & Preparation
- Target Dataset: ~300 articles from the **XL-Sum** Hindi benchmark (`csebuetnlp/xlsum` on Hugging Face).
- Hinglish Synthesis: LLM-assisted code-mixing generation of Hindi articles into natural, colloquial Romanized / Devanagari Hindi-English code-mixed text while strictly preserving underlying factual content.
- Data Validation: Quality checks to ensure alignment between original Hindi and synthetic Hinglish pairs.
- Artifact Storage: Store raw samples in `data/raw/` and processed/paired datasets in `data/processed/`.

### Phase 3: Summarization Experiments
- Define prompt templates in `prompts/`:
  - **Baseline (Plain Prompt):** Standard instruction to summarize the passage concisely.
  - **Factuality-Aware Prompt:** Prompt engineering incorporating strict factual constraints, entity/numerical grounding instructions, source attribution, and self-check/chain-of-thought verification steps.
- Experimental Grid:
  - Hindi Source $\times$ Plain Prompt
  - Hindi Source $\times$ Factuality-Aware Prompt
  - Hinglish Source $\times$ Plain Prompt
  - Hinglish Source $\times$ Factuality-Aware Prompt
- Execution: Batch run through `src/llm_client.py`, logging outputs, timestamps, token usage, and latency into `results/`.

### Phase 4: Fact-Checking & Evaluation Pipeline
- **Entity & Number Precision-Recall-F1:**
  - Named Entity Recognition (NER) using `stanza` Hindi models and multilingual tools.
  - Deterministic regex extraction for numbers, dates, units, and monetary values.
  - Compute precision, recall, and F1 between entities/numbers in source vs. summary (measuring hallucination vs. omission).
- **Natural Language Inference (NLI):**
  - Premise: Source passage.
  - Hypothesis: Summary sentence(s).
  - Model: Cross-lingual NLI model (e.g. `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`).
  - Metric: Entailment, neutral, and contradiction classification scores.
- **Lexical & Semantic Quality:**
  - **ROUGE:** ROUGE-1, ROUGE-2, and ROUGE-L (F1 scores).
  - **BERTScore:** Multilingual contextual embedding similarity (`bert-score`).
- **Compression Ratio:** Summary token count relative to source token count.

### Phase 5: Results Analysis & Visualizations
- Generate consolidated comparative evaluation tables across all configurations.
- Produce publication-ready visualization figures (`results/`):
  - Hindi vs. Hinglish performance comparison charts.
  - Plain vs. Factuality-aware factual consistency improvement deltas.
  - Hallucination rate analysis across entity types (persons, locations, numbers).
- Comprehensive error analysis categorizing specific failure modes (e.g., entity swapping, number corruption, code-mix translation artifacts).

### Phase 6: Final Deliverables
- **IEEE Conference Format Paper:** Written in LaTeX within `paper/`, detailing methodology, metrics, comparative evaluation, and findings.
- **Presentation Slides:** Slide deck synthesized for viva and final project defense.

---

## 3. Directory Layout

```
FactIndic/
├── data/
│   ├── raw/             # Raw downloaded XL-Sum articles (gitignored)
│   └── processed/       # Filtered, paired Hindi & Hinglish datasets
├── docs/                # Project documentation & assignment brief (brief.pdf)
├── notebooks/           # Exploratory data analysis & prototyping
├── paper/               # IEEE formatted paper LaTeX sources & figures
├── prompts/             # Plain and factuality-aware prompt templates
├── results/             # Evaluation metrics, CSV outputs, and visual plots
├── src/
│   ├── __init__.py
│   ├── llm_client.py    # Unified Groq / Gemini API wrapper with retries
│   └── smoke_test.py    # Verification test for API connectivity
├── tests/               # Pytest test suites
├── .env.example         # Template for environment keys
├── .gitignore           # Python, credentials, and artifact exclusions
├── PROJECT.md           # Roadmap and ongoing context reference (this document)
├── README.md            # Quickstart, setup, and execution guide
└── requirements.txt     # Python project dependencies
```

---

## 4. Key Viva & Presentation Talking Points
- **Why Hindi and Hinglish?** Hindi is widely spoken yet resource-constrained in advanced NLP benchmarks; Hinglish is the de facto mode of digital communication across South Asia, presenting unique syntactic, morphological, and orthographic challenges for standard LLMs.
- **Why Factuality-Aware Prompting over Fine-Tuning?** Fine-tuning requires massive annotated datasets of hallucinations and significant compute. Prompt engineering enables lightweight, modular, and explainable grounding mechanisms applicable across any black-box foundation model.
- **Verification Strategy:** Combining deterministic symbolic checks (entity/number matching) with deep semantic models (mDeBERTa XNLI) provides a two-pronged, rigorous evaluation that avoids LLM self-evaluation bias.

