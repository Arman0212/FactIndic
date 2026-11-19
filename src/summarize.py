"""
FactIndic - Phase 3: generate summaries
=======================================
For every article we create 6 summaries:

    3 input conditions  x  2 prompt types
    hi        (Devanagari Hindi)            plain    (prompts/summ_plain.txt)
    hi_roman  (Romanized Hindi, no English) factual  (prompts/summ_factual.txt)
    hinglish  (code-mixed, Roman script)

hi -> hi_roman isolates the effect of the SCRIPT; hi_roman -> hinglish
isolates the effect of CODE-MIXING; plain -> factual measures whether
factuality-aware prompting helps.

Each summary is saved as one JSON line in results/summaries_<model>.jsonl.
The script is resumable: (id, condition, prompt_type) combinations already in
the file are skipped, and each summary is appended as soon as it is made.

Examples:
    python -m src.summarize --provider ollama --model gemma3:4b --limit 3
    python -m src.summarize --provider ollama --model gemma3:4b          # all
    python -m src.summarize --provider groq --model openai/gpt-oss-120b --limit 50
"""

import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

from src.data.fix_flagged import FINAL_PATH
from src.data.make_hinglish import read_jsonl
from src.llm_client import generate
from src.text_utils import normalize

RESULTS_DIR = Path("results")

# Which dataset field is the source text for each condition
CONDITIONS = {
    "hi": "text_hi",
    "hi_roman": "text_hi_roman",
    "hinglish": "text_hinglish",
}

# The output language for each condition, inserted into the prompt's
# {language} slot. Small models ignore a vague "same language and script as
# the article" (in a 3-article test gemma3:4b answered in the wrong language
# or script 8 times out of 18, often in English), so we name the language
# explicitly and repeat it after the article. Both prompt types get the same
# sentence, so the plain-vs-factual comparison stays fair.
LANGUAGES = {
    "hi": "Hindi in Devanagari script",
    "hi_roman": ("Hindi written in Roman (English) letters, exactly like the article. "
                 "Do not use Devanagari script and do not translate into English"),
    "hinglish": ("Hinglish (Hindi mixed with English words) written in Roman letters, "
                 "exactly like the article. Do not use Devanagari script and do not "
                 "translate fully into English"),
}

# Which prompt file belongs to each prompt type
PROMPTS = {
    "plain": Path("prompts/summ_plain.txt"),
    "factual": Path("prompts/summ_factual.txt"),
}


def output_path(model: str) -> Path:
    """results/summaries_<model>.jsonl, with characters that are not safe in
    file names replaced: "gemma3:4b" -> "gemma3_4b",
    "openai/gpt-oss-120b" -> "openai_gpt-oss-120b"."""
    safe = re.sub(r"[^A-Za-z0-9.-]+", "_", model)
    return RESULTS_DIR / f"summaries_{safe}.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Generate summaries for all conditions and prompts.")
    parser.add_argument("--provider", default="ollama", help="ollama, groq or gemini (default: ollama)")
    parser.add_argument("--model", default="gemma3:4b", help="model name (default: gemma3:4b)")
    parser.add_argument("--limit", type=int, default=None, help="only use the first N articles")
    args = parser.parse_args()

    articles = read_jsonl(FINAL_PATH)
    if args.limit:
        articles = articles[: args.limit]
    templates = {name: path.read_text(encoding="utf-8") for name, path in PROMPTS.items()}

    out_path = output_path(args.model)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: collect the (id, condition, prompt_type) already done
    done = {(r["id"], r["condition"], r["prompt_type"]) for r in read_jsonl(out_path)}

    # Build the list of jobs still to do (article x condition x prompt)
    jobs = [
        (art, cond, ptype)
        for art in articles
        for cond in CONDITIONS
        for ptype in PROMPTS
        if (art["id"], cond, ptype) not in done
    ]
    total = len(articles) * len(CONDITIONS) * len(PROMPTS)
    print(f"Model {args.provider}/{args.model}: {total} summaries, "
          f"{total - len(jobs)} already done, {len(jobs)} to go -> {out_path}")

    for art, cond, ptype in tqdm(jobs, desc="Summarizing"):
        source = art[CONDITIONS[cond]]
        # .replace instead of .format so curly braces inside articles are harmless
        prompt = templates[ptype].replace("{language}", LANGUAGES[cond])
        prompt = prompt.replace("{article}", source)

        summary, latency = generate(prompt, temperature=0.0,
                                    provider=args.provider, model=args.model)
        summary = normalize(summary)

        source_words = len(source.split())
        summary_words = len(summary.split())
        record = {
            "id": art["id"],
            "condition": cond,
            "prompt_type": ptype,
            "model": args.model,
            "summary": summary,
            "latency_s": latency,
            "source_words": source_words,
            "summary_words": summary_words,
            "compression_ratio": round(summary_words / source_words, 4),
        }
        # Append immediately so nothing is lost if the run stops
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done: {len(read_jsonl(out_path))} summaries in {out_path}")


if __name__ == "__main__":
    main()
