"""
FactIndic - Step 2: generate Hinglish versions of the Hindi articles
====================================================================
Each Hindi article is sent to the LLM with prompts/to_hinglish.txt at
temperature 0 (deterministic, reproducible). The result is saved to
data/processed/factindic_300.jsonl.

Resumable: ids already present in the output file are skipped, and every
article is appended to the file as soon as it is done, so a crash or a rate
limit never loses finished work - just run the script again.

Run:  python -m src.data.make_hinglish            (all articles)
      python -m src.data.make_hinglish --limit 5  (quick test on 5)
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.llm_client import generate
from src.text_utils import normalize

IN_PATH = Path("data/raw/hindi_300.jsonl")
OUT_PATH = Path("data/processed/factindic_300.jsonl")
PROMPT_PATH = Path("prompts/to_hinglish.txt")


def read_jsonl(path: Path) -> list:
    """Read a .jsonl file (one JSON object per line) into a list of dicts."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list):
    """Write a list of dicts to a .jsonl file (overwrites)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def to_hinglish(text_hi: str, feedback: str = "") -> str:
    """Rewrite one Hindi article into Hinglish with the LLM.

    `feedback` is used only when re-generating a pair that failed the quality
    check: it tells the model what went wrong the first time (e.g. which
    numbers were missing), so the second attempt is not just a repeat.
    """
    template = PROMPT_PATH.read_text(encoding="utf-8")
    # .replace instead of .format so curly braces inside articles are harmless
    prompt = template.replace("{article}", text_hi)
    if feedback:
        prompt += f"\n\nIMPORTANT - a previous attempt had these problems, fix them: {feedback}"
    output, _latency = generate(prompt, temperature=0.0)
    return normalize(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N articles (for testing)")
    args = parser.parse_args()

    articles = read_jsonl(IN_PATH)
    if args.limit:
        articles = articles[: args.limit]

    # Resume support: skip anything already in the output file.
    done_ids = {rec["id"] for rec in read_jsonl(OUT_PATH)}
    todo = [a for a in articles if a["id"] not in done_ids]
    print(f"{len(articles)} articles, {len(done_ids)} already done, {len(todo)} to go")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    for art in tqdm(todo, desc="Hindi -> Hinglish"):
        try:
            hinglish = to_hinglish(art["text_hi"])
        except Exception as e:
            # Rate limit retries exhausted or API error: skip for now,
            # a re-run will pick this article up again.
            failed += 1
            tqdm.write(f"[skip] {art['id']}: {type(e).__name__}: {str(e)[:150]}")
            continue
        if not hinglish:
            failed += 1
            tqdm.write(f"[skip] {art['id']}: empty output")
            continue

        record = {
            "id": art["id"],
            "title": art["title"],
            "text_hi": art["text_hi"],
            "text_hinglish": hinglish,
            "ref_summary_hi": art["ref_summary_hi"],
        }
        # Append immediately so nothing is lost if the script stops.
        with open(OUT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done. Output: {OUT_PATH} ({len(read_jsonl(OUT_PATH))} records)")
    if failed:
        print(f"{failed} articles failed - run the script again to retry them.")


if __name__ == "__main__":
    main()
