"""
FactIndic - Smoke Test
======================
Validates connectivity, rate-limit handling, and summarization generation
on both pure Hindi and code-mixed (Hinglish) sample inputs.

Usage:
------
    python -m src.smoke_test
or:
    python src/smoke_test.py
"""

import os
import sys
from dotenv import load_dotenv

# Ensure the project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm_client import generate


def run_smoke_test():
    """Runs a quick end-to-end smoke test for Hindi and Hinglish summarization."""
    load_dotenv()

    provider = os.getenv("LLM_PROVIDER", "gemini")
    model = os.getenv("LLM_MODEL", "(default)")

    print("=" * 70)
    print(" FactIndic - Phase 1 Smoke Test")
    print("=" * 70)
    print(f"Provider : {provider}")
    print(f"Model    : {model}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Sample 1: Hindi Paragraph
    # -------------------------------------------------------------------------
    hindi_paragraph = (
        "भारतीय अंतरिक्ष अनुसंधान संगठन (इसरो) ने श्रीहरिकोटा के सतीश धवन अंतरिक्ष "
        "केंद्र से अपने नए मौसम उपग्रह इनसैट-3डीएस (INSAT-3DS) को सफलतापूर्वक लॉन्च किया। "
        "यह 2275 किलोग्राम वजनी उपग्रह मौसम पूर्वानुमान, चक्रवात की पूर्व चेतावनी और "
        "समुद्री निगरानी को पहले से कहीं अधिक सटीक बनाएगा।"
    )

    hindi_prompt = (
        f"Summarize the following Hindi paragraph in one single sentence in Hindi:\n\n"
        f"{hindi_paragraph}"
    )

    print("\n[Test 1/2] Generating Hindi Summary...")
    print(f"Input Text:\n\"{hindi_paragraph}\"\n")

    try:
        hindi_summary, hindi_latency = generate(
            prompt=hindi_prompt,
            system="You are an expert multilingual summarizer. Summarize concisely while retaining key factual details.",
            temperature=0.0,
        )
        print(f"Generated Summary ({hindi_latency:.2f}s):")
        print(f"\"{hindi_summary}\"")
        print("-" * 70)
    except Exception as e:
        print(f"[ERROR] Hindi summarization failed: {e}")
        return False

    # -------------------------------------------------------------------------
    # Sample 2: Hinglish Paragraph (Hindi-English Code-Mixed)
    # -------------------------------------------------------------------------
    hinglish_paragraph = (
        "Kal raat Indian cricket team ne T20 World Cup final me South Africa ko "
        "7 runs se defeat karke historic victory achieve ki. Virat Kohli ne crucial "
        "76 runs banaye jab team early wickets gawa chuki thi. Jasprit Bumrah aur "
        "Hardik Pandya ne death overs me zabardast bowling performance show kiya, "
        "jisse India ne 11 saal baad ICC World Cup trophy apne naam ki."
    )

    hinglish_prompt = (
        f"Summarize the following Hinglish paragraph in one single sentence in Hinglish:\n\n"
        f"{hinglish_paragraph}"
    )

    print("\n[Test 2/2] Generating Hinglish Summary...")
    print(f"Input Text:\n\"{hinglish_paragraph}\"\n")

    try:
        hinglish_summary, hinglish_latency = generate(
            prompt=hinglish_prompt,
            system="You are an expert multilingual summarizer. Summarize code-mixed Hinglish concisely while retaining key entities and numbers.",
            temperature=0.0,
        )
        print(f"Generated Summary ({hinglish_latency:.2f}s):")
        print(f"\"{hinglish_summary}\"")
        print("-" * 70)
    except Exception as e:
        print(f"[ERROR] Hinglish summarization failed: {e}")
        return False

    print("\n[SUCCESS] Smoke test completed successfully for both Hindi and Hinglish!")
    print(f"Total time across both calls: {hindi_latency + hinglish_latency:.2f}s\n")
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)

