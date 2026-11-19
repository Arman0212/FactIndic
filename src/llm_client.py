"""
FactIndic - LLM Client Module
=============================
Provides a unified interface `generate()` to query LLMs (Groq and Google Gemini)
with automatic rate-limit retries, exponential backoff, and latency tracking.

Viva Explanation Points:
------------------------
1. Why an abstraction layer?
   - Decouples our NLP summarization and evaluation pipelines from the underlying
     LLM provider. We can switch between Groq (open models like LLaMA) and Google
     Gemini with a single environment variable change.
2. Why temperature=0?
   - Abstractive summarization and factuality evaluation require reproducibility and
     deterministic outputs. A temperature of 0 minimizes sampling randomness.
3. How is rate limiting handled?
   - Both Groq and Gemini APIs impose Requests-Per-Minute (RPM) and Tokens-Per-Minute
     (TPM) quotas on free/tier accounts. When HTTP 429 is received, exponential backoff
     waits progressively longer (e.g. 2s, 4s, 8s...) before retrying.
"""

import os
import time
from typing import Optional, Tuple
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env (searches current directory and parents)
load_dotenv(find_dotenv(usecwd=True))


def _call_gemini(
    prompt: str,
    system: Optional[str],
    model: str,
    temperature: float,
    api_key: str,
) -> str:
    """Internal helper to invoke Google Gemini via the official google-genai SDK."""
    from google import genai
    from google.genai import types

    # Initialize Gemini client with the given API key
    client = genai.Client(api_key=api_key)

    # Configure generation parameters (temperature, optional system instructions)
    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system if system else None,
    )

    # Call the Gemini model
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    if not response.text:
        return ""
    return response.text.strip()


def _call_groq(
    prompt: str,
    system: Optional[str],
    model: str,
    temperature: float,
    api_key: str,
) -> str:
    """Internal helper to invoke Groq API using the groq SDK."""
    from groq import Groq

    # Initialize Groq client
    client = Groq(api_key=api_key)

    # Prepare chat messages
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # Extra settings for reasoning models (e.g. openai/gpt-oss-120b): they
    # "think" before answering and the thinking tokens count against the
    # output limit. With the default limit the thinking can use up the whole
    # budget and the answer is cut off, so we keep reasoning low and raise
    # the limit.
    extra = {}
    if "gpt-oss" in model:
        extra["reasoning_effort"] = "low"

    # Call the Groq chat completion endpoint
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=8192,
        **extra,
    )

    choice = response.choices[0]
    # Never silently return a truncated answer.
    if choice.finish_reason == "length":
        raise RuntimeError("Output truncated (finish_reason=length): hit max_completion_tokens")
    content = choice.message.content
    return content.strip() if content else ""


def _is_rate_limit_error(error: Exception) -> bool:
    """
    Check if an exception is related to rate limiting (HTTP 429 / Resource Exhausted).
    Works across both Groq and Gemini SDK error structures.
    """
    err_str = str(error).lower()
    
    # Check for HTTP status code 429
    if hasattr(error, "code") and getattr(error, "code") == 429:
        return True
    if hasattr(error, "status_code") and getattr(error, "status_code") == 429:
        return True

    # Check for common rate limit strings in error representation
    rate_limit_keywords = [
        "429",
        "rate limit",
        "resource exhausted",
        "resourceexhausted",
        "quota exceeded",
        "too many requests",
    ]
    return any(keyword in err_str for keyword in rate_limit_keywords)


def generate(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.0,
    max_retries: int = 5,
    initial_delay: float = 2.0,
) -> Tuple[str, float]:
    """
    Generate text from an LLM with automated rate-limit retries and latency tracking.

    Parameters:
    -----------
    prompt : str
        The input user prompt to send to the LLM.
    system : Optional[str]
        Optional system instructions to steer the model's persona or constraints.
    temperature : float, default=0.0
        Sampling temperature (0.0 for deterministic, factual outputs).
    max_retries : int, default=5
        Maximum number of retry attempts upon encountering rate limit (429) errors.
    initial_delay : float, default=2.0
        Initial delay in seconds for exponential backoff.

    Returns:
    --------
    Tuple[str, float]
        A tuple containing (generated_text, latency_in_seconds).

    Raises:
    -------
    ValueError:
        If LLM_PROVIDER is unsupported or required API keys are missing.
    Exception:
        If all retries are exhausted or a non-rate-limit API error occurs.
    """
    # 1. Read configuration from environment variables (.env)
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

    if provider == "gemini":
        model = os.getenv("LLM_MODEL", "gemini-2.5-flash").strip()
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. Please set GEMINI_API_KEY in your .env file."
            )
        caller = lambda: _call_gemini(
            prompt=prompt,
            system=system,
            model=model,
            temperature=temperature,
            api_key=api_key,
        )

    elif provider == "groq":
        model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile").strip()
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Please set GROQ_API_KEY in your .env file."
            )
        caller = lambda: _call_groq(
            prompt=prompt,
            system=system,
            model=model,
            temperature=temperature,
            api_key=api_key,
        )

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: '{provider}'. Supported providers are 'gemini' and 'groq'."
        )

    # 2. Execute with exponential backoff on rate limits
    start_time = time.perf_counter()
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            output_text = caller()
            latency = round(time.perf_counter() - start_time, 3)
            return output_text, latency

        except Exception as e:
            # If it's a rate limit error and we have retries remaining, back off and retry
            if _is_rate_limit_error(e) and attempt < max_retries:
                print(
                    f"[Rate Limit] Encountered 429/quota error. Retrying in {delay:.1f}s "
                    f"(Attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(delay)
                delay *= 2.0  # Exponential increase (2s, 4s, 8s, 16s...)
            else:
                # Re-raise if non-rate-limit error or max retries exceeded
                raise e
