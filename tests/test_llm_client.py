"""
Unit tests for src/llm_client.py
"""

import os
from unittest.mock import MagicMock, patch
import pytest

from src.llm_client import generate, _is_rate_limit_error


def test_is_rate_limit_error():
    assert _is_rate_limit_error(Exception("429 Too Many Requests"))
    assert _is_rate_limit_error(Exception("Rate limit reached"))
    assert _is_rate_limit_error(Exception("ResourceExhausted quota exceeded"))
    assert not _is_rate_limit_error(Exception("Invalid API key"))
    assert not _is_rate_limit_error(ValueError("Bad parameter"))


def test_missing_api_key_raises_value_error():
    with patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": ""}):
        with pytest.raises(ValueError, match="GEMINI_API_KEY is missing"):
            generate("Hello")

    with patch.dict(os.environ, {"LLM_PROVIDER": "groq", "GROQ_API_KEY": ""}):
        with pytest.raises(ValueError, match="GROQ_API_KEY is missing"):
            generate("Hello")


def test_unsupported_provider_raises_value_error():
    with patch.dict(os.environ, {"LLM_PROVIDER": "unsupported_llm"}):
        with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
            generate("Hello")


def test_gemini_generation_success():
    with patch.dict(
        os.environ,
        {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test_key", "LLM_MODEL": "gemini-2.5-flash"},
    ):
        with patch("src.llm_client._call_gemini", return_value="Mocked Hindi summary") as mock_call:
            text, latency = generate(prompt="Sample prompt", system="System instructions")
            assert text == "Mocked Hindi summary"
            assert isinstance(latency, float)
            assert latency >= 0.0
            mock_call.assert_called_once()


def test_groq_generation_success():
    with patch.dict(
        os.environ,
        {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "test_groq_key", "LLM_MODEL": "llama-3.3-70b-versatile"},
    ):
        with patch("src.llm_client._call_groq", return_value="Mocked Hinglish summary") as mock_call:
            text, latency = generate(prompt="Sample prompt", system="System instructions")
            assert text == "Mocked Hinglish summary"
            assert isinstance(latency, float)
            assert latency >= 0.0
            mock_call.assert_called_once()


def test_rate_limit_retry_backoff():
    with patch.dict(
        os.environ,
        {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test_key"},
    ):
        # Fail once with rate limit, then succeed
        mock_caller = MagicMock(side_effect=[Exception("429 Resource Exhausted"), "Recovered summary"])
        with patch("src.llm_client._call_gemini", mock_caller):
            with patch("time.sleep", return_value=None) as mock_sleep:
                text, latency = generate("Test prompt", max_retries=2, initial_delay=0.1)
                assert text == "Recovered summary"
                assert mock_caller.call_count == 2
                mock_sleep.assert_called_once_with(0.1)



def test_ollama_generation_sends_options():
    """generate(provider="ollama") must POST to the local Ollama API with
    temperature 0, an 8192-token context and a 300-token output limit."""
    fake_response = MagicMock()
    fake_response.json.return_value = {"message": {"content": "  Mocked local summary  "}}
    with patch("src.llm_client.requests.post", return_value=fake_response) as mock_post:
        text, latency = generate("Sample prompt", provider="ollama", model="gemma3:4b")

    assert text == "Mocked local summary"
    assert isinstance(latency, float)
    url = mock_post.call_args.args[0]
    body = mock_post.call_args.kwargs["json"]
    assert url.endswith("/api/chat")
    assert body["model"] == "gemma3:4b"
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0.0, "num_ctx": 8192, "num_predict": 300}


def test_explicit_provider_uses_its_default_model():
    """If .env says groq but we ask for ollama without a model, we must not
    send the groq model name to Ollama."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "groq", "LLM_MODEL": "openai/gpt-oss-120b"}):
        with patch("src.llm_client._call_ollama", return_value="ok") as mock_call:
            generate("Sample prompt", provider="ollama")
    assert mock_call.call_args.kwargs["model"] == "gemma3:4b"
