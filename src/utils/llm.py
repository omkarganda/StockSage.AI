from __future__ import annotations

"""LLM Utilities for StockSage.AI

This module provides a minimal wrapper around the OpenAI ChatCompletion API so that
other parts of the codebase can easily leverage large language models for tasks such
as sentiment scoring, headline/event summarisation, and scenario generation.

Design goals:
1. Keep dependencies optional – the rest of the system must work even if the `openai`
   package is absent or no API key is configured.
2. Provide safe fall-backs that return neutral or empty outputs so the calling code
   does not break in offline / CI environments.
3. Encapsulate all LLM prompts here so it is straightforward to modify or swap the
   underlying provider (e.g. Azure OpenAI, Anthropic, Llama-CPP) in the future.
"""

from typing import List, Dict, Any

from ..config import APIConfig
from .logging import get_logger

logger = get_logger(__name__)

# Optional import – only required when OPENAI_API_KEY is set
try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover – handled gracefully below
    openai = None  # type: ignore


class LLMNotAvailableError(RuntimeError):
    """Raised when an LLM feature is invoked but no provider is configured."""


class GPTClient:
    """Minimal OpenAI GPT wrapper used across the project.

    Example
    -------
    >>> client = GPTClient()
    >>> client.chat([{"role": "user", "content": "Hello!"}])
    'Hi there! How can I help you today?'
    """

    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.2):
        if not APIConfig.OPENAI_API_KEY:
            raise LLMNotAvailableError(
                "OPENAI_API_KEY is not set – cannot initialise GPTClient."
            )
        if openai is None:  # pragma: no cover
            raise LLMNotAvailableError(
                "The `openai` package is not installed. Add it to your environment to use LLM features."
            )

        openai.api_key = APIConfig.OPENAI_API_KEY
        self._client = openai
        self.model = model
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Low-level interface
    # ------------------------------------------------------------------
    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Send a ChatCompletion request and return the assistant message text."""
        try:
            response = self._client.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", 512),
            )
            return response.choices[0].message["content"].strip()
        except Exception as exc:  # pragma: no cover – network errors etc.
            logger.error("OpenAI API call failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------
    def score_sentiment(self, text: str) -> float:
        """Return a sentiment score in the range [-1, 1] for the provided text.

        Uses a very short prompt so it is cheap. The assistant **MUST** respond
        with a JSON payload like: `{ "score": 0.42 }`.
        """
        prompt = (
            "You are a financial news sentiment classifier. "
            "Return a single JSON object with key 'score' whose value is a number "
            "between -1 (very negative) and 1 (very positive) that captures the "
            "overall sentiment of the text. Do not return anything else."
        )
        completion = self.chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text[:4000]},  # keep under context limit
            ],
            max_tokens=20,
        )
        # Basic sanitisation – fall back to neutral if parsing fails
        try:
            import json

            data = json.loads(completion)
            score = float(data.get("score", 0.0))
            return max(min(score, 1.0), -1.0)
        except Exception:  # pragma: no cover
            logger.warning("Failed to parse LLM sentiment response – returning 0.0: %s", completion)
            return 0.0

    def summarise(self, texts: List[str], max_words: int = 60) -> str:
        """Return a short bullet-point summary of the supplied texts."""
        if not texts:
            return ""
        merged_text = "\n".join(texts)[:6000]
        prompt = (
            "Summarise the following financial news snippets into concise bullet points "
            f"(max {max_words} words total). Focus on key events and market implications."
        )
        summary = self.chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": merged_text},
            ],
            max_tokens=96,
        )
        return summary.strip()

    # ------------------------------------------------------------------
    # Narrative trend detection & scenario generation
    # ------------------------------------------------------------------
    def detect_trends(self, texts: List[str], max_trends: int = 5) -> List[str]:
        """Extract high-level narrative trends from a collection of news texts.

        Returns **up to** `max_trends` concise trend statements suitable for
        downstream feature engineering (e.g. bag-of-trends frequency counts).
        """
        if not texts:
            return []

        joined_text = "\n".join(texts)[:6000]
        prompt = (
            "You are a financial news analyst. Identify the key narrative trends "
            "in the following articles. Return a JSON list of at most "
            f"{max_trends} short trend headline strings."
        )
        completion = self.chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": joined_text},
            ],
            max_tokens=128,
        )
        try:
            import json

            trends = json.loads(completion)
            if isinstance(trends, list):
                return [str(t).strip() for t in trends][:max_trends]
        except Exception:
            logger.warning("Failed to parse trend response – %s", completion)
        return []

    def generate_scenarios(self, symbol: str, horizon_days: int = 30, n_scenarios: int = 3) -> List[str]:
        """Generate what-if market scenarios for a given symbol.

        Each scenario should be a *single sentence* describing a plausible
        market development within the forecast horizon. Returned as a list of
        strings.
        """
        prompt = (
            "You are a seasoned equity strategist. Generate "
            f"{n_scenarios} plausible but diverse market scenarios for the stock "
            f"{symbol} over the next {horizon_days} days. Each scenario must be a "
            "single succinct sentence. Return the scenarios as a JSON array of strings."
        )
        completion = self.chat(
            [{"role": "system", "content": prompt}],
            max_tokens=128,
        )
        try:
            import json

            scenarios = json.loads(completion)
            if isinstance(scenarios, list):
                return [str(s).strip() for s in scenarios][:n_scenarios]
        except Exception:
            logger.warning("Failed to parse scenarios response – %s", completion)
        return []

# ----------------------------------------------------------------------
# Safe façade wrappers
# ----------------------------------------------------------------------

def safe_score_sentiment(text: str) -> float:
    """Return sentiment score or 0.0 if LLM not available."""
    try:
        client = GPTClient()
        return client.score_sentiment(text)
    except LLMNotAvailableError:
        return 0.0


def safe_summarise_texts(texts: List[str]) -> str:
    """Return summary or empty string if LLM not available."""
    try:
        client = GPTClient()
        return client.summarise(texts)
    except LLMNotAvailableError:
        return ""


def safe_detect_trends(texts: List[str]) -> List[str]:
    try:
        client = GPTClient()
        return client.detect_trends(texts)
    except LLMNotAvailableError:
        return []


def safe_generate_scenarios(symbol: str, horizon_days: int = 30) -> List[str]:
    try:
        client = GPTClient()
        return client.generate_scenarios(symbol, horizon_days)
    except LLMNotAvailableError:
        return []