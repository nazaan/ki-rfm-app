"""
ai/llm_router.py
----------------
Unified LLM interface. Supports:
    - Anthropic (Claude)
    - OpenAI (GPT-4o)
    - Groq (fast, free tier)
    - Ollama (local, privacy-first)

Usage:
    router = LLMRouter(provider="anthropic", api_key="sk-...")
    response = router.complete(system_prompt, user_prompt)

All providers receive identical prompts.
Response is always a plain string.
"""

from typing import Optional
import os


# ── Provider constants ─────────────────────────────────────────────────────

PROVIDERS = {
    "anthropic": {
        "label":       "Anthropic (Claude)",
        "models":      ["claude-sonnet-4-20250514"],
        "default":     "claude-sonnet-4-20250514",
        "needs_key":   True,
        "key_label":   "Anthropic API Key",
        "key_prefix":  "sk-ant-",
        "note":        "Best quality. Get key at console.anthropic.com",
    },
    "openai": {
        "label":       "OpenAI (GPT-4o)",
        "models":      ["gpt-4o", "gpt-4o-mini"],
        "default":     "gpt-4o-mini",
        "needs_key":   True,
        "key_label":   "OpenAI API Key",
        "key_prefix":  "sk-",
        "note":        "Most users already have a key. platform.openai.com",
    },
    "groq": {
        "label":       "Groq (Fast + Free tier)",
        "models":      ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        "default":     "llama-3.3-70b-versatile",
        "needs_key":   True,
        "key_label":   "Groq API Key",
        "key_prefix":  "gsk_",
        "note":        "Free tier available. console.groq.com",
    },
    "ollama": {
        "label":       "Ollama (Local / Private)",
        "models":      ["llama3.2", "llama3.1", "mistral"],
        "default":     "llama3.2",
        "needs_key":   False,
        "key_label":   None,
        "key_prefix":  None,
        "note":        "Runs locally. Your data never leaves your machine. ollama.ai",
    },
}

QUALITY_NOTE = {
    "anthropic": None,
    "openai":    None,
    "groq":      "⚠️  Output quality may vary with smaller Groq models.",
    "ollama":    "⚠️  Output quality depends on your local model. Results may vary.",
}


# ── Main router class ──────────────────────────────────────────────────────

class LLMRouter:
    """
    Unified interface for all supported LLM providers.

    Args:
        provider:   one of "anthropic", "openai", "groq", "ollama"
        api_key:    API key string (not needed for ollama)
        model:      override the default model for the provider
        ollama_url: base URL for Ollama (default: http://localhost:11434)
    """

    def __init__(
        self,
        provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
    ):
        if provider not in PROVIDERS:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Choose from: {', '.join(PROVIDERS.keys())}"
            )

        self.provider    = provider
        self.api_key     = api_key or os.getenv(_env_key_name(provider), "")
        self.model       = model or PROVIDERS[provider]["default"]
        self.ollama_url  = ollama_url
        self._client     = None  # lazy init

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """
        Send a prompt to the configured provider.
        Returns the response as a plain string.
        Raises LLMError on failure.
        """
        try:
            if self.provider == "anthropic":
                return self._anthropic(system_prompt, user_prompt, max_tokens, temperature)
            elif self.provider == "openai":
                return self._openai(system_prompt, user_prompt, max_tokens, temperature)
            elif self.provider == "groq":
                return self._groq(system_prompt, user_prompt, max_tokens, temperature)
            elif self.provider == "ollama":
                return self._ollama(system_prompt, user_prompt, max_tokens, temperature)
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"[{self.provider}] Unexpected error: {e}") from e

    # ── Provider implementations ───────────────────────────────────────────

    def _anthropic(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        try:
            import anthropic
        except ImportError:
            raise LLMError(
                "anthropic package not installed. Run: pip install anthropic"
            )

        if not self.api_key:
            raise LLMError("Anthropic API key is missing.")

        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key)

        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return message.content[0].text
        except anthropic.AuthenticationError:
            raise LLMError("Invalid Anthropic API key. Please check your key.")
        except anthropic.RateLimitError:
            raise LLMError("Anthropic rate limit reached. Please wait and try again.")
        except anthropic.APIError as e:
            raise LLMError(f"Anthropic API error: {e}")

    def _openai(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        try:
            from openai import OpenAI, AuthenticationError, RateLimitError, APIError
        except ImportError:
            raise LLMError(
                "openai package not installed. Run: pip install openai"
            )

        if not self.api_key:
            raise LLMError("OpenAI API key is missing.")

        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            return response.choices[0].message.content
        except AuthenticationError:
            raise LLMError("Invalid OpenAI API key. Please check your key.")
        except RateLimitError:
            raise LLMError("OpenAI rate limit reached. Please wait and try again.")
        except APIError as e:
            raise LLMError(f"OpenAI API error: {e}")

    def _groq(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        try:
            from groq import Groq, AuthenticationError, RateLimitError
        except ImportError:
            raise LLMError(
                "groq package not installed. Run: pip install groq"
            )

        if not self.api_key:
            raise LLMError("Groq API key is missing.")

        if self._client is None:
            self._client = Groq(api_key=self.api_key)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            return response.choices[0].message.content
        except AuthenticationError:
            raise LLMError("Invalid Groq API key. Please check your key.")
        except RateLimitError:
            raise LLMError("Groq rate limit reached. Please wait and try again.")
        except Exception as e:
            raise LLMError(f"Groq API error: {e}")

    def _ollama(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        try:
            import requests
        except ImportError:
            raise LLMError("requests package not installed. Run: pip install requests")

        url = f"{self.ollama_url}/api/chat"
        payload = {
            "model":    self.model,
            "stream":   False,
            "options":  {"temperature": temperature, "num_predict": max_tokens},
            "messages": [
                {"role": "system",  "content": system},
                {"role": "user",    "content": user},
            ],
        }

        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except requests.exceptions.ConnectionError:
            raise LLMError(
                f"Cannot connect to Ollama at {self.ollama_url}. "
                "Make sure Ollama is running locally."
            )
        except requests.exceptions.Timeout:
            raise LLMError("Ollama request timed out. The model may be loading.")
        except Exception as e:
            raise LLMError(f"Ollama error: {e}")

    # ── Validation helper ──────────────────────────────────────────────────

    def validate_key(self) -> tuple[bool, str]:
        """
        Quick check that the API key works without burning tokens.
        Returns (success: bool, message: str)
        Used by the UI settings panel.
        """
        try:
            self.complete(
                system_prompt="You are a test assistant.",
                user_prompt="Reply with only the word: OK",
                max_tokens=5,
                temperature=0,
            )
            return True, f"✅ Connected to {PROVIDERS[self.provider]['label']}"
        except LLMError as e:
            return False, f"❌ {e}"


# ── Custom exception ───────────────────────────────────────────────────────

class LLMError(Exception):
    """Raised for any LLM provider error. Always has a user-friendly message."""
    pass


# ── Helpers ────────────────────────────────────────────────────────────────

def _env_key_name(provider: str) -> str:
    """Returns the environment variable name for a provider's API key."""
    return {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai":    "OPENAI_API_KEY",
        "groq":      "GROQ_API_KEY",
        "ollama":    "",
    }.get(provider, "")


def get_provider_info(provider: str) -> dict:
    """Returns the metadata dict for a provider. Safe for UI display."""
    return PROVIDERS.get(provider, {})


def get_quality_note(provider: str) -> Optional[str]:
    """Returns a warning string if the provider may produce lower quality output."""
    return QUALITY_NOTE.get(provider)
