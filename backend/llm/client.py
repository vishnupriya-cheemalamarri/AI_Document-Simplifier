"""Shared OpenAI client (gpt-4o-mini, see docs/ARCHITECTURE.md > Tech Stack)."""

from openai import OpenAI

from .. import config

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file (see .env.example)."
            )
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client
