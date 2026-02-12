"""LLM configuration with fallback support.

Provides factory functions for creating primary and OpenRouter fallback LLMs.
"""

import logging
import os

from crewai import LLM

logger = logging.getLogger(__name__)


def create_primary_llm() -> LLM:
    """Create the primary LLM from OPENAI_* environment variables.

    Returns:
        LLM configured with OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL_NAME.

    Raises:
        ValueError: If required env vars are missing.
    """
    base_url = os.getenv("OPENAI_BASE_URL", "")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model_name = os.getenv("OPENAI_MODEL_NAME", "")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    if not model_name:
        raise ValueError("OPENAI_MODEL_NAME is not set")

    kwargs: dict = {
        "model": model_name,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url

    logger.info("Primary LLM: model=%s base_url=%s", model_name, base_url or "(default)")
    return LLM(**kwargs)


def create_openrouter_llm() -> LLM | None:
    """Create an OpenRouter fallback LLM if configured.

    Reads OPENROUTER_API_KEY and OPENROUTER_MODEL_NAME from env.

    Returns:
        LLM instance or None if not configured.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    model_name = os.getenv("OPENROUTER_MODEL_NAME", "")

    if not api_key or not model_name:
        logger.info("OpenRouter fallback not configured (missing OPENROUTER_API_KEY or OPENROUTER_MODEL_NAME)")
        return None

    full_model = f"openrouter/{model_name}" if not model_name.startswith("openrouter/") else model_name

    logger.info("OpenRouter fallback LLM: model=%s", full_model)
    return LLM(
        model=full_model,
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def get_available_llms() -> list[tuple[str, LLM]]:
    """Return a list of (label, LLM) for all configured LLMs.

    The primary LLM is always first. OpenRouter is appended if configured.
    Entries with missing configuration are skipped.
    """
    llms: list[tuple[str, LLM]] = []

    try:
        llms.append(("主模型", create_primary_llm()))
    except ValueError as e:
        logger.warning("Primary LLM not available: %s", e)

    openrouter = create_openrouter_llm()
    if openrouter is not None:
        llms.append(("OpenRouter", openrouter))

    return llms
