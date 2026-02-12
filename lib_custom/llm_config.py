"""LLM configuration for OpenRouter.

Provides factory functions for creating the primary LLM.
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


def get_available_llms() -> list[tuple[str, LLM]]:
    """Return a list of (label, LLM) for all configured LLMs.

    Returns the primary LLM if configured, otherwise an empty list.
    """
    llms: list[tuple[str, LLM]] = []

    try:
        llms.append(("主模型", create_primary_llm()))
    except ValueError as e:
        logger.warning("Primary LLM not available: %s", e)

    return llms
