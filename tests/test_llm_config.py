"""Tests for lib_custom.llm_config module."""

import os
from unittest.mock import MagicMock, patch

from lib_custom.llm_config import (
    create_primary_llm,
    get_available_llms,
)
import pytest


def _make_fake_llm(**kwargs):
    """Create a fake LLM object that records constructor args."""
    fake = MagicMock()
    fake.model = kwargs.get("model", "")
    fake.base_url = kwargs.get("base_url", "")
    fake.api_key = kwargs.get("api_key", "")
    return fake


# ---------------------------------------------------------------------------
# create_primary_llm
# ---------------------------------------------------------------------------


class TestCreatePrimaryLlm:
    """Tests for create_primary_llm."""

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_MODEL_NAME": "gpt-4o",
        "OPENAI_BASE_URL": "http://localhost:8045/v1",
    })
    def test_returns_llm_with_all_env_vars(self):
        llm = create_primary_llm()
        assert llm is not None

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_MODEL_NAME": "gpt-4o",
    }, clear=True)
    def test_works_without_base_url(self):
        llm = create_primary_llm()
        assert llm is not None

    @patch.dict(os.environ, {
        "OPENAI_MODEL_NAME": "gpt-4o",
    }, clear=True)
    def test_raises_when_api_key_missing(self):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            create_primary_llm()

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
    }, clear=True)
    def test_raises_when_model_name_missing(self):
        with pytest.raises(ValueError, match="OPENAI_MODEL_NAME"):
            create_primary_llm()


# ---------------------------------------------------------------------------
# get_available_llms
# ---------------------------------------------------------------------------


class TestGetAvailableLlms:
    """Tests for get_available_llms."""

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_MODEL_NAME": "gpt-4o",
    }, clear=True)
    def test_returns_only_primary_when_configured(self):
        llms = get_available_llms()
        labels = [label for label, _ in llms]
        assert labels == ["主模型"]

    @patch.dict(os.environ, {}, clear=True)
    def test_returns_empty_when_nothing_configured(self):
        llms = get_available_llms()
        assert llms == []
