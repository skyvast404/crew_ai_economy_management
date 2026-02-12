"""Tests for lib_custom.llm_config module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from lib_custom.llm_config import (
    create_openrouter_llm,
    create_primary_llm,
    get_available_llms,
)


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
# create_openrouter_llm (mock LLM to avoid LiteLLM dependency)
# ---------------------------------------------------------------------------


class TestCreateOpenrouterLlm:
    """Tests for create_openrouter_llm."""

    @patch("lib_custom.llm_config.LLM", side_effect=_make_fake_llm)
    @patch.dict(os.environ, {
        "OPENROUTER_API_KEY": "or-test-key",
        "OPENROUTER_MODEL_NAME": "openai/gpt-4o",
    })
    def test_returns_llm_when_configured(self, mock_llm):
        llm = create_openrouter_llm()
        assert llm is not None
        mock_llm.assert_called_once_with(
            model="openrouter/openai/gpt-4o",
            base_url="https://openrouter.ai/api/v1",
            api_key="or-test-key",
        )

    @patch("lib_custom.llm_config.LLM", side_effect=ImportError("Fallback to LiteLLM is not available"))
    @patch.dict(os.environ, {
        "OPENROUTER_API_KEY": "or-test-key",
        "OPENROUTER_MODEL_NAME": "openai/gpt-4o",
    })
    def test_returns_none_when_litellm_fallback_unavailable(self, _mock_llm):
        result = create_openrouter_llm()
        assert result is None

    @patch.dict(os.environ, {}, clear=True)
    def test_returns_none_when_not_configured(self):
        result = create_openrouter_llm()
        assert result is None

    @patch.dict(os.environ, {
        "OPENROUTER_API_KEY": "or-test-key",
    }, clear=True)
    def test_returns_none_when_model_missing(self):
        result = create_openrouter_llm()
        assert result is None

    @patch.dict(os.environ, {
        "OPENROUTER_MODEL_NAME": "openai/gpt-4o",
    }, clear=True)
    def test_returns_none_when_key_missing(self):
        result = create_openrouter_llm()
        assert result is None

    @patch("lib_custom.llm_config.LLM", side_effect=_make_fake_llm)
    @patch.dict(os.environ, {
        "OPENROUTER_API_KEY": "or-test-key",
        "OPENROUTER_MODEL_NAME": "openrouter/openai/gpt-4o",
    })
    def test_does_not_double_prefix_openrouter(self, mock_llm):
        llm = create_openrouter_llm()
        assert llm is not None
        # Should pass through as-is, not add another "openrouter/" prefix
        mock_llm.assert_called_once_with(
            model="openrouter/openai/gpt-4o",
            base_url="https://openrouter.ai/api/v1",
            api_key="or-test-key",
        )


# ---------------------------------------------------------------------------
# get_available_llms
# ---------------------------------------------------------------------------


class TestGetAvailableLlms:
    """Tests for get_available_llms."""

    @patch("lib_custom.llm_config.LLM", side_effect=_make_fake_llm)
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_MODEL_NAME": "gpt-4o",
        "OPENROUTER_API_KEY": "or-key",
        "OPENROUTER_MODEL_NAME": "openai/gpt-4o",
    })
    def test_returns_both_when_both_configured(self, mock_llm):
        llms = get_available_llms()
        labels = [label for label, _ in llms]
        assert "主模型" in labels
        assert "OpenRouter" in labels
        assert len(llms) == 2

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_MODEL_NAME": "gpt-4o",
    }, clear=True)
    def test_returns_only_primary_when_openrouter_not_configured(self):
        llms = get_available_llms()
        labels = [label for label, _ in llms]
        assert labels == ["主模型"]

    @patch("lib_custom.llm_config.LLM", side_effect=_make_fake_llm)
    @patch.dict(os.environ, {
        "OPENROUTER_API_KEY": "or-key",
        "OPENROUTER_MODEL_NAME": "openai/gpt-4o",
    }, clear=True)
    def test_returns_only_openrouter_when_primary_missing(self, mock_llm):
        llms = get_available_llms()
        labels = [label for label, _ in llms]
        assert labels == ["OpenRouter"]

    @patch.dict(os.environ, {}, clear=True)
    def test_returns_empty_when_nothing_configured(self):
        llms = get_available_llms()
        assert llms == []
