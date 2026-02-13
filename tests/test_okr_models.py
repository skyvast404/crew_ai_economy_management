"""Tests for OKR data models, evaluation dimensions, and default OKR templates."""

from lib_custom.okr_models import (
    DEFAULT_OKRS,
    EVALUATION_DIMENSIONS,
    KeyResult,
    OKRSet,
    format_okrs_for_prompt,
    get_dimension_weights,
    get_okr_for_project,
    validate_dimensions_weights,
)
from pydantic import ValidationError
import pytest


class TestKeyResult:
    def test_valid_key_result(self):
        kr = KeyResult(
            id="kr1",
            description="完成率",
            target="≥95%",
            weight=0.5,
            unit="%",
        )
        assert kr.id == "kr1"
        assert kr.weight == 0.5

    def test_weight_must_be_positive(self):
        with pytest.raises(ValidationError):
            KeyResult(id="kr1", description="test", target="100", weight=0.0)

    def test_weight_must_not_exceed_one(self):
        with pytest.raises(ValidationError):
            KeyResult(id="kr1", description="test", target="100", weight=1.5)

    def test_empty_id_rejected(self):
        with pytest.raises(ValidationError):
            KeyResult(id="", description="test", target="100", weight=0.5)


class TestOKRSet:
    def test_valid_okr_set(self):
        okr = OKRSet(
            objective="这是一个测试目标",
            project_type_id="urgent_launch",
            key_results=[
                KeyResult(id="a", description="A指标测试", target="100", weight=0.6),
                KeyResult(id="b", description="B指标测试", target="50", weight=0.4),
            ],
        )
        assert okr.objective == "这是一个测试目标"
        assert len(okr.key_results) == 2

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValidationError, match=r"weights must sum to 1\.0"):
            OKRSet(
                objective="这是一个测试目标",
                project_type_id="test",
                key_results=[
                    KeyResult(id="a", description="指标A测试", target="1", weight=0.3),
                    KeyResult(id="b", description="指标B测试", target="2", weight=0.3),
                ],
            )

    def test_empty_key_results_rejected(self):
        with pytest.raises(ValidationError):
            OKRSet(
                objective="这是一个测试目标",
                project_type_id="test",
                key_results=[],
            )


class TestEvaluationDimensions:
    def test_eight_dimensions_defined(self):
        assert len(EVALUATION_DIMENSIONS) == 8

    def test_weights_sum_to_one(self):
        assert validate_dimensions_weights()

    def test_all_dimensions_have_required_fields(self):
        for dim_id, dim in EVALUATION_DIMENSIONS.items():
            assert dim.id == dim_id
            assert len(dim.name_zh) > 0
            assert len(dim.description) >= 5
            assert 0 < dim.weight <= 1.0

    def test_get_dimension_weights(self):
        weights = get_dimension_weights()
        assert len(weights) == 8
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_expected_dimension_ids(self):
        expected = {
            "task_completion",
            "collaboration",
            "decision_quality",
            "innovation",
            "morale",
            "communication",
            "risk_management",
            "goal_alignment",
        }
        assert set(EVALUATION_DIMENSIONS.keys()) == expected


class TestDefaultOKRs:
    def test_three_templates_defined(self):
        assert len(DEFAULT_OKRS) == 3

    def test_project_types_covered(self):
        expected_types = {"urgent_launch", "long_term_platform", "exploratory_prototype"}
        assert set(DEFAULT_OKRS.keys()) == expected_types

    def test_each_okr_has_valid_weights(self):
        for project_type, okr in DEFAULT_OKRS.items():
            total = sum(kr.weight for kr in okr.key_results)
            assert abs(total - 1.0) < 0.01, (
                f"{project_type} KR weights sum to {total}"
            )

    def test_each_okr_has_key_results(self):
        for project_type, okr in DEFAULT_OKRS.items():
            assert len(okr.key_results) >= 3, (
                f"{project_type} should have ≥3 KRs"
            )

    def test_project_type_ids_match(self):
        for project_type, okr in DEFAULT_OKRS.items():
            assert okr.project_type_id == project_type

    def test_get_okr_for_project(self):
        okr = get_okr_for_project("urgent_launch")
        assert okr is not None
        assert okr.project_type_id == "urgent_launch"

    def test_get_okr_for_unknown_project(self):
        assert get_okr_for_project("nonexistent") is None

    def test_unique_kr_ids_within_each_okr(self):
        for project_type, okr in DEFAULT_OKRS.items():
            kr_ids = [kr.id for kr in okr.key_results]
            assert len(kr_ids) == len(set(kr_ids)), (
                f"{project_type} has duplicate KR ids"
            )


class TestFormatOkrsForPrompt:
    def test_format_contains_objective(self):
        okr = DEFAULT_OKRS["urgent_launch"]
        text = format_okrs_for_prompt(okr)
        assert okr.objective in text

    def test_format_contains_key_results(self):
        okr = DEFAULT_OKRS["urgent_launch"]
        text = format_okrs_for_prompt(okr)
        for kr in okr.key_results:
            assert kr.description in text

    def test_format_contains_weights(self):
        okr = DEFAULT_OKRS["urgent_launch"]
        text = format_okrs_for_prompt(okr)
        assert "权重" in text
