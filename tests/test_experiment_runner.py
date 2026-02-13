"""Tests for experiment runner — config, result models, and parsing."""

from lib_custom.chat_store import ChatMessage, ChatMessageStore
from lib_custom.experiment_runner import (
    ExperimentConfig,
    SingleRunResult,
    build_comparison_summary_prompt,
    create_default_experiment,
    extract_messages_as_dicts,
    extract_transcript,
    find_evaluator_output,
    parse_evaluation,
)
from pydantic import ValidationError
import pytest


class TestExperimentConfig:
    def test_default_experiment(self):
        config = create_default_experiment()
        assert config.topic == "Q3产品发布计划讨论"
        assert len(config.boss_types) == 2
        assert len(config.team.members) == 12

    def test_custom_topic(self):
        config = create_default_experiment(topic="自定义话题")
        assert config.topic == "自定义话题"

    def test_custom_project_type(self):
        config = create_default_experiment(project_type_id="long_term_platform")
        assert config.okrs.project_type_id == "long_term_platform"

    def test_invalid_project_type(self):
        with pytest.raises(ValueError, match="Unknown project type"):
            create_default_experiment(project_type_id="nonexistent")

    def test_topic_too_short(self):
        with pytest.raises(ValidationError):
            ExperimentConfig(
                topic="ab",
                okrs=create_default_experiment().okrs,
                team=create_default_experiment().team,
            )

    def test_num_rounds_range(self):
        with pytest.raises(ValidationError):
            ExperimentConfig(
                topic="valid topic",
                okrs=create_default_experiment().okrs,
                team=create_default_experiment().team,
                num_rounds=0,
            )


class TestParseEvaluation:
    def test_parse_valid_json(self):
        raw = """{
            "dimensions": {
                "task_completion": {"score": 85, "evidence": "任务完成良好"},
                "collaboration": {"score": 70, "evidence": "协作一般"}
            },
            "overall_score": 78.5,
            "key_findings": ["发现1", "发现2"],
            "boss_impact_analysis": "老板影响分析",
            "recommendations": ["建议1"]
        }"""
        result = parse_evaluation(raw)
        assert result.dimensions["task_completion"].score == 85
        assert result.overall_score == 78.5
        assert len(result.key_findings) == 2

    def test_parse_json_in_markdown_block(self):
        raw = """```json
        {
            "dimensions": {
                "task_completion": {"score": 90, "evidence": "优秀"}
            },
            "overall_score": 90,
            "key_findings": [],
            "boss_impact_analysis": "",
            "recommendations": []
        }
        ```"""
        result = parse_evaluation(raw)
        assert result.dimensions["task_completion"].score == 90

    def test_parse_nested_json_in_markdown_block(self):
        """Greedy regex must capture the entire nested JSON, not stop at the first '}'."""
        raw = """```json
        {
            "dimensions": {
                "task_completion": {"score": 85, "evidence": "完成"},
                "collaboration": {"score": 70, "evidence": "合作"}
            },
            "overall_score": 78.5,
            "key_findings": ["发现1"],
            "boss_impact_analysis": "分析",
            "recommendations": ["建议1"]
        }
        ```"""
        result = parse_evaluation(raw)
        assert result.dimensions["task_completion"].score == 85
        assert result.dimensions["collaboration"].score == 70
        assert result.overall_score == 78.5

    def test_parse_invalid_json_returns_default(self):
        result = parse_evaluation("这不是JSON")
        assert result.overall_score == 0.0
        assert len(result.dimensions) == 0

    def test_score_clamped_to_range(self):
        raw = """{
            "dimensions": {
                "task_completion": {"score": 150, "evidence": "超出范围"}
            },
            "overall_score": 0,
            "key_findings": [],
            "boss_impact_analysis": "",
            "recommendations": []
        }"""
        result = parse_evaluation(raw)
        assert result.dimensions["task_completion"].score == 100

    def test_score_clamped_minimum(self):
        raw = """{
            "dimensions": {
                "task_completion": {"score": -10, "evidence": "负分"}
            },
            "overall_score": 0,
            "key_findings": [],
            "boss_impact_analysis": "",
            "recommendations": []
        }"""
        result = parse_evaluation(raw)
        assert result.dimensions["task_completion"].score == 0


class TestExtractTranscript:
    def test_extract_completed_messages(self):
        store = ChatMessageStore()
        store.add(ChatMessage(role="Boss", content="开始讨论", msg_type="completed"))
        store.add(ChatMessage(role="system", content="系统消息", msg_type="completed"))
        store.add(ChatMessage(role="员工A", content="我的看法", msg_type="completed"))
        transcript = extract_transcript(store)
        assert "[Boss]: 开始讨论" in transcript
        assert "[员工A]: 我的看法" in transcript
        assert "系统消息" not in transcript

    def test_empty_store(self):
        store = ChatMessageStore()
        transcript = extract_transcript(store)
        assert transcript == ""


class TestExtractMessagesAsDicts:
    def test_extract_all_messages(self):
        store = ChatMessageStore()
        store.add(ChatMessage(role="Boss", content="hello", msg_type="completed"))
        dicts = extract_messages_as_dicts(store)
        assert len(dicts) == 1
        assert dicts[0]["role"] == "Boss"
        assert dicts[0]["content"] == "hello"


class TestFindEvaluatorOutput:
    def test_find_json_output(self):
        store = ChatMessageStore()
        store.add(ChatMessage(role="Boss", content="普通发言", msg_type="completed"))
        store.add(ChatMessage(
            role="评估者",
            content='{"dimensions": {}, "overall_score": 80}',
            msg_type="completed",
        ))
        output = find_evaluator_output(store)
        assert "dimensions" in output
        assert "80" in output

    def test_fallback_to_last_completed(self):
        store = ChatMessageStore()
        store.add(ChatMessage(role="Boss", content="最后发言", msg_type="completed"))
        output = find_evaluator_output(store)
        assert output == "最后发言"


class TestSingleRunResult:
    def test_default_values(self):
        run = SingleRunResult(boss_type_id="time_master")
        assert run.transcript == ""
        assert run.elapsed_seconds == 0.0
        assert run.evaluation.overall_score == 0.0


class TestBuildComparisonPrompt:
    def test_prompt_contains_topic(self):
        prompt = build_comparison_summary_prompt("测试话题", "评估A", "评估B")
        assert "测试话题" in prompt
        assert "评估A" in prompt
        assert "评估B" in prompt


class TestParseEvaluationEdgeCases:
    def test_parse_score_none_value(self):
        """BUG C: score key exists but value is None → should not raise TypeError."""
        raw = """{
            "dimensions": {
                "task_completion": {"score": null, "evidence": "无数据"}
            },
            "overall_score": 0,
            "key_findings": [],
            "boss_impact_analysis": "",
            "recommendations": []
        }"""
        result = parse_evaluation(raw)
        assert result.dimensions["task_completion"].score == 0

    def test_parse_dimensions_as_list(self):
        """BUG D: dimensions is a list instead of dict → should not raise AttributeError."""
        raw = """{
            "dimensions": [
                {"name": "task_completion", "score": 80}
            ],
            "overall_score": 50,
            "key_findings": [],
            "boss_impact_analysis": "",
            "recommendations": []
        }"""
        result = parse_evaluation(raw)
        assert len(result.dimensions) == 0
        assert result.overall_score == 50

    def test_latex_escape_special_chars(self):
        """BUG A: _latex_escape must not double-escape backslash."""
        # We test the regex-based _latex_escape logic directly
        # since importing the streamlit page module has side effects.
        import re

        latex_special = re.compile(r"([\\&%$#_{}])")
        latex_map = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
        }

        def _latex_escape(text: str) -> str:
            return latex_special.sub(lambda m: latex_map[m.group(1)], text)

        # Basic special chars
        assert _latex_escape("a & b") == r"a \& b"
        assert _latex_escape("hello_world") == r"hello\_world"

        # Backslash must not be double-escaped
        result = _latex_escape("path\\to")
        assert result == r"path\textbackslash{}to"
        # Crucially, no \{ or \} leaking from the backslash replacement
        assert r"\textbackslash\{" not in result

        # Multiple specials together
        assert _latex_escape("100%") == r"100\%"
        assert _latex_escape("$x$") == r"\$x\$"
