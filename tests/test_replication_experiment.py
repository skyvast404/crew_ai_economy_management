"""Tests for lib_custom/replication_experiment.py."""

import json
import os
import tempfile

import pytest

from lib_custom.replication_experiment import (
    ExperimentDesignMatrix,
    RunRecord,
    average_evaluator_scores,
    build_design_matrix,
    export_to_csv,
    export_to_json,
    load_checkpoint,
    run_replication,
)


# ---------------------------------------------------------------------------
# build_design_matrix
# ---------------------------------------------------------------------------
class TestBuildDesignMatrix:
    """Tests for build_design_matrix()."""

    def test_produces_72_runs(self):
        matrix = build_design_matrix(n_per_stratum=8, seed=42)
        assert len(matrix.runs) == 72

    def test_total_runs_field(self):
        matrix = build_design_matrix(n_per_stratum=8, seed=42)
        assert matrix.total_runs == 72

    def test_nine_cells(self):
        matrix = build_design_matrix(n_per_stratum=8, seed=42)
        cells = {r.cell_label for r in matrix.runs}
        assert len(cells) == 9

    def test_three_diversity_strata(self):
        matrix = build_design_matrix(n_per_stratum=8, seed=42)
        strata = {r.diversity_stratum for r in matrix.runs}
        assert strata == {"low", "medium", "high"}

    def test_three_ttl_levels(self):
        matrix = build_design_matrix(n_per_stratum=8, seed=42)
        levels = {r.ttl_level for r in matrix.runs}
        assert levels == {"low", "medium", "high"}

    def test_ttl_codes_correct(self):
        matrix = build_design_matrix(n_per_stratum=8, seed=42)
        for run in matrix.runs:
            if run.ttl_level == "low":
                assert run.ttl_code == 0.0
            elif run.ttl_level == "medium":
                assert run.ttl_code == 0.5
            elif run.ttl_level == "high":
                assert run.ttl_code == 1.0

    def test_team_size_five(self):
        matrix = build_design_matrix(team_size=5, n_per_stratum=2, seed=42)
        for run in matrix.runs:
            assert len(run.type_ids) == 5

    def test_run_ids_unique(self):
        matrix = build_design_matrix(n_per_stratum=8, seed=42)
        ids = [r.run_id for r in matrix.runs]
        assert len(ids) == len(set(ids))

    def test_all_pending(self):
        matrix = build_design_matrix(n_per_stratum=4, seed=42)
        for run in matrix.runs:
            assert run.status == "pending"

    def test_smaller_stratum(self):
        matrix = build_design_matrix(n_per_stratum=2, seed=42)
        # 3 strata × 3 TTL × 2 reps = 18
        assert len(matrix.runs) == 18

    def test_deterministic(self):
        m1 = build_design_matrix(n_per_stratum=4, seed=99)
        m2 = build_design_matrix(n_per_stratum=4, seed=99)
        for r1, r2 in zip(m1.runs, m2.runs):
            assert r1.composition_id == r2.composition_id
            assert r1.type_ids == r2.type_ids


# ---------------------------------------------------------------------------
# average_evaluator_scores
# ---------------------------------------------------------------------------
class TestAverageScores:
    """Tests for average_evaluator_scores()."""

    def test_basic_average(self):
        a = {"task_completion": 80.0, "collaboration": 70.0}
        b = {"task_completion": 90.0, "collaboration": 60.0}
        result = average_evaluator_scores(a, b)
        assert result["task_completion"] == 85.0
        assert result["collaboration"] == 65.0

    def test_missing_dimension(self):
        a = {"task_completion": 80.0}
        b = {"task_completion": 90.0, "innovation": 60.0}
        result = average_evaluator_scores(a, b)
        assert result["task_completion"] == 85.0
        # innovation: (0 + 60) / 2 = 30
        assert result["innovation"] == 30.0

    def test_empty_inputs(self):
        result = average_evaluator_scores({}, {})
        assert result == {}


# ---------------------------------------------------------------------------
# CSV / JSON export
# ---------------------------------------------------------------------------
class TestExport:
    """Tests for export functions."""

    def test_csv_export(self):
        matrix = build_design_matrix(n_per_stratum=1, seed=42)
        # Mark first run as completed with dummy scores
        matrix.runs[0] = matrix.runs[0].model_copy(
            update={
                "status": "completed",
                "scores_mean": {"task_completion": 75.0},
                "overall_mean": 75.0,
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            export_to_csv(matrix, path)
            assert os.path.exists(path)
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
            assert "run_id" in content
            assert "75" in content
        finally:
            os.unlink(path)

    def test_json_export_and_load(self):
        matrix = build_design_matrix(n_per_stratum=1, seed=42)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            export_to_json(matrix, path)
            loaded = load_checkpoint(path)
            assert loaded is not None
            assert len(loaded.runs) == len(matrix.runs)
            assert loaded.topic == matrix.topic
        finally:
            os.unlink(path)

    def test_load_missing_checkpoint(self):
        result = load_checkpoint("/nonexistent/path.json")
        assert result is None


# ---------------------------------------------------------------------------
# run_replication
# ---------------------------------------------------------------------------
class TestRunReplication:
    """Tests for run_replication()."""

    def test_skips_completed_runs(self):
        matrix = build_design_matrix(n_per_stratum=1, seed=42)
        # Mark all as completed
        for i, run in enumerate(matrix.runs):
            matrix.runs[i] = run.model_copy(update={"status": "completed"})

        call_count = 0

        def mock_fn(record, design):
            nonlocal call_count
            call_count += 1
            return record

        run_replication(matrix, mock_fn)
        assert call_count == 0  # nothing to run

    def test_runs_pending(self):
        matrix = build_design_matrix(n_per_stratum=1, seed=42)
        run_count = 0

        def mock_fn(record, design):
            nonlocal run_count
            run_count += 1
            return record.model_copy(update={"status": "completed"})

        result = run_replication(matrix, mock_fn)
        assert run_count == len(matrix.runs)
        assert result.completed_runs == len(matrix.runs)

    def test_handles_failure(self):
        matrix = build_design_matrix(n_per_stratum=1, seed=42)

        def fail_fn(record, design):
            raise RuntimeError("test failure")

        result = run_replication(matrix, fail_fn)
        # Should not raise, but mark as failed
        for run in result.runs:
            assert run.status == "failed"
            assert "test failure" in run.error

    def test_progress_callback(self):
        matrix = build_design_matrix(n_per_stratum=1, seed=42)
        progress: list[tuple[int, int]] = []

        def mock_fn(record, design):
            return record.model_copy(update={"status": "completed"})

        def on_complete(record, completed, total):
            progress.append((completed, total))

        run_replication(matrix, mock_fn, on_run_complete=on_complete)
        assert len(progress) == len(matrix.runs)
