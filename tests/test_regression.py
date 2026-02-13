"""Tests for lib_custom/analysis/regression.py."""

import math

import numpy as np
import pytest

from lib_custom.analysis.regression import (
    ICCResult,
    OLSResult,
    RegressionData,
    SimpleSlopeResult,
    compute_icc,
    compute_simple_slopes,
    format_regression_table,
    prepare_regression_data,
    run_ols,
)


# ---------------------------------------------------------------------------
# prepare_regression_data
# ---------------------------------------------------------------------------
class TestPrepareRegressionData:
    """Tests for prepare_regression_data()."""

    def test_centering(self):
        data = prepare_regression_data(
            diversity_raw=[0.2, 0.4, 0.6],
            ttl_raw=[0.0, 0.5, 1.0],
            outcome_raw=[50.0, 60.0, 70.0],
        )
        # After centering, mean should be ~0
        assert abs(np.mean(data.diversity)) < 1e-10
        assert abs(np.mean(data.ttl)) < 1e-10

    def test_interaction_term(self):
        data = prepare_regression_data(
            diversity_raw=[0.2, 0.4, 0.6],
            ttl_raw=[0.0, 0.5, 1.0],
            outcome_raw=[50.0, 60.0, 70.0],
        )
        # interaction = diversity_centered * ttl_centered
        expected = data.diversity * data.ttl
        np.testing.assert_array_almost_equal(data.interaction, expected)

    def test_correct_n(self):
        data = prepare_regression_data(
            diversity_raw=[0.1, 0.2, 0.3, 0.4],
            ttl_raw=[0.0, 0.5, 0.5, 1.0],
            outcome_raw=[40.0, 50.0, 60.0, 70.0],
        )
        assert data.n == 4

    def test_unequal_lengths_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            prepare_regression_data(
                diversity_raw=[0.1, 0.2],
                ttl_raw=[0.0],
                outcome_raw=[50.0, 60.0],
            )

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            prepare_regression_data(
                diversity_raw=[0.1, 0.2],
                ttl_raw=[0.0, 0.5],
                outcome_raw=[50.0, 60.0],
            )

    def test_sd_stored(self):
        data = prepare_regression_data(
            diversity_raw=[0.2, 0.4, 0.6],
            ttl_raw=[0.0, 0.5, 1.0],
            outcome_raw=[50.0, 60.0, 70.0],
        )
        assert data.diversity_sd > 0
        assert data.ttl_sd > 0


# ---------------------------------------------------------------------------
# run_ols (synthetic data with known relationship)
# ---------------------------------------------------------------------------
class TestRunOLS:
    """Tests for run_ols()."""

    @pytest.fixture()
    def synthetic_data(self):
        """Create synthetic data with known coefficients.

        y = 60 + (-5)*div + 3*ttl + 2*div*ttl + noise
        """
        rng = np.random.RandomState(42)
        n = 200
        div = rng.uniform(0.1, 0.9, n)
        ttl = rng.choice([0.0, 0.5, 1.0], n)
        noise = rng.normal(0, 1, n)
        y = 60 + (-5) * div + 3 * ttl + 2 * div * ttl + noise
        return prepare_regression_data(
            diversity_raw=div.tolist(),
            ttl_raw=ttl.tolist(),
            outcome_raw=y.tolist(),
        )

    def test_returns_ols_result(self, synthetic_data):
        result = run_ols(synthetic_data)
        assert isinstance(result, OLSResult)

    def test_has_four_coefficients(self, synthetic_data):
        result = run_ols(synthetic_data)
        assert "intercept" in result.coefficients
        assert "diversity" in result.coefficients
        assert "ttl" in result.coefficients
        assert "interaction" in result.coefficients

    def test_r_squared_range(self, synthetic_data):
        result = run_ols(synthetic_data)
        assert 0.0 <= result.r_squared <= 1.0

    def test_diversity_coefficient_negative(self, synthetic_data):
        result = run_ols(synthetic_data)
        # True β₁ = -5, should be negative
        assert result.coefficients["diversity"] < 0

    def test_ttl_coefficient_positive(self, synthetic_data):
        result = run_ols(synthetic_data)
        # True β₂ = 3, should be positive
        assert result.coefficients["ttl"] > 0

    def test_interaction_coefficient_positive(self, synthetic_data):
        result = run_ols(synthetic_data)
        # True β₃ = 2, should be positive
        assert result.coefficients["interaction"] > 0

    def test_n_correct(self, synthetic_data):
        result = run_ols(synthetic_data)
        assert result.n == 200

    def test_delta_r_squared_nonnegative(self, synthetic_data):
        result = run_ols(synthetic_data)
        assert result.delta_r_squared >= -0.01  # allow tiny float error

    def test_residuals_length(self, synthetic_data):
        result = run_ols(synthetic_data)
        assert len(result.residuals) == 200


# ---------------------------------------------------------------------------
# compute_simple_slopes
# ---------------------------------------------------------------------------
class TestSimpleSlopes:
    """Tests for compute_simple_slopes()."""

    def test_returns_three_slopes(self):
        rng = np.random.RandomState(42)
        n = 30
        div = rng.uniform(0.1, 0.9, n)
        ttl = rng.choice([0.0, 0.5, 1.0], n)
        y = 60 + (-5) * div + 3 * ttl + 2 * div * ttl + rng.normal(0, 1, n)
        data = prepare_regression_data(div.tolist(), ttl.tolist(), y.tolist())
        ols = run_ols(data)
        slopes = compute_simple_slopes(ols, data)
        assert len(slopes) == 3

    def test_slope_labels(self):
        rng = np.random.RandomState(42)
        n = 30
        div = rng.uniform(0.1, 0.9, n)
        ttl = rng.choice([0.0, 0.5, 1.0], n)
        y = 60 - 5 * div + 3 * ttl + rng.normal(0, 1, n)
        data = prepare_regression_data(div.tolist(), ttl.tolist(), y.tolist())
        ols = run_ols(data)
        slopes = compute_simple_slopes(ols, data)
        labels = [s.ttl_label for s in slopes]
        assert "-1SD" in labels
        assert "Mean" in labels
        assert "+1SD" in labels


# ---------------------------------------------------------------------------
# compute_icc
# ---------------------------------------------------------------------------
class TestComputeICC:
    """Tests for compute_icc()."""

    def test_perfect_agreement(self):
        scores = [70.0, 80.0, 90.0, 60.0, 75.0]
        result = compute_icc(scores, scores, "test")
        assert result.icc == 1.0
        assert result.interpretation == "excellent"

    def test_no_agreement(self):
        a = [90.0, 10.0, 90.0, 10.0]
        b = [10.0, 90.0, 10.0, 90.0]
        result = compute_icc(a, b, "test")
        assert result.icc < 0.0  # negatively correlated

    def test_moderate_agreement(self):
        a = [70.0, 80.0, 60.0, 90.0, 50.0]
        b = [68.0, 82.0, 58.0, 88.0, 52.0]
        result = compute_icc(a, b, "test")
        assert result.icc > 0.5

    def test_unequal_lengths_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            compute_icc([1.0, 2.0], [1.0], "test")

    def test_single_observation(self):
        result = compute_icc([50.0], [55.0], "test")
        assert result.n == 1

    def test_interpretation_categories(self):
        # Test interpretation thresholds
        result_excellent = compute_icc(
            [50.0, 60.0, 70.0, 80.0, 90.0],
            [50.0, 60.0, 70.0, 80.0, 90.0],
            "test",
        )
        assert result_excellent.interpretation == "excellent"


# ---------------------------------------------------------------------------
# format_regression_table
# ---------------------------------------------------------------------------
class TestFormatRegressionTable:
    """Tests for format_regression_table()."""

    def test_contains_key_sections(self):
        ols = OLSResult(
            coefficients={"intercept": 60.0, "diversity": -5.0, "ttl": 3.0, "interaction": 2.0},
            std_errors={"intercept": 1.0, "diversity": 0.5, "ttl": 0.5, "interaction": 0.3},
            t_values={"intercept": 60.0, "diversity": -10.0, "ttl": 6.0, "interaction": 6.67},
            p_values={"intercept": 0.0, "diversity": 0.001, "ttl": 0.01, "interaction": 0.005},
            r_squared=0.75,
            adj_r_squared=0.73,
            f_statistic=45.0,
            f_p_value=0.0001,
            n=72,
            residuals=np.zeros(72),
            delta_r_squared=0.05,
        )
        table = format_regression_table(ols)
        assert "R²" in table
        assert "diversity" in table
        assert "interaction" in table
        assert "OLS Regression" in table

    def test_significance_stars(self):
        ols = OLSResult(
            coefficients={"intercept": 60.0, "diversity": -5.0, "ttl": 3.0, "interaction": 2.0},
            std_errors={"intercept": 1.0, "diversity": 0.5, "ttl": 0.5, "interaction": 0.3},
            t_values={"intercept": 60.0, "diversity": -10.0, "ttl": 6.0, "interaction": 6.67},
            p_values={"intercept": 0.0, "diversity": 0.0001, "ttl": 0.005, "interaction": 0.03},
            r_squared=0.75,
            adj_r_squared=0.73,
            f_statistic=45.0,
            f_p_value=0.0001,
            n=72,
            residuals=np.zeros(72),
            delta_r_squared=0.05,
        )
        table = format_regression_table(ols)
        assert "***" in table  # diversity p < 0.001
        assert "**" in table   # ttl p < 0.01
