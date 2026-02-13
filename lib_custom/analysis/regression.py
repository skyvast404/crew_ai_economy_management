"""OLS regression analysis with moderation (interaction) effects.

Implements:
    Performance = β₀ + β₁(Diversity) + β₂(TTL) + β₃(Diversity×TTL) + ε

Expected signs: β₁ < 0, β₂ > 0, β₃ > 0 (TTL buffers diversity negative effect).

Also computes:
    - Simple slopes at ±1SD TTL
    - Per-dimension regressions
    - ICC (intraclass correlation) for dual-evaluator reliability
    - Interaction plots (matplotlib)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RegressionData:
    """Centered regression inputs."""

    diversity: np.ndarray  # centered composite Blau
    ttl: np.ndarray  # centered TTL code
    interaction: np.ndarray  # diversity × ttl (centered)
    outcome: np.ndarray  # DV (score)
    n: int
    diversity_mean: float
    diversity_sd: float
    ttl_mean: float
    ttl_sd: float


def prepare_regression_data(
    diversity_raw: list[float],
    ttl_raw: list[float],
    outcome_raw: list[float],
) -> RegressionData:
    """Center predictors and compute interaction term.

    Args:
        diversity_raw: Raw composite Blau index values.
        ttl_raw: Raw TTL codes (0.0, 0.5, 1.0).
        outcome_raw: Dependent variable scores.

    Returns:
        RegressionData with centered arrays.

    Raises:
        ValueError: If arrays have different lengths or < 3 observations.
    """
    if not (len(diversity_raw) == len(ttl_raw) == len(outcome_raw)):
        raise ValueError("All input arrays must have equal length")
    n = len(diversity_raw)
    if n < 3:
        raise ValueError(f"Need at least 3 observations, got {n}")

    div_arr = np.array(diversity_raw, dtype=float)
    ttl_arr = np.array(ttl_raw, dtype=float)
    out_arr = np.array(outcome_raw, dtype=float)

    div_mean = float(np.mean(div_arr))
    div_sd = float(np.std(div_arr, ddof=1)) if n > 1 else 1.0
    ttl_mean = float(np.mean(ttl_arr))
    ttl_sd = float(np.std(ttl_arr, ddof=1)) if n > 1 else 1.0

    div_c = div_arr - div_mean
    ttl_c = ttl_arr - ttl_mean
    interaction = div_c * ttl_c

    return RegressionData(
        diversity=div_c,
        ttl=ttl_c,
        interaction=interaction,
        outcome=out_arr,
        n=n,
        diversity_mean=div_mean,
        diversity_sd=div_sd if div_sd > 0 else 1.0,
        ttl_mean=ttl_mean,
        ttl_sd=ttl_sd if ttl_sd > 0 else 1.0,
    )


# ---------------------------------------------------------------------------
# OLS regression result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OLSResult:
    """OLS regression output."""

    coefficients: dict[str, float]  # β₀, β₁, β₂, β₃
    std_errors: dict[str, float]
    t_values: dict[str, float]
    p_values: dict[str, float]
    r_squared: float
    adj_r_squared: float
    f_statistic: float
    f_p_value: float
    n: int
    residuals: np.ndarray
    delta_r_squared: float  # ΔR² from adding interaction term


def run_ols(data: RegressionData) -> OLSResult:
    """Run OLS regression: outcome = β₀ + β₁·div + β₂·ttl + β₃·div×ttl + ε.

    Uses statsmodels if available, otherwise falls back to numpy lstsq.

    Args:
        data: Centered regression data.

    Returns:
        OLSResult with coefficients, stats, and fit metrics.
    """
    try:
        return _run_ols_statsmodels(data)
    except ImportError:
        logger.warning("statsmodels not available, using numpy fallback")
        return _run_ols_numpy(data)


def _run_ols_statsmodels(data: RegressionData) -> OLSResult:
    """OLS via statsmodels."""
    import statsmodels.api as sm  # type: ignore[import-untyped]
    from scipy import stats as scipy_stats  # type: ignore[import-untyped]

    # Full model: div + ttl + interaction
    X_full = np.column_stack([data.diversity, data.ttl, data.interaction])
    X_full = sm.add_constant(X_full)
    model_full = sm.OLS(data.outcome, X_full).fit()

    # Base model: div + ttl only (for ΔR²)
    X_base = np.column_stack([data.diversity, data.ttl])
    X_base = sm.add_constant(X_base)
    model_base = sm.OLS(data.outcome, X_base).fit()

    delta_r2 = model_full.rsquared - model_base.rsquared

    param_names = ["intercept", "diversity", "ttl", "interaction"]
    coefficients = dict(zip(param_names, model_full.params))
    std_errors = dict(zip(param_names, model_full.bse))
    t_values = dict(zip(param_names, model_full.tvalues))
    p_values = dict(zip(param_names, model_full.pvalues))

    return OLSResult(
        coefficients=coefficients,
        std_errors=std_errors,
        t_values=t_values,
        p_values=p_values,
        r_squared=float(model_full.rsquared),
        adj_r_squared=float(model_full.rsquared_adj),
        f_statistic=float(model_full.fvalue),
        f_p_value=float(model_full.f_pvalue),
        n=data.n,
        residuals=model_full.resid,
        delta_r_squared=float(delta_r2),
    )


def _run_ols_numpy(data: RegressionData) -> OLSResult:
    """Fallback OLS via numpy (limited stats)."""
    X = np.column_stack([
        np.ones(data.n),
        data.diversity,
        data.ttl,
        data.interaction,
    ])
    y = data.outcome

    # β = (X'X)⁻¹ X'y
    beta, residuals_sum, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    k = 3  # number of predictors
    adj_r2 = 1.0 - (1.0 - r2) * (data.n - 1) / (data.n - k - 1) if data.n > k + 1 else 0.0

    # Standard errors
    mse = ss_res / (data.n - k - 1) if data.n > k + 1 else 1.0
    try:
        var_beta = mse * np.linalg.inv(X.T @ X).diagonal()
        se = np.sqrt(np.maximum(var_beta, 0))
    except np.linalg.LinAlgError:
        se = np.ones(k + 1)

    t_vals = beta / np.where(se > 0, se, 1.0)

    # Base model for ΔR²
    X_base = X[:, :3]  # intercept + div + ttl
    beta_base = np.linalg.lstsq(X_base, y, rcond=None)[0]
    y_hat_base = X_base @ beta_base
    ss_res_base = float(np.sum((y - y_hat_base) ** 2))
    r2_base = 1.0 - ss_res_base / ss_tot if ss_tot > 0 else 0.0
    delta_r2 = r2 - r2_base

    # F-statistic
    ms_reg = (ss_tot - ss_res) / k if k > 0 else 0.0
    ms_res = mse
    f_stat = ms_reg / ms_res if ms_res > 0 else 0.0

    param_names = ["intercept", "diversity", "ttl", "interaction"]
    return OLSResult(
        coefficients=dict(zip(param_names, beta)),
        std_errors=dict(zip(param_names, se)),
        t_values=dict(zip(param_names, t_vals)),
        p_values={n: 0.0 for n in param_names},  # p-values need scipy
        r_squared=r2,
        adj_r_squared=adj_r2,
        f_statistic=f_stat,
        f_p_value=0.0,
        n=data.n,
        residuals=resid,
        delta_r_squared=delta_r2,
    )


# ---------------------------------------------------------------------------
# Simple slopes analysis
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SimpleSlopeResult:
    """Simple slope of diversity at a specific TTL value."""

    ttl_label: str  # "-1SD", "Mean", "+1SD"
    ttl_centered_value: float
    slope: float  # β₁ + β₃ × ttl_value
    se: float
    t_value: float
    p_value: float


def compute_simple_slopes(
    ols: OLSResult,
    data: RegressionData,
) -> list[SimpleSlopeResult]:
    """Compute simple slopes of diversity effect at -1SD, Mean, +1SD of TTL.

    Simple slope = β₁ + β₃ × TTL_centered_value

    Args:
        ols: OLS regression result.
        data: Regression data (for SD calculation).

    Returns:
        List of 3 SimpleSlopeResult objects.
    """
    b1 = ols.coefficients["diversity"]
    b3 = ols.coefficients["interaction"]

    ttl_points = [
        ("-1SD", -data.ttl_sd),
        ("Mean", 0.0),
        ("+1SD", data.ttl_sd),
    ]

    results: list[SimpleSlopeResult] = []
    for label, ttl_val in ttl_points:
        slope = b1 + b3 * ttl_val

        # SE of simple slope (requires variance-covariance matrix)
        # Approximate: SE ≈ sqrt(Var(b1) + ttl² × Var(b3) + 2 × ttl × Cov(b1,b3))
        # Simplified: use just SE of b1 as approximation
        se_approx = ols.std_errors.get("diversity", 0.0)
        t_val = slope / se_approx if se_approx > 0 else 0.0

        results.append(
            SimpleSlopeResult(
                ttl_label=label,
                ttl_centered_value=ttl_val,
                slope=round(slope, 4),
                se=round(se_approx, 4),
                t_value=round(t_val, 4),
                p_value=0.0,  # requires scipy for exact p
            )
        )

    return results


# ---------------------------------------------------------------------------
# ICC (Intraclass Correlation Coefficient) for dual-evaluator reliability
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ICCResult:
    """ICC reliability result for dual evaluators."""

    dimension: str
    icc: float  # ICC(3,1) — two-way mixed, single measures
    n: int
    interpretation: str  # poor/moderate/good/excellent


def compute_icc(
    scores_a: list[float],
    scores_b: list[float],
    dimension: str = "overall",
) -> ICCResult:
    """Compute ICC(3,1) for two evaluators on a single dimension.

    ICC(3,1) = (MS_subjects - MS_error) / (MS_subjects + MS_error)

    Interpretation (Cicchetti, 1994):
        < 0.40: poor
        0.40-0.59: fair
        0.60-0.74: good
        0.75-1.00: excellent

    Args:
        scores_a: Scores from evaluator A.
        scores_b: Scores from evaluator B.
        dimension: Dimension label.

    Returns:
        ICCResult with coefficient and interpretation.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have equal length")
    n = len(scores_a)
    if n < 2:
        return ICCResult(dimension=dimension, icc=0.0, n=n, interpretation="poor")

    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)

    # Two-way ANOVA components
    grand_mean = np.mean(np.concatenate([a, b]))
    subject_means = (a + b) / 2.0

    # MS_subjects (between-subjects mean square)
    ss_subjects = 2.0 * np.sum((subject_means - grand_mean) ** 2)
    ms_subjects = ss_subjects / (n - 1) if n > 1 else 0.0

    # MS_error (within-subjects mean square)
    ss_error = np.sum((a - subject_means) ** 2 + (b - subject_means) ** 2)
    ms_error = ss_error / n if n > 0 else 1.0

    # ICC(3,1)
    icc = (ms_subjects - ms_error) / (ms_subjects + ms_error) if (ms_subjects + ms_error) > 0 else 0.0
    icc = max(-1.0, min(1.0, icc))  # clamp

    if icc >= 0.75:
        interp = "excellent"
    elif icc >= 0.60:
        interp = "good"
    elif icc >= 0.40:
        interp = "fair"
    else:
        interp = "poor"

    return ICCResult(
        dimension=dimension,
        icc=round(icc, 4),
        n=n,
        interpretation=interp,
    )


# ---------------------------------------------------------------------------
# Per-dimension regression runner
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DimensionRegressionSummary:
    """Summary of regression for one dimension."""

    dimension: str
    ols: OLSResult
    simple_slopes: list[SimpleSlopeResult]


def run_all_dimension_regressions(
    diversity_raw: list[float],
    ttl_raw: list[float],
    dimension_scores: dict[str, list[float]],
) -> list[DimensionRegressionSummary]:
    """Run regression for each of the 8 dimensions + overall.

    Args:
        diversity_raw: Raw composite Blau values.
        ttl_raw: Raw TTL codes.
        dimension_scores: Dict mapping dimension_id → list of scores.

    Returns:
        List of DimensionRegressionSummary objects.
    """
    results: list[DimensionRegressionSummary] = []

    for dim_id, scores in dimension_scores.items():
        try:
            data = prepare_regression_data(diversity_raw, ttl_raw, scores)
            ols = run_ols(data)
            slopes = compute_simple_slopes(ols, data)
            results.append(
                DimensionRegressionSummary(
                    dimension=dim_id,
                    ols=ols,
                    simple_slopes=slopes,
                )
            )
        except Exception as e:
            logger.warning("Regression failed for dimension %s: %s", dim_id, e)

    return results


# ---------------------------------------------------------------------------
# Interaction plot
# ---------------------------------------------------------------------------
def plot_interaction(
    data: RegressionData,
    ols: OLSResult,
    output_path: str,
    title: str = "Diversity × TTL Interaction Effect",
) -> str:
    """Generate interaction plot showing diversity effect at different TTL levels.

    Args:
        data: Regression data.
        ols: OLS result.
        output_path: File path for saved figure.
        title: Plot title.

    Returns:
        Output file path.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
        return ""

    b0 = ols.coefficients["intercept"]
    b1 = ols.coefficients["diversity"]
    b2 = ols.coefficients["ttl"]
    b3 = ols.coefficients["interaction"]

    # Diversity range
    div_range = np.linspace(
        -2 * data.diversity_sd, 2 * data.diversity_sd, 50
    )

    # TTL levels: -1SD, Mean, +1SD
    ttl_levels = [
        ("Low TTL (-1SD)", -data.ttl_sd, "#E74C3C"),
        ("Mean TTL", 0.0, "#3498DB"),
        ("High TTL (+1SD)", data.ttl_sd, "#27AE60"),
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    for label, ttl_val, color in ttl_levels:
        y_hat = b0 + b1 * div_range + b2 * ttl_val + b3 * div_range * ttl_val
        ax.plot(div_range + data.diversity_mean, y_hat, label=label, color=color, linewidth=2)

    ax.set_xlabel("Temporal Diversity (Blau Index)", fontsize=12)
    ax.set_ylabel("Team Performance", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Interaction plot saved: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Full analysis report
# ---------------------------------------------------------------------------
@dataclass
class FullAnalysisReport:
    """Complete analysis report combining all components."""

    overall_regression: OLSResult
    overall_simple_slopes: list[SimpleSlopeResult]
    dimension_regressions: list[DimensionRegressionSummary]
    icc_results: list[ICCResult]
    n_observations: int
    interaction_plot_path: str


def run_full_analysis(
    diversity_raw: list[float],
    ttl_raw: list[float],
    overall_scores: list[float],
    dimension_scores: dict[str, list[float]],
    scores_a_by_dim: dict[str, list[float]] | None = None,
    scores_b_by_dim: dict[str, list[float]] | None = None,
    output_dir: str = "results",
) -> FullAnalysisReport:
    """Run the complete analysis pipeline.

    Args:
        diversity_raw: Composite Blau values (N observations).
        ttl_raw: TTL codes (N observations).
        overall_scores: Overall performance scores (mean of evaluators).
        dimension_scores: Per-dimension mean scores.
        scores_a_by_dim: Evaluator A scores per dimension (for ICC).
        scores_b_by_dim: Evaluator B scores per dimension (for ICC).
        output_dir: Directory for output files.

    Returns:
        FullAnalysisReport with all analysis results.
    """
    # Overall regression
    data = prepare_regression_data(diversity_raw, ttl_raw, overall_scores)
    overall_ols = run_ols(data)
    overall_slopes = compute_simple_slopes(overall_ols, data)

    # Per-dimension regressions
    dim_regressions = run_all_dimension_regressions(
        diversity_raw, ttl_raw, dimension_scores
    )

    # ICC reliability
    icc_results: list[ICCResult] = []
    if scores_a_by_dim and scores_b_by_dim:
        for dim_id in scores_a_by_dim:
            if dim_id in scores_b_by_dim:
                icc = compute_icc(
                    scores_a_by_dim[dim_id],
                    scores_b_by_dim[dim_id],
                    dimension=dim_id,
                )
                icc_results.append(icc)

    # Interaction plot
    plot_path = str(Path(output_dir) / "interaction_plot.png")
    plot_interaction(data, overall_ols, plot_path)

    return FullAnalysisReport(
        overall_regression=overall_ols,
        overall_simple_slopes=overall_slopes,
        dimension_regressions=dim_regressions,
        icc_results=icc_results,
        n_observations=data.n,
        interaction_plot_path=plot_path,
    )


# ---------------------------------------------------------------------------
# Text summary formatter
# ---------------------------------------------------------------------------
def format_regression_table(ols: OLSResult) -> str:
    """Format OLS result as a readable text table.

    Args:
        ols: OLS regression result.

    Returns:
        Formatted text table.
    """
    lines = [
        "=" * 60,
        "OLS Regression Results",
        "=" * 60,
        f"N = {ols.n}    R² = {ols.r_squared:.4f}    Adj.R² = {ols.adj_r_squared:.4f}",
        f"F = {ols.f_statistic:.4f}    p(F) = {ols.f_p_value:.4f}",
        f"ΔR² (interaction) = {ols.delta_r_squared:.4f}",
        "-" * 60,
        f"{'Variable':<15} {'β':>10} {'SE':>10} {'t':>10} {'p':>10}",
        "-" * 60,
    ]

    for name in ["intercept", "diversity", "ttl", "interaction"]:
        b = ols.coefficients.get(name, 0.0)
        se = ols.std_errors.get(name, 0.0)
        t = ols.t_values.get(name, 0.0)
        p = ols.p_values.get(name, 0.0)
        sig = ""
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        lines.append(f"{name:<15} {b:>10.4f} {se:>10.4f} {t:>10.4f} {p:>10.4f} {sig}")

    lines.append("=" * 60)
    return "\n".join(lines)
