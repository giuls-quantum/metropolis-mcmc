"""
MCMC diagnostics for 1D samples, including:
- Autocorrelation function
- Integrated autocorrelation time
- Effective sample size
"""

from typing import Any

import arviz as az
import numpy as np
import pandas as pd
from .results import MCMCResults


def autocorrelation(series: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Compute autocorrelation coefficients for a 1D series."""
    series = np.asarray(series, dtype=float)
    if series.size < 2:
        return np.array([1.0])

    centered = series - np.mean(series)
    denominator = np.dot(centered, centered)
    if denominator == 0:
        return np.ones(1, dtype=float)

    if max_lag is None:
        max_lag = len(series) - 1
    max_lag = min(max_lag, len(series) - 1)

    values = np.empty(max_lag + 1, dtype=float)
    values[0] = 1.0
    for lag in range(1, max_lag + 1):
        values[lag] = np.dot(centered[:-lag], centered[lag:]) / denominator
    return values


def effective_sample_size(series: np.ndarray) -> float:
    """Estimate effective sample size from autocorrelation."""
    autocorr_values = autocorrelation(series)
    integrated_time = integrated_autocorr_time(series)
    return float(len(series) / max(1.0, integrated_time))


def integrated_autocorr_time(series: np.ndarray) -> float:
    """Estimate the integrated autocorrelation time."""
    autocorr_values = autocorrelation(series)
    return float(max(1.0, 1.0 + 2.0 * np.sum(autocorr_values[1:])))


def compute_summary(results: MCMCResults, hdi_prob: float = 0.94) -> Any:
    """
    Computes summary statistics (mean, std, HDI, ESS, and r_hat) via ArviZ.
    """
    idata = results.to_arviz()
    return az.summary(idata, ci_prob=hdi_prob)


def compute_ess(results: MCMCResults, method: str = "bulk") -> float:
    """
    Calculates Effective Sample Size (ESS) for the chain ('bulk' or 'tail').
    """
    idata = results.to_arviz()
    ess_ds = az.ess(idata, method=method)
    return float(ess_ds["x"].values)


def compute_rhat(results: MCMCResults) -> float:
    """
    Calculates Gelman-Rubin diagnostic (r_hat).
    Best used when comparing multiple chains.
    """
    idata = results.to_arviz()
    rhat_ds = az.rhat(idata)
    return float(rhat_ds["x"].values)


def summarize_diagnostics(samples: np.ndarray, max_lag: int = 5000) -> dict[str, Any]:
    """Return a compact diagnostics summary for a 1D chain."""
    samples = np.asarray(samples, dtype=float)
    if samples.size < 2:
        return {
            "autocorrelation": np.array([1.0]),
            "integrated_autocorr_time": 1.0,
            "effective_sample_size": 1.0,
        }

    centered = samples - np.mean(samples)
    max_lag = min(max_lag, len(samples) - 1)
    autocorr_values = np.empty(max_lag + 1, dtype=float)
    autocorr_values[0] = 1.0

    for lag in range(1, max_lag + 1):
        if lag >= len(samples):
            autocorr_values[lag] = 0.0
            continue
        numerator = np.dot(centered[:-lag], centered[lag:])
        denominator = np.dot(centered, centered)
        autocorr_values[lag] = numerator / denominator if denominator != 0 else 0.0

    integrated_autocorr_time = float(1.0 + 2.0 * np.sum(autocorr_values[1:]))
    effective_sample_size = float(len(samples) / max(1.0, integrated_autocorr_time))

    return {
        "autocorrelation": autocorr_values,
        "integrated_autocorr_time": integrated_autocorr_time,
        "effective_sample_size": effective_sample_size,
    }