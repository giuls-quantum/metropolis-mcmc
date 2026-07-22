"""
Script to run the MCMC experiment and compute diagnostics for a 1D target distribution.
"""

import numpy as np

from src.mcmc_1d.diagnostics import summarize_diagnostics
from src.mcmc_1d.results import MCMCResults, MCMCDiagnostics
from src.mcmc_1d.sampler import metropolis
from src.mcmc_1d.targets import quartic_log_target


def run_experiment(
        x0: float,
        n_steps: int,
        sigma: float,
        seed: int | None = None,
        max_lag: int = 5000,
) -> tuple[MCMCResults, MCMCDiagnostics]:
    """
    Run the MCMC experiment and return results and diagnostics.

    Args:
        x0: Initial value for the MCMC chain.
        n_steps: Number of MCMC steps to run.
        sigma: Standard deviation of the proposal distribution.
        seed: Random seed for reproducibility (optional).
        max_lag: Maximum lag for autocorrelation computation.

    Returns:
        A tuple containing the MCMCResult and MCMCDiagnostics.
    """

    result = metropolis(
        log_pdf=quartic_log_target,
        x0=x0,
        n_steps=n_steps,
        sigma=sigma,
        rng=np.random.default_rng(seed),
    )

    diagnostics = summarize_diagnostics(result.samples, max_lag=max_lag)
    diagnostics_summary = MCMCDiagnostics(
        autocorrelation=diagnostics['autocorrelation'],
        integrated_autocorr_time=diagnostics['integrated_autocorr_time'],
        effective_sample_size=diagnostics['effective_sample_size'],
    )

    return result, diagnostics_summary
