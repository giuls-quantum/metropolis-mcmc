"""1D Metropolis-Hastings MCMC Sampler and Diagnostics."""

from .results import MCMCResults
from .diagnostics import compute_summary, compute_ess, compute_rhat
from .plotting import plot_trace, plot_posterior, plot_autocorr

__all__ = [
    "MCMCResults",
    "compute_summary",
    "compute_ess",
    "compute_rhat",
    "plot_trace",
    "plot_posterior",
    "plot_autocorr",
]