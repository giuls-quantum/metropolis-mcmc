"""
Plot generators for the 1D MCMC example.
"""

from typing import Any, Optional
import arviz as az
from .results import MCMCResults


def plot_trace(results: MCMCResults, save_path: Optional[str] = None) -> Any:
    """
    Generates posterior density and trace plot.
    """
    idata = results.to_arviz()
    plot_obj = az.plot_trace(idata)

    if save_path:
        plot_obj.savefig(save_path, bbox_inches="tight")
    return plot_obj


def plot_posterior(results: MCMCResults, save_path: Optional[str] = None) -> Any:
    """
    Plots marginal posterior distribution with Highest Density Interval (HDI).
    """
    idata = results.to_arviz()
    plot_obj = az.plot_posterior(idata)

    if save_path:
        plot_obj.savefig(save_path, bbox_inches="tight")
    return plot_obj


def plot_autocorr(results: MCMCResults, save_path: Optional[str] = None) -> Any:
    """
    Plots autocorrelation for the sampling chain.
    """
    idata = results.to_arviz()
    plot_obj = az.plot_autocorr(idata)

    if save_path:
        plot_obj.savefig(save_path, bbox_inches="tight")
    return plot_obj