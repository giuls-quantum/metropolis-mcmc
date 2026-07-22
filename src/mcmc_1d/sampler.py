"""
Metropolis-Hastings MCMC sampler for 1D distributions.
"""

from typing import Callable

import numpy as np

# Import custom dataclass for structured output of MCMC runs
from src.mcmc_1d.results import MCMCResults

def metropolis(
        log_pdf: Callable[[float], float], 
        x0: float, 
        n_steps: int, 
        sigma: float, 
        rng: np.random.Generator | None = None,
    ) -> MCMCResults:
    """Run the Metropolis-Hastings sampler.
    
    Args:
        log_pdf: The log probability density function of the target distribution.
        x0: Initial value for the MCMC chain.
        n_steps: Number of MCMC steps to run.
        sigma: Standard deviation of the proposal distribution.
        rng: Random number generator (optional).

    Returns:
        The result of the MCMC sampling.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.empty(n_steps)
    x[0] = x0
    n_accept = 0

    for i in range(1, n_steps):
        current = x[i-1]
        proposal = current + rng.normal(0, sigma)
        log_r = log_pdf(proposal) - log_pdf(current)
        if np.log(rng.random()) < min(0.0, log_r):
            x[i] = proposal
            n_accept += 1
        else:
            x[i] = current

    acceptance_rate = n_accept / (n_steps - 1)
    return MCMCResults(samples=x, acceptance_rate=acceptance_rate)