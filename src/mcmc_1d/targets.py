"""
Target distribution(s) for the 1D MCMC example.
More target distributions can be added here as needed.
"""

import numpy as np

def quartic_log_target(x: float) -> float:
    """Logarithm of the target distribution: f(x) = exp(-x^4 + x^2 - x)."""
    return -x**4 + x**2 - x

def f(x: float) -> float:
    """Target distribution (unnormalized): f(x) = exp(-x^4 + x^2 - x)"""
    return np.exp(quartic_log_target(x))