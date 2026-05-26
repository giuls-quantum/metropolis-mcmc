"""
Custom data classes to represent MCMC results and diagnostics.
"""

from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class MCMCResult:
    """Result of a single MCMC sampling run."""
    samples: np.ndarray
    acceptance_rate: float

@dataclass(frozen=True)
class MCMCDiagnostics:
    """Diagnostics for MCMC sampling."""
    autocorrelation: np.ndarray
    integrated_autocorr_time: float
    effective_sample_size: float