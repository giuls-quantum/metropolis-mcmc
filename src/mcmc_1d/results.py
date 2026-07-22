"""
Custom data classes to represent MCMC results and diagnostics.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import arviz as az


@dataclass
class MCMCDiagnostics:
    """Container for basic MCMC diagnostics."""
    autocorrelation: np.ndarray
    integrated_autocorr_time: float
    effective_sample_size: float


@dataclass
class MCMCResults:
    """
    Container class for MCMC sampling results and metadata.
    """
    samples: np.ndarray
    acceptance_rate: float
    burn_in: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        self.samples = np.asarray(self.samples)

    @property
    def post_burn_in_samples(self) -> np.ndarray:
        """Returns sampling chain with burn-in iterations discarded."""
        return self.samples[self.burn_in:]

    def to_arviz(self) -> Any:
        """
        Converts post-burn-in samples into an ArviZ InferenceData object.
        Supports both 1D arrays (single chain) and 2D arrays (multi-chain).
        """
        chain_data = self.post_burn_in_samples

        # Reshape 1D array (samples,) to 2D array (1, samples) for ArviZ chain formatting
        if chain_data.ndim == 1:
            chain_data = chain_data[np.newaxis, :]

        return az.from_dict(
            data={"posterior": {"x": chain_data}}
        )