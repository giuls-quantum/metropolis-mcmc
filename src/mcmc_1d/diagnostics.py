"""
MCMC diagnostics for 1D samples, including:
- Autocorrelation function
- Integrated autocorrelation time
- Effective sample size
"""

import numpy as np

def autocorr( samples: np.ndarray, max_lag: int = 1000 ) -> np.ndarray:
    """Compute the autocorrelation function of the samples.
    
    Args:
        samples: Array of MCMC samples.
        max_lag: Maximum lag for which to compute the autocorrelation.

    Returns:
        Array of autocorrelations at each lag.
    """
    x = np.asarray(samples)
    if len(x) == 0:
        return np.array([], dtype=float)
    if len(x) == 1:
        return np.array([1.0], dtype=float)

    x = x - np.mean(x)
    n = len(x)
    m = 1 << (2*n-1).bit_length()  # Next power of 2 for zero-padding
    f = np.fft.rfft(x, n=m)
    acf = np.fft.irfft(f * np.conjugate(f), n=m)[:n]
    acf /= acf[0]  # Normalize
    return acf[:min(max_lag, n-1)+1]

def integrated_autocorr_time( samples: np.ndarray, max_lag: int = 5000 ) -> float:
    """Compute the integrated autocorrelation time.
    
    Args:        
        samples: Array of MCMC samples.
        max_lag: Maximum lag for which to compute the autocorrelation.
    
    Returns: 
        Integrated autocorrelation time.
    """
    rho = autocorr(samples, max_lag=max_lag)
    s = 0.0
    for l in range(1, len(rho)//2*2, 2):
        pair_sum = rho[l] + rho[l+1]
        if pair_sum <= 0:
            break
        s += pair_sum
    return max(1.0, 1 + 2*s)

def effective_sample_size( samples: np.ndarray, max_lag: int = 5000 ) -> float:
    """Compute the effective sample size.
    
    Args:
        samples: Array of MCMC samples.
        max_lag: Maximum lag for which to compute the autocorrelation.

    Returns:
        Effective sample size.
    """
    n_samples = len(samples)
    iat = integrated_autocorr_time(samples, max_lag=max_lag)
    return n_samples / iat

def summarize_diagnostics( samples: np.ndarray, max_lag: int = 5000 ) -> dict:
    """Summarize MCMC diagnostics in a dictionary.
    
    Args:        
        samples: Array of MCMC samples.
        max_lag: Maximum lag for which to compute the autocorrelation.

    Returns:
        Dictionary containing the diagnostics.
    """
    return {
        'autocorrelation': autocorr(samples, max_lag=max_lag),
        'integrated_autocorr_time': integrated_autocorr_time(samples, max_lag=max_lag),
        'effective_sample_size': effective_sample_size(samples, max_lag=max_lag),
    }