import pytest
import numpy as np


from src.mcmc_1d import targets

def _get_target_function(*names):
    for name in names:
        if hasattr(targets, name):
            return getattr(targets, name)
    pytest.fail(f"No target function found in module for names {names}")

def test_quartic_log_target_returns_scalar():
    log_target = _get_target_function("quartic_log_target", "log_pdf")
    x = 0.5
    result = log_target(x)

    assert np.isscalar(result)
    assert np.isfinite(result)

def test_f_returns_positive_value():
    target_pdf = _get_target_function("f", "target_pdf")
    x = 0.5
    result = target_pdf(x)

    assert np.isscalar(result)
    assert result > 0.0
    assert np.isfinite(result)