import pytest
import numpy as np


from src.mcmc_1d import diagnostics

def _get_diagnostic_function(*names):
    for name in names:
        if hasattr(diagnostics, name):
            return getattr(diagnostics, name)
    pytest.fail(f"No diagnostic function found in module for names {names}")


def test_autocorrelation_has_expected_properties():
    autocorr = _get_diagnostic_function("autocorrelation", "autocorr")
    series = np.array([1.0, 0.0, -1.0, 0.0])
    result = autocorr(series)
    result = np.asarray(result)

    assert result.ndim == 1
    assert result[0] == pytest.approx(1.0)
    assert np.all(np.isfinite(result))


def test_effective_sample_size_returns_reasonable_value():
    ess = _get_diagnostic_function("effective_sample_size", "ess")
    chain = np.linspace(0, 1, 50)
    value = ess(chain)

    assert np.isscalar(value)
    assert value >= 1.0
    assert value <= 50.0


def test_integrated_autocorr_time_returns_reasonable_value():
    iat = _get_diagnostic_function("integrated_autocorr_time", "iat")
    chain = np.linspace(0, 1, 50)
    value = iat(chain)

    assert np.isscalar(value)
    assert value >= 1.0
    assert value <= 50.0

def test_summarize_diagnostics_returns_dict():
    summarize_diagnostics = _get_diagnostic_function("summarize_diagnostics")
    chain = np.linspace(0, 1, 50)
    result = summarize_diagnostics(chain)

    assert isinstance(result, dict)
    assert "autocorrelation" in result
    assert "effective_sample_size" in result
    assert "integrated_autocorr_time" in result