import pytest
import numpy as np


from src.mcmc_1d.targets import quartic_log_target, f
from src.mcmc_1d.sampler import metropolis
from src.mcmc_1d.results import MCMCResult


def test_metropolis_returns_mcmc_result():
    
    result = metropolis(
        log_pdf=quartic_log_target,
        x0=0.0,
        n_steps=10,
        sigma=1.0,
        rng=np.random.default_rng(123),
    )

    assert isinstance(result, MCMCResult)
    assert isinstance(result.samples, np.ndarray)
    assert result.samples.shape == (10,)
    assert isinstance(result.acceptance_rate, float)

def test_metropolis_keeps_initial_sample_as_first_element():

    result = metropolis(
        log_pdf=quartic_log_target,
        x0=2.5,
        n_steps=5,
        sigma=1.0,
        rng=np.random.default_rng(1),
    )

    assert result.samples[0] == pytest.approx(2.5)

def test_metropolis_acceptance_rate_in_range():

    result = metropolis(
        log_pdf=quartic_log_target,
        x0=0.0,
        n_steps=50,
        sigma=1.0,
        rng=np.random.default_rng(7),
    )

    assert 0.0 <= result.acceptance_rate <= 1.0

def test_metropolis_accepts_all_proposals_for_constant_log_pdf():
    def log_pdf_flat(_):
        return 0.0

    result = metropolis(
        log_pdf=log_pdf_flat,
        x0=0.0,
        n_steps=20,
        sigma=0.5,
        rng=np.random.default_rng(42),
    )

    assert result.acceptance_rate == pytest.approx(1.0)