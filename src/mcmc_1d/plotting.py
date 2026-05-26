"""
Plot generators for the 1D MCMC example.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.mcmc_1d.targets import quartic_log_target as log_pdf
from src.mcmc_1d.targets import f as target_pdf
from src.mcmc_1d.diagnostics import autocorr as acf

def plot_target_distribution(log_pdf, x_range=(-3, 3), num_points=1000):
    """Plot the target distribution."""
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = target_pdf(x)
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label='Target Distribution', color='blue')
    plt.title('Target Distribution')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid()
    plt.legend()
    plt.show()

def plot_trace(x, burn_in=0):
    """Plot the trace of the MCMC samples."""
    plt.figure(figsize=(8, 4))
    plt.plot(x[burn_in:], label='MCMC Trace', color='orange')
    plt.title('MCMC Trace Plot')
    plt.xlabel('Iteration')
    plt.ylabel('x')
    plt.grid()
    plt.legend()
    plt.show()

def plot_histogram_vs_target(samples, log_pdf, x_range=(-3, 3), num_points=1000):
    """Plot histogram of samples vs target distribution."""
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = target_pdf(x)
    
    plt.figure(figsize=(8, 4))
    plt.hist(samples, bins=50, density=True, alpha=0.6, label='MCMC Samples', color='green')
    plt.plot(x, y, label='Target Distribution', color='red', lw=2)
    plt.title('Histogram of MCMC Samples vs Target Distribution')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.grid()
    plt.legend()
    plt.show()

def plot_autocorrelation(samples, max_lag=100):
    """Plot the autocorrelation function of the samples."""
    rho = acf(samples, max_lag=max_lag)
    
    plt.figure(figsize=(8, 4))
    plt.stem(range(len(rho)), rho)
    plt.title('Autocorrelation Function')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.grid()
    plt.show()

def plot_running_mean(samples, burn_in=0):
    """Plot the running mean of the samples."""
    running_mean = np.cumsum(samples) / (np.arange(len(samples)) + 1)
    
    plt.figure(figsize=(8, 4))
    plt.plot(running_mean[burn_in:], label='Running Mean', color='purple')
    plt.axhline(np.mean(samples[burn_in:]), color='red', linestyle='--', label='Posterior Mean')
    plt.title('Running Mean of MCMC Samples')
    plt.xlabel('Iteration')
    plt.ylabel('Running Mean')
    plt.grid()
    plt.legend()
    plt.show()

def plot_all(samples, log_pdf):
    """Generate all plots for the MCMC samples."""
    plot_target_distribution(log_pdf)
    plot_trace(samples, burn_in=20000)
    plot_histogram_vs_target(samples[20000:], log_pdf)
    plot_autocorrelation(samples[20000:])
    plot_running_mean(samples, burn_in=20000)