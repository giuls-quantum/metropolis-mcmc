# Metropolis MCMC

![Python](https://img.shields.io/badge/python-3.13-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

Python implementation of the **Metropolis-Hastings algorithm** for 1D target distributions, with visualization and diagnostic tools.  
This repository provides a simple yet educational framework for running and analyzing MCMC simulations.

## Features

- Implements the **Metropolis-Hastings sampler** in Python.
- Supports **1D target distributions** (unnormalized).
- Includes **diagnostic tools**:
  - Acceptance rate calculation
  - Burn-in and thinning
  - Autocorrelation analysis
  - Effective Sample Size (ESS)
- **Visualizations**:
  - Trace plots
  - Histograms vs target distribution
  - Autocorrelation function
  - Running mean for convergence check
- Fully reproducible using **NumPy random generators**.

## Requirements

Install dependencies with pip:

```bash
python3 -m pip install numpy matplotlib
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/giuls-quantum/metropolis-mcmc.git
cd metropolis-mcmc
```

2. Run the main script:

```bash
python3 mcmc_metropolis_1d.py
```

The script will:

- Run the Metropolis sampler for the specified number of steps.
- Print acceptance rate, posterior mean, standard deviation, and effective sample size.
- Generate plots for trace, histogram, autocorrelation, and running mean.
- Save the summary plot as `mcmc_plots.png`.

## Customization

You can modify parameters directly in `mcmc_metropolis_1d.py`:

```bash
N_STEPS = 200_000       # Number of MCMC iterations
X0 = 0.0                # Initial state
SIGMA = 0.6             # Proposal standard deviation
BURN_IN = 20_000        # Number of burn-in steps
THIN = 10               # Thinning factor
```

You can also replace the target distribution `log_f(x)` with any other 1D function proportional to the desired probability density.

## Notes

- For long chains, **saving plots without displaying them** (`plt.show()` commented out) allows automatic script execution without blocking the terminal.
- The code is designed for **educational and research purposes**, and can be extended to higher dimensions or other MCMC algorithms.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## References

- Metropolis, N., et al. (1953). *Equation of State Calculations by Fast Computing Machines*. Journal of Chemical Physics.  
- Hastings, W.K. (1970). *Monte Carlo Sampling Methods Using Markov Chains and Their Applications*. Biometrika.
