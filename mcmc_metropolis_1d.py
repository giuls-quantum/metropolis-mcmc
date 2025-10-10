# mcmc_metropolis_1d.py
# ------------------------------------------------------------
# Markov Chain Monte Carlo (MCMC) simulation using Metropolis–Hastings
# to sample from an unnormalized 1D target distribution:
#     f(x) = exp(-x^4 + x^2 - x)
# Author: Giulia Maniccia
# ------------------------------------------------------------
# mcmc_metropolis_1d.py
# ------------------------------------------------------------
# Markov Chain Monte Carlo (MCMC) simulation using Metropolis–Hastings
# to sample from an unnormalized 1D target distribution:
#     f(x) ∝ exp(-x^4 + x^2 - x)
# Author: [Your Name]
# Date: 2025-10-10
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Function definitions
# -----------------------------

# Target distribution
def log_f(x):
    return -x**4 + x**2 - x

def f(x):
    return np.exp(log_f(x))

# Metropolis-Hastings sampler
def metropolis(log_pdf, x0, n_steps, sigma, rng=None):
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
    return x, acceptance_rate

# Autocorrelation and ESS
def autocorr(x, max_lag=1000):
    x = np.asarray(x)
    x = x - np.mean(x)
    n = len(x)
    m = 1 << (2*n-1).bit_length()
    f = np.fft.rfft(x, n=m)
    acf = np.fft.irfft(f*np.conjugate(f), n=m)[:n]
    acf /= acf[0]
    return acf[:min(max_lag, n-1)+1]

def integrated_autocorr_time(x, max_lag=5000):
    rho = autocorr(x, max_lag=max_lag)
    s = 0.0
    for l in range(1, len(rho)//2*2, 2):
        pair_sum = rho[l] + rho[l+1]
        if pair_sum <= 0:
            break
        s += pair_sum
    return max(1.0, 1 + 2*s)

def effective_sample_size(n_samples, iat):
    return n_samples / iat

# -----------------------------
# Program start
# -----------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(2025)
    N_STEPS = 200_000
    X0 = 0.0
    SIGMA = 0.6

    samples, acc_rate = metropolis(log_f, X0, N_STEPS, SIGMA, rng)
    print(f"Acceptance rate: {acc_rate:.3f}")

    BURN_IN = 20_000
    THIN = 10
    posterior = samples[BURN_IN::THIN]
    print(f"Samples after burn-in & thinning: {len(posterior)}")

    mean_post = np.mean(posterior)
    std_post = np.std(posterior, ddof=1)
    print(f"Posterior mean = {mean_post:.5f}, std = {std_post:.5f}")

    iat = integrated_autocorr_time(posterior, max_lag=5000)
    ess = effective_sample_size(len(posterior), iat)
    print(f"Estimated IAT: {iat:.2f}, Effective sample size: {ess:.1f}")

    # Data Visualization
    import matplotlib.gridspec as gridspec
    plt.figure(figsize=(12,9))
    gs = gridspec.GridSpec(3,2,height_ratios=[1,1,1])

    # Trace plot
    ax0 = plt.subplot(gs[0,:])
    ax0.plot(samples[:2000], lw=0.6)
    ax0.set_title("Trace plot (first 2000 steps)")
    ax0.set_xlabel("Step")
    ax0.set_ylabel("x")

    # Histogram vs target
    ax1 = plt.subplot(gs[1,0])
    ax1.hist(posterior, bins=100, density=True, alpha=0.6, label="MCMC samples")
    xs = np.linspace(min(posterior)-1, max(posterior)+1, 1000)
    px = f(xs)/np.trapz(f(xs), xs)
    ax1.plot(xs, px, "r-", lw=2, label="Target (normalized)")
    ax1.set_title("Histogram vs Target Distribution")
    ax1.legend()

    # Autocorrelation
    ax2 = plt.subplot(gs[1,1])
    rho = autocorr(posterior, max_lag=200)
    ax2.plot(np.arange(len(rho)), rho, marker=".", lw=1)
    ax2.set_xlabel("Lag")
    ax2.set_ylabel("ρ(lag)")
    ax2.set_title("Autocorrelation Function")

    # Zoomed density
    ax3 = plt.subplot(gs[2,0])
    ax3.hist(posterior, bins=100, density=True, alpha=0.6)
    ax3.plot(xs, px, "r-", lw=2)
    ax3.set_xlim(-3,3)
    ax3.set_title("Zoomed Density (x ∈ [-3,3])")

    # Running mean
    ax4 = plt.subplot(gs[2,1])
    running_mean = np.cumsum(samples)/(np.arange(len(samples))+1)
    ax4.plot(running_mean)
    ax4.axvline(BURN_IN, color="r", linestyle="--", label="Burn-in")
    ax4.set_title("Running Mean Convergence")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Mean")
    ax4.legend()

    plt.tight_layout()
    plt.savefig("mcmc_plots.png", dpi=300)
    #Uncomment the following line to immediately see the plot in a matplotlib window:
    #plt.show()
    
# -----------------------------
# Program end
# -----------------------------
