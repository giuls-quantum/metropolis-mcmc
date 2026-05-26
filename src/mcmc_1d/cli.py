"""
Command-line interface for the 1D Metropolis MCMC sampler.
"""

import argparse

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from src.mcmc_1d.diagnostics import autocorr
from src.mcmc_1d.experiment import run_experiment
from src.mcmc_1d.targets import f as target_pdf


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 1D Metropolis-Hastings experiment and save a diagnostics plot."
    )
    parser.add_argument("--x0", type=float, default=0.0, help="Initial value for the chain")
    parser.add_argument("--n_steps", type=int, default=200_000, help="Number of MCMC steps")
    parser.add_argument("--sigma", type=float, default=0.6, help="Standard deviation of the proposal distribution")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")
    parser.add_argument("--burn-in", dest="burn_in", type=int, default=20_000, help="Number of burn-in samples to discard")
    parser.add_argument("--thin", type=int, default=10, help="Thinning factor applied after burn-in")
    parser.add_argument("--output", type=str, default="mcmc_plots.png", help="Path where the summary plot is saved")
    parser.add_argument("--show", action="store_true", help="Display the plot window after saving")
    return parser


def _plot_summary(samples: np.ndarray, burn_in: int, thin: int, output: str, show: bool) -> None:
    effective_burn_in = min(max(0, burn_in), max(0, len(samples) - 1))
    posterior = samples[effective_burn_in::max(1, thin)]
    xs = np.linspace(np.min(posterior) - 1.0, np.max(posterior) + 1.0, 1000)
    px = target_pdf(xs)
    px = px / np.trapezoid(px, xs)

    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

    ax0 = plt.subplot(gs[0, :])
    ax0.plot(samples[:2000], lw=0.6)
    ax0.set_title("Trace plot (first 2000 steps)")
    ax0.set_xlabel("Step")
    ax0.set_ylabel("x")

    ax1 = plt.subplot(gs[1, 0])
    ax1.hist(posterior, bins=100, density=True, alpha=0.6, label="MCMC samples")
    ax1.plot(xs, px, "r-", lw=2, label="Target (normalized)")
    ax1.set_title("Histogram vs Target Distribution")
    ax1.legend()

    ax2 = plt.subplot(gs[1, 1])
    rho = autocorr(posterior, max_lag=200)
    ax2.plot(np.arange(len(rho)), rho, marker=".", lw=1)
    ax2.set_xlabel("Lag")
    ax2.set_ylabel("ρ(lag)")
    ax2.set_title("Autocorrelation Function")

    ax3 = plt.subplot(gs[2, 0])
    ax3.hist(posterior, bins=100, density=True, alpha=0.6)
    ax3.plot(xs, px, "r-", lw=2)
    ax3.set_xlim(-3, 3)
    ax3.set_title("Zoomed Density (x ∈ [-3,3])")

    ax4 = plt.subplot(gs[2, 1])
    running_mean = np.cumsum(samples) / (np.arange(len(samples)) + 1)
    ax4.plot(running_mean)
    ax4.axvline(burn_in, color="r", linestyle="--", label="Burn-in")
    ax4.set_title("Running Mean Convergence")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Mean")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    """Parse command-line arguments and run the MCMC experiment."""
    parser = _build_parser()
    args = parser.parse_args()

    result, diagnostics = run_experiment(
        x0=args.x0,
        n_steps=args.n_steps,
        sigma=args.sigma,
        seed=args.seed,
    )

    effective_burn_in = min(max(0, args.burn_in), max(0, len(result.samples) - 1))
    posterior = result.samples[effective_burn_in::max(1, args.thin)]
    mean_post = np.mean(posterior)
    std_post = np.std(posterior, ddof=1) if len(posterior) > 1 else 0.0

    print(f"Acceptance rate: {result.acceptance_rate:.3f}")
    print(f"Samples after burn-in & thinning: {len(posterior)}")
    print(f"Posterior mean = {mean_post:.5f}, std = {std_post:.5f}")
    print(f"Estimated IAT: {diagnostics.integrated_autocorr_time:.2f}, Effective sample size: {diagnostics.effective_sample_size:.1f}")

    _plot_summary(result.samples, effective_burn_in, args.thin, args.output, args.show)


if __name__ == "__main__":
    main()