"""Generate all Noise Sweep (label-flip noise) artefacts.

Artefacts produced:
  1. noise_heatmap.{pdf,png}               – Heatmap (n x eta): acceptance rate
  2. acceptance_correctness_vs_eta.{pdf,png} – Lines for representative n values
  3. fourier_weight_attenuation.{pdf,png}   – Median weight vs eta with theory curve
  4. breakdown_points.csv                   – Per-n empirical & theoretical breakdown

Usage:
    uv run python results/figures/noise_sweep/plot_noise_sweep.py
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DATA_FILE = PROJECT_ROOT / "results" / "noise_sweep_4_16_100.pb"
OUT_DIR = SCRIPT_DIR  # figures land next to the script


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    half = z * math.sqrt((p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_data() -> dict:
    """Deserialise the noise_sweep .pb file to a Python dict."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.decode import decode
    return json.loads(decode(DATA_FILE))


def parse_eta(phi_description: str) -> float:
    """Extract eta value from phiDescription like 'noisy_parity_eta=0.2_s=12'."""
    m = re.search(r"eta=([\d.]+)", phi_description)
    return float(m.group(1)) if m else 0.0


def build_tables(trials: list[dict], epsilon: float) -> dict:
    """Aggregate trial-level data into per-(n, eta) summaries."""
    # Group trials by (n, eta)
    groups: dict[tuple[int, float], list[dict]] = defaultdict(list)
    for t in trials:
        eta = parse_eta(t["phiDescription"])
        groups[(t["n"], eta)].append(t)

    ns = sorted({n for (n, _) in groups})
    etas = sorted({eta for (_, eta) in groups})

    # Per-(n, eta) aggregates
    acceptance_rate: dict[tuple[int, float], float] = {}
    correctness_rate: dict[tuple[int, float], float] = {}
    acceptance_ci: dict[tuple[int, float], tuple[float, float]] = {}
    correctness_ci: dict[tuple[int, float], tuple[float, float]] = {}
    median_weight: dict[tuple[int, float], float] = {}
    median_threshold: dict[tuple[int, float], float] = {}

    for key, group in groups.items():
        total = len(group)
        accepted = sum(1 for t in group if t.get("accepted", False))
        correct = sum(1 for t in group if t.get("hypothesisCorrect", False))

        acceptance_rate[key] = accepted / total
        correctness_rate[key] = correct / total
        acceptance_ci[key] = wilson_ci(accepted, total)
        correctness_ci[key] = wilson_ci(correct, total)

        weights = [t["accumulatedWeight"] for t in group]
        thresholds = [t["acceptanceThreshold"] for t in group]
        median_weight[key] = float(np.median(weights))
        median_threshold[key] = float(np.median(thresholds))

    return {
        "ns": ns,
        "etas": etas,
        "acceptance_rate": acceptance_rate,
        "correctness_rate": correctness_rate,
        "acceptance_ci": acceptance_ci,
        "correctness_ci": correctness_ci,
        "median_weight": median_weight,
        "median_threshold": median_threshold,
        "epsilon": epsilon,
    }


# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------

def setup_style() -> None:
    sns.set_context("paper", font_scale=1.1)
    sns.set_palette("colorblind")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path)
        print(f"  {path.relative_to(PROJECT_ROOT)}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Artefact 1: Noise heatmap (n x eta) – acceptance rate
# ---------------------------------------------------------------------------

def plot_noise_heatmap(tables: dict) -> None:
    """Heatmap with rows=n, columns=eta, colour=acceptance rate."""
    ns = tables["ns"]
    etas = tables["etas"]
    acc = tables["acceptance_rate"]

    # Build matrix: rows=n (top to bottom: largest n first for readability)
    matrix = np.zeros((len(ns), len(etas)))
    for i, n in enumerate(ns):
        for j, eta in enumerate(etas):
            matrix[i, j] = acc.get((n, eta), 0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        origin="lower",
    )

    # Annotate cells
    for i in range(len(ns)):
        for j in range(len(etas)):
            val = matrix[i, j]
            text_colour = "black" if 0.3 < val < 0.8 else "white"
            ax.text(j, i, f"{val*100:.0f}%", ha="center", va="center",
                    fontsize=7, color=text_colour, fontweight="bold")

    ax.set_xticks(range(len(etas)))
    ax.set_xticklabels([f"{e:.2f}" for e in etas])
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels([str(n) for n in ns])
    ax.set_xlabel(r"Noise rate $\eta$")
    ax.set_ylabel(r"Dimension $n$")
    ax.set_title(r"Acceptance rate under label-flip noise ($n \times \eta$)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Acceptance rate")
    fig.tight_layout()
    save(fig, "noise_heatmap")


# ---------------------------------------------------------------------------
# Artefact 2: Acceptance & correctness vs eta for representative n
# ---------------------------------------------------------------------------

def plot_acceptance_correctness_vs_eta(tables: dict) -> None:
    """Line plot: acceptance and correctness vs eta for 3 representative n."""
    ns = tables["ns"]
    etas = tables["etas"]
    acc = tables["acceptance_rate"]
    corr = tables["correctness_rate"]
    epsilon = tables["epsilon"]

    # Pick representative n values: small, medium, large
    n_choices = []
    if 4 in ns:
        n_choices.append(4)
    if 8 in ns:
        n_choices.append(8)
    if 14 in ns:
        n_choices.append(14)
    elif ns[-1] >= 12:
        n_choices.append(ns[-1])

    # Theoretical max tolerable eta: (1-2*eta)^2 = eps^2/8
    # => 1-2*eta = eps / (2*sqrt(2))
    # => eta_max = (1 - eps/(2*sqrt(2))) / 2
    eta_theory = (1 - epsilon / (2 * math.sqrt(2))) / 2

    colours = sns.color_palette("colorblind", len(n_choices))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    eta_arr = np.array(etas)

    for idx, n in enumerate(n_choices):
        acc_vals = [acc.get((n, e), 0.0) for e in etas]
        corr_vals = [corr.get((n, e), 0.0) for e in etas]

        ax.plot(eta_arr, acc_vals, "-o", color=colours[idx], markersize=4,
                label=f"Acceptance ($n={n}$)")
        ax.plot(eta_arr, corr_vals, "--s", color=colours[idx], markersize=4,
                alpha=0.7, label=f"Correctness ($n={n}$)")

    # Theoretical threshold line
    if 0 < eta_theory < 0.5:
        ax.axvline(eta_theory, ls=":", color="grey", alpha=0.8, linewidth=1.5,
                   label=rf"$(1-2\eta)^2 = \varepsilon^2/8$"
                         f"\n" rf"$\eta_{{max}} = {eta_theory:.3f}$")
        # Shade safe region
        ax.axvspan(0, eta_theory, alpha=0.08, color="green")
        ax.text(eta_theory / 2, 0.05, "safe", ha="center", va="bottom",
                fontsize=9, color="green", alpha=0.6, fontstyle="italic")

    ax.set_xlabel(r"Noise rate $\eta$")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(-0.01, max(etas) + 0.01)
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.set_title(r"Acceptance & correctness vs label-flip noise $\eta$")
    fig.tight_layout()
    save(fig, "acceptance_correctness_vs_eta")


# ---------------------------------------------------------------------------
# Artefact 3: Fourier weight attenuation
# ---------------------------------------------------------------------------

def plot_fourier_weight_attenuation(tables: dict) -> None:
    """Median accumulated weight vs eta with theoretical (1-2*eta)^2 overlay."""
    ns = tables["ns"]
    etas = tables["etas"]
    mw = tables["median_weight"]
    mt = tables["median_threshold"]
    epsilon = tables["epsilon"]

    # Pick a representative n (middle of range)
    n_rep = ns[len(ns) // 2]

    eta_arr = np.array(etas)
    weight_vals = np.array([mw.get((n_rep, e), 0.0) for e in etas])
    thresh_vals = np.array([mt.get((n_rep, e), 0.0) for e in etas])

    # Theoretical curve: (1-2*eta)^2
    eta_fine = np.linspace(0, 0.5, 200)
    theory_weight = (1 - 2 * eta_fine) ** 2
    # Theoretical threshold: (1-2*eta)^2 - eps^2/8
    theory_threshold = (1 - 2 * eta_fine) ** 2 - epsilon**2 / 8

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Theory curves
    ax.plot(eta_fine, theory_weight, "-", color="grey", alpha=0.6, linewidth=2,
            label=r"Theory: $(1-2\eta)^2$")
    ax.plot(eta_fine, theory_threshold, "--", color="grey", alpha=0.6, linewidth=2,
            label=r"Theory threshold: $(1-2\eta)^2 - \varepsilon^2/8$")

    # Empirical
    ax.plot(eta_arr, weight_vals, "o-", color=sns.color_palette("colorblind")[0],
            markersize=5, linewidth=1.5,
            label=f"Median weight ($n={n_rep}$)")
    ax.plot(eta_arr, thresh_vals, "s--", color=sns.color_palette("colorblind")[1],
            markersize=5, linewidth=1.5,
            label=f"Acceptance threshold ($n={n_rep}$)")

    # Also plot weight for a couple more n values
    for extra_n, marker, cidx in [(ns[0], "^", 2), (ns[-1], "v", 3)]:
        wv = [mw.get((extra_n, e), 0.0) for e in etas]
        ax.plot(eta_arr, wv, f"{marker}-", color=sns.color_palette("colorblind")[cidx],
                markersize=4, linewidth=1, alpha=0.7,
                label=f"Median weight ($n={extra_n}$)")

    # Mark crossing region
    ax.axhline(0, ls="-", color="black", alpha=0.3, linewidth=0.5)
    ax.set_xlabel(r"Noise rate $\eta$")
    ax.set_ylabel("Fourier weight")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(-0.01, max(etas) + 0.01)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(r"Fourier weight attenuation under label-flip noise")
    fig.tight_layout()
    save(fig, "fourier_weight_attenuation")


# ---------------------------------------------------------------------------
# Artefact 4: Breakdown points table (CSV)
# ---------------------------------------------------------------------------

def write_breakdown_table(tables: dict) -> None:
    """CSV: per-n empirical eta where acceptance < 50%, theoretical eta threshold."""
    ns = tables["ns"]
    etas = tables["etas"]
    acc = tables["acceptance_rate"]
    epsilon = tables["epsilon"]

    # Theoretical: (1-2*eta)^2 = eps^2/8 => eta = (1 - eps/(2*sqrt(2))) / 2
    eta_theory = (1 - epsilon / (2 * math.sqrt(2))) / 2

    path = OUT_DIR / "breakdown_points.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n", "empirical_eta_50pct", "theoretical_eta_threshold",
            "acceptance_at_max_eta", "match_within_0.05",
        ])
        for n in ns:
            # Find empirical breakdown: first eta where acceptance < 0.5
            empirical_eta = None
            for eta in etas:
                rate = acc.get((n, eta), 1.0)
                if rate < 0.5:
                    empirical_eta = eta
                    break

            acc_at_max = acc.get((n, etas[-1]), 0.0)
            match = ""
            if empirical_eta is not None:
                match = "yes" if abs(empirical_eta - eta_theory) <= 0.05 else "no"
            else:
                match = "no_breakdown"

            writer.writerow([
                n,
                f"{empirical_eta:.2f}" if empirical_eta is not None else "none",
                f"{eta_theory:.4f}",
                f"{acc_at_max:.2f}",
                match,
            ])
    print(f"  {path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(tables: dict) -> None:
    """Print a summary of key findings to stdout."""
    ns = tables["ns"]
    etas = tables["etas"]
    acc = tables["acceptance_rate"]
    corr = tables["correctness_rate"]
    mw = tables["median_weight"]
    epsilon = tables["epsilon"]

    eta_theory = (1 - epsilon / (2 * math.sqrt(2))) / 2

    print("\n" + "=" * 70)
    print("NOISE SWEEP ANALYSIS REPORT")
    print("=" * 70)

    print(f"\nExperiment: n in {ns[0]}..{ns[-1]}, "
          f"eta in {{{', '.join(f'{e:.2f}' for e in etas)}}}")
    print(f"Epsilon = {epsilon}, theoretical eta_max = {eta_theory:.4f}")
    print(f"  (from (1-2*eta)^2 = eps^2/8)")

    # 1. Breakdown analysis
    print("\n--- Breakdown Analysis ---")
    for n in ns:
        breakdown_eta = None
        for eta in etas:
            if acc.get((n, eta), 1.0) < 0.5:
                breakdown_eta = eta
                break
        if breakdown_eta is not None:
            print(f"  n={n:2d}: acceptance < 50% at eta={breakdown_eta:.2f}"
                  f"  (theory: {eta_theory:.3f},"
                  f" diff={abs(breakdown_eta - eta_theory):.3f})")
        else:
            print(f"  n={n:2d}: acceptance never drops below 50%"
                  f" (min acc={min(acc.get((n, e), 1.0) for e in etas)*100:.1f}%)")

    # 2. N-dependence
    print("\n--- N-dependence at fixed eta ---")
    for eta in [0.2, 0.3, 0.4]:
        rates = [acc.get((n, eta), 0.0) for n in ns]
        print(f"  eta={eta:.1f}: acceptance range = "
              f"{min(rates)*100:.0f}%--{max(rates)*100:.0f}%"
              f" (spread={abs(max(rates)-min(rates))*100:.1f}pp)")

    # 3. Degradation character
    print("\n--- Degradation Character ---")
    n_mid = ns[len(ns) // 2]
    prev_rate = 1.0
    for eta in etas:
        rate = acc.get((n_mid, eta), 0.0)
        drop = prev_rate - rate
        marker = " <-- large drop" if drop > 0.2 else ""
        print(f"  n={n_mid}, eta={eta:.2f}: acceptance={rate*100:5.1f}%"
              f"  (delta={-drop*100:+.1f}pp){marker}")
        prev_rate = rate

    # 4. Weight attenuation
    print("\n--- Weight Attenuation vs Theory ---")
    for n in [ns[0], ns[len(ns)//2], ns[-1]]:
        print(f"  n={n}:")
        for eta in etas:
            observed = mw.get((n, eta), 0.0)
            theory = (1 - 2 * eta) ** 2
            rel_err = abs(observed - theory) / theory if theory > 0.01 else float("inf")
            print(f"    eta={eta:.2f}: weight={observed:.4f},"
                  f" theory={(1-2*eta)**2:.4f},"
                  f" rel_err={rel_err*100:.1f}%")

    # 5. Correctness
    print("\n--- Correctness Under Noise ---")
    for n in [ns[0], ns[len(ns)//2], ns[-1]]:
        for eta in [0.0, 0.2, 0.4]:
            c = corr.get((n, eta), 0.0)
            a = acc.get((n, eta), 0.0)
            print(f"  n={n:2d}, eta={eta:.1f}: correctness={c*100:.0f}%,"
                  f" acceptance={a*100:.0f}%")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_style()

    print("Loading noise sweep data...")
    data = load_data()
    trials = data["trials"]
    params = data["parameters"]
    epsilon = params["epsilon"]
    print(f"  {len(trials)} trials, n in {params['nRange'][0]}..{params['nRange'][-1]}, "
          f"eta in {params['noiseRates']}, epsilon={epsilon}")

    tables = build_tables(trials, epsilon)

    print("\nGenerating artefacts:")
    plot_noise_heatmap(tables)
    plot_acceptance_correctness_vs_eta(tables)
    plot_fourier_weight_attenuation(tables)
    write_breakdown_table(tables)

    print_report(tables)
    print("\nDone.")


if __name__ == "__main__":
    main()
