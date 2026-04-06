"""Generate all Truncation experiment artefacts.

Artefacts produced:
  1. heatmap_acceptance.{pdf,png}     – Small-multiple heatmaps per n
  2. sample_budget_knee.{pdf,png}     – Acceptance vs verifier_samples curves
  3. min_viable_budget.{pdf,png}      – Minimum budget for >=90% accept vs n
  4. truncation_summary.csv           – Per (n, epsilon) minimum viable budget

Usage:
    uv run python results/figures/truncation/plot_truncation.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_DIR = SCRIPT_DIR  # figures land next to the script

# All available n values for truncation data
ALL_N = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

# Experiment parameters
EPSILONS = [0.1, 0.2, 0.3, 0.4, 0.5]
VERIFIER_SAMPLES = [50, 100, 200, 500, 1000, 3000]
NOISE_RATE = 0.15
A_SQ = (1 - 2 * NOISE_RATE) ** 2  # 0.49


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


def load_all_data() -> list[dict]:
    """Load and combine all truncation .pb files."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.decode import decode

    all_trials = []
    loaded_ns = []
    for n in ALL_N:
        pb_file = RESULTS_DIR / f"truncation_{n}_{n}_100.pb"
        if not pb_file.exists():
            print(f"  Warning: {pb_file.name} not found, skipping n={n}")
            continue
        data = json.loads(decode(pb_file))
        all_trials.extend(data["trials"])
        loaded_ns.append(n)
        print(f"  Loaded n={n}: {len(data['trials'])} trials")

    print(f"  Total: {len(all_trials)} trials across n in {loaded_ns}")
    return all_trials, loaded_ns


def build_acceptance_table(
    trials: list[dict],
) -> tuple[
    dict[tuple[int, float, int], float],      # acceptance_rate[(n, eps, vsamp)]
    dict[tuple[int, float, int], tuple[float, float]],  # ci[(n, eps, vsamp)]
    dict[tuple[int, float, int], float],      # correctness_rate[(n, eps, vsamp)]
    dict[tuple[int, float, int], int],        # total count
]:
    """Aggregate per-(n, epsilon, verifier_samples) acceptance & correctness."""
    counts: dict[tuple[int, float, int], int] = defaultdict(int)
    accepts: dict[tuple[int, float, int], int] = defaultdict(int)
    corrects: dict[tuple[int, float, int], int] = defaultdict(int)

    for t in trials:
        n = t["n"]
        eps = t["epsilon"]
        vsamp = t["verifierSamples"]
        key = (n, eps, vsamp)
        counts[key] += 1
        if t.get("accepted", False):
            accepts[key] += 1
        if t.get("hypothesisCorrect", False):
            corrects[key] += 1

    acceptance_rate = {}
    correctness_rate = {}
    ci = {}
    for key, total in counts.items():
        acc = accepts.get(key, 0)
        cor = corrects.get(key, 0)
        acceptance_rate[key] = acc / total if total else 0
        correctness_rate[key] = cor / total if total else 0
        ci[key] = wilson_ci(acc, total)

    return acceptance_rate, ci, correctness_rate, counts


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
# Artefact 1: Heatmap grid (acceptance rate per n)
# ---------------------------------------------------------------------------

def plot_heatmaps(
    acceptance_rate: dict[tuple[int, float, int], float],
    ns: list[int],
) -> None:
    """Small-multiple heatmaps: one per n, axes = verifier_samples x epsilon."""
    # Select representative n values (up to 6)
    if len(ns) <= 6:
        ns_show = ns
    else:
        # Pick evenly spaced subset including endpoints
        ns_show = [ns[0]]
        for n in ns:
            if n in (4, 6, 8, 10, 12, 14):
                if n not in ns_show:
                    ns_show.append(n)
        ns_show = sorted(set(ns_show))

    ncols = min(3, len(ns_show))
    nrows = math.ceil(len(ns_show) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols + 1, figsize=(4.2 * ncols + 0.8, 3.5 * nrows),
        squeeze=False,
        gridspec_kw={"width_ratios": [1] * ncols + [0.05]},
    )

    im = None
    for idx, n in enumerate(ns_show):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        # Build matrix: rows = epsilon (top-to-bottom ascending), cols = verifier_samples
        matrix = np.full((len(EPSILONS), len(VERIFIER_SAMPLES)), np.nan)
        for i, eps in enumerate(EPSILONS):
            for j, vsamp in enumerate(VERIFIER_SAMPLES):
                key = (n, eps, vsamp)
                if key in acceptance_rate:
                    matrix[i, j] = acceptance_rate[key]

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                        vmin=0, vmax=1, interpolation="nearest")

        ax.set_xticks(range(len(VERIFIER_SAMPLES)))
        ax.set_xticklabels([str(v) for v in VERIFIER_SAMPLES], fontsize=7, rotation=45)
        ax.set_yticks(range(len(EPSILONS)))
        ax.set_yticklabels([f"{e:.1f}" for e in EPSILONS], fontsize=7)
        ax.set_title(f"$n = {n}$", fontsize=10)

        if col == 0:
            ax.set_ylabel(r"$\varepsilon$")
        if row == nrows - 1:
            ax.set_xlabel("Verifier samples")

        # Annotate cells with values
        for i in range(len(EPSILONS)):
            for j in range(len(VERIFIER_SAMPLES)):
                val = matrix[i, j]
                if not np.isnan(val):
                    colour = "white" if val < 0.5 else "black"
                    ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                            fontsize=6, color=colour, fontweight="bold")

    # Hide unused data axes
    for idx in range(len(ns_show), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # Shared colourbar in the rightmost column
    if im is not None:
        # Merge the rightmost-column axes into a single colourbar axis
        cbar_ax = axes[0][-1]
        for r in range(1, nrows):
            axes[r][-1].set_visible(False)
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Acceptance rate")

    fig.suptitle(
        r"Truncation: acceptance rate by $(\varepsilon,\,\mathrm{verifier\_samples})$"
        f"\n($\\eta = {NOISE_RATE}$, $a^2 = b^2 = {A_SQ:.2f}$)",
        fontsize=12, y=1.02,
    )
    fig.subplots_adjust(wspace=0.35, hspace=0.45)
    save(fig, "heatmap_acceptance")


# ---------------------------------------------------------------------------
# Artefact 2: Sample budget knee
# ---------------------------------------------------------------------------

def plot_sample_budget_knee(
    acceptance_rate: dict[tuple[int, float, int], float],
    ci: dict[tuple[int, float, int], tuple[float, float]],
    ns: list[int],
) -> None:
    """For each epsilon, plot acceptance rate vs verifier_samples for multiple n."""
    # Use eps=0.3 and eps=0.5 as two representative values
    eps_values = [0.3, 0.5]
    fig, axes = plt.subplots(1, len(eps_values), figsize=(6 * len(eps_values), 4.5),
                              sharey=True)
    if len(eps_values) == 1:
        axes = [axes]

    colours = sns.color_palette("viridis", len(ns))

    for ax, eps in zip(axes, eps_values):
        for i, n in enumerate(ns):
            rates = []
            ci_lo = []
            ci_hi = []
            for vsamp in VERIFIER_SAMPLES:
                key = (n, eps, vsamp)
                r = acceptance_rate.get(key, 0)
                lo, hi = ci.get(key, (0, 0))
                rates.append(r)
                ci_lo.append(max(0, r - lo))
                ci_hi.append(max(0, hi - r))

            ax.errorbar(
                VERIFIER_SAMPLES, rates,
                yerr=[ci_lo, ci_hi],
                label=f"$n={n}$",
                marker="o", markersize=3, linewidth=1.2,
                color=colours[i], capsize=2,
            )

        ax.axhline(0.9, ls="--", color="grey", alpha=0.7, label=r"90% threshold")
        ax.set_xscale("log")
        ax.set_xlabel("Verifier samples")
        ax.set_ylabel("Acceptance rate")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"$\\varepsilon = {eps}$")
        ax.legend(fontsize=6, ncol=2, loc="lower right")

    fig.suptitle(
        "Truncation: acceptance rate vs verifier sample budget\n"
        f"($\\eta = {NOISE_RATE}$, $a^2 = {A_SQ:.2f}$)",
        fontsize=12,
    )
    fig.tight_layout()
    save(fig, "sample_budget_knee")


# ---------------------------------------------------------------------------
# Artefact 3: Minimum viable budget vs n
# ---------------------------------------------------------------------------

def find_knee_points(
    acceptance_rate: dict[tuple[int, float, int], float],
    ns: list[int],
    threshold: float = 0.90,
) -> dict[tuple[int, float], int | None]:
    """For each (n, eps), find smallest verifier_samples with acceptance >= threshold."""
    knees = {}
    for n in ns:
        for eps in EPSILONS:
            knee = None
            for vsamp in VERIFIER_SAMPLES:
                key = (n, eps, vsamp)
                if acceptance_rate.get(key, 0) >= threshold:
                    knee = vsamp
                    break
            knees[(n, eps)] = knee
    return knees


def plot_min_viable_budget(
    knees: dict[tuple[int, float], int | None],
    acceptance_rate: dict[tuple[int, float, int], float],
    ns: list[int],
) -> None:
    """Two-panel plot: (a) min budget vs n, (b) max acceptance at budget=3000 vs n."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colours = sns.color_palette("colorblind", len(EPSILONS))

    # --- Panel (a): Minimum viable budget ---
    for i, eps in enumerate(EPSILONS):
        ns_plot = []
        budgets = []
        for n in ns:
            knee = knees.get((n, eps))
            if knee is not None:
                ns_plot.append(n)
                budgets.append(knee)

        if ns_plot:
            ax1.plot(ns_plot, budgets, marker="s", markersize=5, linewidth=1.5,
                     color=colours[i], label=f"$\\varepsilon = {eps}$")

    ax1.axhline(3000, ls=":", color="grey", alpha=0.5,
                label="Max tested budget")
    ax1.set_xlabel("$n$ (number of qubits)")
    ax1.set_ylabel("Minimum verifier samples\nfor $\\geq 90\\%$ acceptance")
    ax1.set_yscale("log")
    ax1.set_xticks(ns)
    ax1.set_ylim(30, 5000)
    ax1.legend(fontsize=7, loc="upper left")
    ax1.set_title("(a) Minimum viable budget", fontsize=10)

    # --- Panel (b): Max acceptance at largest budget vs n ---
    for i, eps in enumerate(EPSILONS):
        max_acc = []
        for n in ns:
            best = max(acceptance_rate.get((n, eps, v), 0) for v in VERIFIER_SAMPLES)
            max_acc.append(best)

        ax2.plot(ns, max_acc, marker="o", markersize=4, linewidth=1.5,
                 color=colours[i], label=f"$\\varepsilon = {eps}$")

    ax2.axhline(0.9, ls="--", color="grey", alpha=0.6, label="90% threshold")
    ax2.set_xlabel("$n$ (number of qubits)")
    ax2.set_ylabel("Best acceptance rate\n(over all budgets)")
    ax2.set_xticks(ns)
    ax2.set_ylim(0.4, 1.05)
    ax2.legend(fontsize=7, loc="lower left")
    ax2.set_title("(b) Best achievable acceptance", fontsize=10)

    fig.suptitle(
        "Verifier sample budget constraints vs problem dimension\n"
        f"($\\eta = {NOISE_RATE}$, $a^2 = {A_SQ:.2f}$)",
        fontsize=11, y=1.03,
    )
    fig.tight_layout()
    save(fig, "min_viable_budget")


# ---------------------------------------------------------------------------
# Artefact 4: Summary CSV
# ---------------------------------------------------------------------------

def write_summary_csv(
    knees: dict[tuple[int, float], int | None],
    acceptance_rate: dict[tuple[int, float, int], float],
    correctness_rate: dict[tuple[int, float, int], float],
    ns: list[int],
) -> None:
    """Write CSV: per (n, epsilon) minimum budget and max acceptance/correctness."""
    path = OUT_DIR / "truncation_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n", "epsilon", "min_budget_90pct", "threshold_tau",
            "accept_at_50", "accept_at_3000",
            "correct_at_50", "correct_at_3000",
        ])
        for n in ns:
            for eps in EPSILONS:
                knee = knees.get((n, eps))
                tau = A_SQ - eps**2 / 8
                a50 = acceptance_rate.get((n, eps, 50), 0)
                a3000 = acceptance_rate.get((n, eps, 3000), 0)
                c50 = correctness_rate.get((n, eps, 50), 0)
                c3000 = correctness_rate.get((n, eps, 3000), 0)
                writer.writerow([
                    n,
                    f"{eps:.1f}",
                    knee if knee is not None else "never",
                    f"{tau:.4f}",
                    f"{a50:.2f}",
                    f"{a3000:.2f}",
                    f"{c50:.2f}",
                    f"{c3000:.2f}",
                ])
    print(f"  {path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Analysis report
# ---------------------------------------------------------------------------

def print_analysis(
    acceptance_rate: dict[tuple[int, float, int], float],
    correctness_rate: dict[tuple[int, float, int], float],
    knees: dict[tuple[int, float], int | None],
    ns: list[int],
) -> None:
    """Print key findings to stdout."""
    print("\n" + "=" * 72)
    print("TRUNCATION EXPERIMENT ANALYSIS")
    print("=" * 72)

    # 1. Minimum budget per n
    print("\n--- Minimum verifier budget for >=90% acceptance ---")
    for n in ns:
        budgets_str = []
        for eps in EPSILONS:
            knee = knees.get((n, eps))
            budgets_str.append(f"eps={eps}: {knee if knee else 'never'}")
        print(f"  n={n:2d}: {', '.join(budgets_str)}")

    # 2. Correctness vs acceptance correlation
    print("\n--- Correctness tracks acceptance? ---")
    mismatch_count = 0
    total_keys = 0
    for key in acceptance_rate:
        total_keys += 1
        acc = acceptance_rate[key]
        cor = correctness_rate.get(key, 0)
        if abs(acc - cor) > 0.05:
            mismatch_count += 1
    print(f"  Keys with >5pp gap between acceptance and correctness: "
          f"{mismatch_count}/{total_keys}")

    # 3. Epsilon below which protocol always fails
    print("\n--- Epsilon below which acceptance never reaches 90% ---")
    for n in ns:
        failing_eps = []
        for eps in EPSILONS:
            max_accept = max(
                acceptance_rate.get((n, eps, v), 0) for v in VERIFIER_SAMPLES
            )
            if max_accept < 0.90:
                failing_eps.append(eps)
        if failing_eps:
            print(f"  n={n:2d}: eps in {failing_eps} never reach 90%")
        else:
            print(f"  n={n:2d}: all eps values can reach 90%")

    # 4. Tight margin analysis
    print("\n--- Tight margin analysis (a^2=0.49 vs tau=a^2-eps^2/8) ---")
    for eps in EPSILONS:
        tau = A_SQ - eps**2 / 8
        margin = A_SQ - tau
        print(f"  eps={eps}: tau={tau:.4f}, margin={margin:.4f}, "
              f"margin/a^2={margin/A_SQ:.2%}")

    # 5. Budget scaling with n
    print("\n--- Budget scaling with n (for eps=0.5) ---")
    eps_ref = 0.5
    prev_knee = None
    for n in ns:
        knee = knees.get((n, eps_ref))
        if knee is not None and prev_knee is not None:
            ratio = knee / prev_knee if prev_knee > 0 else float("inf")
            print(f"  n={n:2d}: budget={knee}, ratio to prev={ratio:.2f}")
        elif knee is not None:
            print(f"  n={n:2d}: budget={knee}")
        else:
            print(f"  n={n:2d}: never reaches 90%")
        prev_knee = knee

    # 6. Acceptance direction vs sample count (normal statistical convergence)
    print("\n--- Acceptance vs sample count direction ---")
    for eps in [0.1, 0.3, 0.5]:
        increasing = 0
        decreasing = 0
        for n in ns:
            r50 = acceptance_rate.get((n, eps, 50), 0)
            r3000 = acceptance_rate.get((n, eps, 3000), 0)
            if r3000 > r50 + 0.05:
                increasing += 1
            elif r50 > r3000 + 0.05:
                decreasing += 1
        print(f"  eps={eps}: {increasing} n-values increase with budget, "
              f"{decreasing} decrease, {len(ns)-increasing-decreasing} flat")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_style()

    print("Loading truncation data...")
    trials, ns = load_all_data()

    print("\nBuilding acceptance tables...")
    acceptance_rate, ci, correctness_rate, counts = build_acceptance_table(trials)

    print("\nFinding knee points...")
    knees = find_knee_points(acceptance_rate, ns)

    print("\nGenerating artefacts:")
    plot_heatmaps(acceptance_rate, ns)
    plot_sample_budget_knee(acceptance_rate, ci, ns)
    plot_min_viable_budget(knees, acceptance_rate, ns)
    write_summary_csv(knees, acceptance_rate, correctness_rate, ns)

    print_analysis(acceptance_rate, correctness_rate, knees, ns)

    print("\nDone.")


if __name__ == "__main__":
    main()
