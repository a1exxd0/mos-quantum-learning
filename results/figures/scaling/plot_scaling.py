"""Generate all Scaling experiment artefacts.

Artefacts produced:
  1. scaling_summary.csv             – Per-n baseline summary table
  2. completeness_vs_n.{pdf,png}     – Acceptance & correctness rate vs n
  3. postselection_vs_n.{pdf,png}    – Median postselection rate vs n
  4. resource_scaling.{pdf,png}       – Total copies & wall-clock time vs n
  5. list_size_vs_n.{pdf,png}        – Median list size vs n with IQR

Usage:
    uv run python results/figures/scaling/plot_scaling.py
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DATA_FILE = PROJECT_ROOT / "results" / "scaling_4_16_100.pb"
OUT_DIR = SCRIPT_DIR  # figures land next to the script


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95 % confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    half = z * math.sqrt((p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_data() -> dict:
    """Deserialise the scaling .pb file to a Python dict."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.decode import decode

    return json.loads(decode(DATA_FILE))


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
# Data aggregation
# ---------------------------------------------------------------------------

def aggregate(trials: list[dict], ns: list[int]) -> dict:
    """Build per-n aggregated statistics."""
    by_n: dict[int, list[dict]] = defaultdict(list)
    for t in trials:
        by_n[t["n"]].append(t)

    stats: dict[int, dict] = {}
    for n in ns:
        ts = by_n[n]
        total = len(ts)
        accepted = sum(1 for t in ts if t["accepted"])
        correct = sum(1 for t in ts if t["hypothesisCorrect"])

        list_sizes = sorted(t["listSize"] for t in ts)
        postsel_rates = sorted(t["postselectionRate"] for t in ts)
        total_copies = sorted(t["totalCopies"] for t in ts)
        total_times = sorted(t["totalTimeS"] for t in ts)

        def median(arr: list[float]) -> float:
            m = len(arr)
            if m % 2 == 1:
                return arr[m // 2]
            return (arr[m // 2 - 1] + arr[m // 2]) / 2

        def q1(arr: list[float]) -> float:
            m = len(arr)
            lower = arr[:m // 2]
            return median(lower) if lower else arr[0]

        def q3(arr: list[float]) -> float:
            m = len(arr)
            upper = arr[(m + 1) // 2:]
            return median(upper) if upper else arr[-1]

        stats[n] = {
            "total": total,
            "accepted": accepted,
            "correct": correct,
            "acceptance_rate": accepted / total,
            "correctness_rate": correct / total,
            "acceptance_ci": wilson_ci(accepted, total),
            "correctness_ci": wilson_ci(correct, total),
            "median_list_size": median(list_sizes),
            "q1_list_size": q1(list_sizes),
            "q3_list_size": q3(list_sizes),
            "median_postsel": median(postsel_rates),
            "q1_postsel": q1(postsel_rates),
            "q3_postsel": q3(postsel_rates),
            "median_copies": median(total_copies),
            "median_time": median(total_times),
        }
    return stats


# ---------------------------------------------------------------------------
# Artefact 1: Baseline summary table (CSV)
# ---------------------------------------------------------------------------

def write_summary_table(stats: dict[int, dict], ns: list[int]) -> None:
    """Write CSV: per-n acceptance %, correctness %, median |L|, etc."""
    path = OUT_DIR / "scaling_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n", "acceptance_pct", "acc_ci_lo", "acc_ci_hi",
            "correctness_pct", "corr_ci_lo", "corr_ci_hi",
            "median_list_size", "median_postselection_rate",
            "median_total_copies", "median_total_time_s",
        ])
        for n in ns:
            s = stats[n]
            writer.writerow([
                n,
                f"{s['acceptance_rate'] * 100:.1f}",
                f"{s['acceptance_ci'][0] * 100:.1f}",
                f"{s['acceptance_ci'][1] * 100:.1f}",
                f"{s['correctness_rate'] * 100:.1f}",
                f"{s['correctness_ci'][0] * 100:.1f}",
                f"{s['correctness_ci'][1] * 100:.1f}",
                f"{s['median_list_size']:.1f}",
                f"{s['median_postsel']:.4f}",
                f"{s['median_copies']:.0f}",
                f"{s['median_time']:.4f}",
            ])
    print(f"  {path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Artefact 2: Completeness vs n
# ---------------------------------------------------------------------------

def plot_completeness(stats: dict[int, dict], ns: list[int], delta: float) -> None:
    """Line plot: acceptance rate & correctness vs n with 95% Wilson CI."""
    fig, ax = plt.subplots(figsize=(6, 4))
    colours = sns.color_palette("colorblind", 3)

    # Acceptance rate
    acc_rates = [stats[n]["acceptance_rate"] for n in ns]
    acc_lo = [stats[n]["acceptance_ci"][0] for n in ns]
    acc_hi = [stats[n]["acceptance_ci"][1] for n in ns]
    ax.plot(ns, acc_rates, "o-", color=colours[0], label="Acceptance rate", markersize=5)
    ax.fill_between(ns, acc_lo, acc_hi, alpha=0.15, color=colours[0])

    # Correctness rate
    corr_rates = [stats[n]["correctness_rate"] for n in ns]
    corr_lo = [stats[n]["correctness_ci"][0] for n in ns]
    corr_hi = [stats[n]["correctness_ci"][1] for n in ns]
    ax.plot(ns, corr_rates, "s-", color=colours[1], label="Correctness rate", markersize=5)
    ax.fill_between(ns, corr_lo, corr_hi, alpha=0.15, color=colours[1])

    # Theoretical guarantee: 1 - delta
    ax.axhline(1 - delta, ls="--", color="grey", alpha=0.6,
               label=rf"$1-\delta = {1 - delta:.1f}$ (Thm 12)")

    ax.set_xlabel("$n$ (number of bits)")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.80, 1.02)
    ax.set_xticks(ns)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Completeness: acceptance and correctness vs $n$")
    save(fig, "completeness_vs_n")


# ---------------------------------------------------------------------------
# Artefact 3: Postselection rate vs n
# ---------------------------------------------------------------------------

def plot_postselection(stats: dict[int, dict], ns: list[int]) -> None:
    """Bar chart: median postselection rate per n."""
    fig, ax = plt.subplots(figsize=(6, 4))
    colours = sns.color_palette("colorblind", 2)

    medians = [stats[n]["median_postsel"] for n in ns]
    q1s = [stats[n]["q1_postsel"] for n in ns]
    q3s = [stats[n]["q3_postsel"] for n in ns]
    yerr_lo = [m - q for m, q in zip(medians, q1s)]
    yerr_hi = [q - m for m, q in zip(medians, q3s)]

    ax.bar(ns, medians, color=colours[0], edgecolor="white", linewidth=0.5,
           yerr=[yerr_lo, yerr_hi], capsize=3, width=0.7,
           label="Median (IQR)")

    # Theoretical prediction: 0.5
    ax.axhline(0.5, ls="--", color="grey", alpha=0.6,
               label=r"Theoretical $\frac{1}{2}$ (Thm 5)")

    ax.set_xlabel("$n$ (number of bits)")
    ax.set_ylabel("Postselection rate")
    ax.set_ylim(0.35, 0.6)
    ax.set_xticks(ns)
    ax.legend(fontsize=8)
    ax.set_title(r"Postselection rate vs $n$")
    save(fig, "postselection_vs_n")


# ---------------------------------------------------------------------------
# Artefact 4: Resource scaling (dual-axis, log-linear)
# ---------------------------------------------------------------------------

def plot_resource_scaling(
    stats: dict[int, dict], ns: list[int], params: dict,
) -> None:
    """Dual-axis line plot: median total copies & wall-clock time vs n."""
    fig, ax1 = plt.subplots(figsize=(6, 4))
    colours = sns.color_palette("colorblind", 3)

    copies = [stats[n]["median_copies"] for n in ns]
    times = [stats[n]["median_time"] for n in ns]

    # Primary axis: copies (log scale)
    ax1.plot(ns, copies, "o-", color=colours[0], label="Median total copies", markersize=5)
    ax1.set_xlabel("$n$ (number of bits)")
    ax1.set_ylabel("Total copies", color=colours[0])
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor=colours[0])
    ax1.set_xticks(ns)

    # Theoretical bound: copies ~ C * n * log(1/(delta*theta^2)) / theta^4
    # From Thm 12 with QFS shots + verifier samples
    delta = params.get("delta", 0.1)
    theta = params.get("epsilon", 0.3)  # theta = epsilon in functional case
    theory_factor = np.array([n_val * math.log(1.0 / (delta * theta**2)) / theta**4
                              for n_val in ns])
    # Fit constant C from data
    copies_arr = np.array(copies)
    C_fit = np.median(copies_arr / theory_factor)
    theory_copies = C_fit * theory_factor
    ax1.plot(ns, theory_copies, "--", color=colours[0], alpha=0.5,
             label=rf"$C \cdot n \log(1/\delta\theta^2)/\theta^4$, $C={C_fit:.1f}$")

    # Secondary axis: time
    ax2 = ax1.twinx()
    ax2.plot(ns, times, "s-", color=colours[2], label="Median wall-clock time (s)",
             markersize=5)
    ax2.set_ylabel("Wall-clock time (s)", color=colours[2])
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor=colours[2])

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

    ax1.set_title("Resource scaling vs $n$")
    save(fig, "resource_scaling")


# ---------------------------------------------------------------------------
# Artefact 5: List size vs n
# ---------------------------------------------------------------------------

def plot_list_size(stats: dict[int, dict], ns: list[int], params: dict) -> None:
    """Median list size vs n with IQR whiskers."""
    fig, ax = plt.subplots(figsize=(6, 4))
    colours = sns.color_palette("colorblind", 2)

    medians = [stats[n]["median_list_size"] for n in ns]
    q1s = [stats[n]["q1_list_size"] for n in ns]
    q3s = [stats[n]["q3_list_size"] for n in ns]
    yerr_lo = [m - q for m, q in zip(medians, q1s)]
    yerr_hi = [q - m for m, q in zip(medians, q3s)]

    ax.errorbar(ns, medians, yerr=[yerr_lo, yerr_hi], fmt="o-",
                color=colours[0], capsize=4, markersize=6,
                label=r"Median $|L|$ (IQR)")

    # Parseval bound on number of large Fourier coefficients: 4/theta^2
    # (distinct from the verifier list-size bound 64*b^2/theta^2 in Thm 12 Step 3)
    theta = params.get("epsilon", 0.3)
    upper_bound = 4.0 / theta**2
    ax.axhline(upper_bound, ls="--", color="grey", alpha=0.6,
               label=rf"$4/\theta^2 = {upper_bound:.1f}$ (Parseval bound)")

    # Expected for single parities: |L| = 1
    ax.axhline(1.0, ls=":", color=colours[1], alpha=0.7,
               label=r"Expected $|L|=1$ (single parity)")

    ax.set_xlabel("$n$ (number of bits)")
    ax.set_ylabel("List size $|L|$")
    ax.set_xticks(ns)
    # Set y-lim to show both the data (near 1) and the upper bound
    ax.set_ylim(0, upper_bound + 5)
    ax.legend(fontsize=8)
    ax.set_title(r"List size $|L|$ vs $n$")
    save(fig, "list_size_vs_n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_style()

    print("Loading scaling data...")
    data = load_data()
    trials = data["trials"]
    params = data["parameters"]
    ns = sorted(params["nRange"])
    print(f"  {len(trials)} trials, n in {ns[0]}..{ns[-1]}, "
          f"epsilon={params['epsilon']}, delta={params['delta']}")

    stats = aggregate(trials, ns)

    # Print quick summary
    print("\nPer-n summary:")
    print(f"  {'n':>3s}  {'acc%':>5s}  {'corr%':>5s}  {'|L|':>4s}  "
          f"{'postsel':>7s}  {'copies':>8s}  {'time(s)':>8s}")
    for n in ns:
        s = stats[n]
        print(f"  {n:3d}  {s['acceptance_rate']*100:5.1f}  "
              f"{s['correctness_rate']*100:5.1f}  "
              f"{s['median_list_size']:4.1f}  "
              f"{s['median_postsel']:7.4f}  "
              f"{s['median_copies']:8.0f}  "
              f"{s['median_time']:8.4f}")

    print("\nGenerating artefacts:")
    write_summary_table(stats, ns)
    plot_completeness(stats, ns, params["delta"])
    plot_postselection(stats, ns)
    plot_resource_scaling(stats, ns, params)
    plot_list_size(stats, ns, params)

    print("\nDone.")


if __name__ == "__main__":
    main()
