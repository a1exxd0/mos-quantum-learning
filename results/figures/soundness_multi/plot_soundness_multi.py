"""Generate all Soundness Multi artefacts.

Artefacts produced:
  1. rejection_vs_n_by_k.{pdf,png}         -- Faceted 2x2: rejection rate vs n, one line per k
  2. comparison_single_vs_multi.{pdf,png}   -- Grouped bar: single-element vs multi-element
  3. soundness_multi_summary.csv            -- Per (strategy, k, n) rejection rate + outcome breakdown

Usage:
    uv run python results/figures/soundness_multi/plot_soundness_multi.py
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
DATA_FILE = PROJECT_ROOT / "results" / "soundness_multi_4_16_100.pb"
SINGLE_DATA_FILE = PROJECT_ROOT / "results" / "soundness_4_20_100.pb"
OUT_DIR = SCRIPT_DIR

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


# Multi-element strategies
STRATEGY_LABELS = {
    "partial_real": "Partial real",
    "diluted_list": "Diluted list",
    "shifted_coefficients": "Shifted coefficients",
    "subset_plus_noise": "Subset + noise",
}

STRATEGY_ORDER = ["partial_real", "diluted_list", "shifted_coefficients", "subset_plus_noise"]

# Single-element strategies
SINGLE_STRATEGY_LABELS = {
    "random_list": "Random list",
    "wrong_parity": "Wrong parity",
    "partial_list": "Partial list",
    "inflated_list": "Inflated list",
}

# k inference: in the experiment code, theta = min(epsilon, max(0.01, (1/k)*0.9))
# For epsilon=0.3:  k=2 -> theta=0.3,  k=4 -> theta=0.225
THETA_TO_K = {0.3: 2, 0.225: 4}


def load_multi_data() -> dict:
    """Deserialise the soundness_multi .pb file."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.decode import decode
    return json.loads(decode(DATA_FILE))


def load_single_data() -> dict:
    """Deserialise the single-element soundness .pb file."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.decode import decode
    return json.loads(decode(SINGLE_DATA_FILE))


def infer_k(trial: dict) -> int:
    """Infer k from theta value (the k field was not populated in the proto)."""
    theta = trial["theta"]
    k = THETA_TO_K.get(theta)
    if k is None:
        raise ValueError(f"Unknown theta={theta}; cannot infer k")
    return k


def parse_strategy(phi: str) -> str:
    """Extract the strategy name from phiDescription."""
    return phi.replace("soundness_", "").replace("soundness_multi_", "")


def build_multi_tables(
    trials: list[dict],
) -> tuple[
    dict[tuple[str, int, int], float],                  # rejection_rate[(strategy, k, n)]
    dict[tuple[str, int, int], tuple[float, float]],    # ci[(strategy, k, n)]
    dict[tuple[str, int, int, str], int],               # mechanism_counts[(strategy, k, n, outcome)]
    list[int],                                          # sorted n values
    list[int],                                          # sorted k values
]:
    """Aggregate trial-level data into per-(strategy, k, n) summaries."""
    counts: dict[tuple[str, int, int], int] = defaultdict(int)
    rejects: dict[tuple[str, int, int], int] = defaultdict(int)
    mechanisms: dict[tuple[str, int, int, str], int] = defaultdict(int)

    for t in trials:
        strategy = parse_strategy(t["phiDescription"])
        k = infer_k(t)
        n = t["n"]
        key = (strategy, k, n)
        counts[key] += 1
        if not t.get("accepted", False):
            rejects[key] += 1
        mechanisms[(strategy, k, n, t["outcome"])] += 1

    rejection_rate: dict[tuple[str, int, int], float] = {}
    ci: dict[tuple[str, int, int], tuple[float, float]] = {}
    for key, total in counts.items():
        rej = rejects.get(key, 0)
        rejection_rate[key] = rej / total if total else 0
        ci[key] = wilson_ci(rej, total)

    ns = sorted({n for (_, _, n) in counts})
    ks = sorted({k for (_, k, _) in counts})
    return rejection_rate, ci, mechanisms, ns, ks


def build_single_tables(
    trials: list[dict],
) -> tuple[
    dict[tuple[str, int], float],          # rejection_rate[(strategy, n)]
    dict[tuple[str, int], tuple[float, float]],  # ci[(strategy, n)]
    list[int],                              # sorted n values
]:
    """Aggregate single-element soundness data."""
    counts: dict[tuple[str, int], int] = defaultdict(int)
    rejects: dict[tuple[str, int], int] = defaultdict(int)

    for t in trials:
        strategy = t["phiDescription"].replace("soundness_", "")
        n = t["n"]
        counts[(strategy, n)] += 1
        if not t.get("accepted", False):
            rejects[(strategy, n)] += 1

    rejection_rate: dict[tuple[str, int], float] = {}
    ci: dict[tuple[str, int], tuple[float, float]] = {}
    for key, total in counts.items():
        rej = rejects.get(key, 0)
        rejection_rate[key] = rej / total if total else 0
        ci[key] = wilson_ci(rej, total)

    ns = sorted({n for (_, n) in counts})
    return rejection_rate, ci, ns


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
# Artefact 1: Rejection vs n by k (2x2 faceted by strategy)
# ---------------------------------------------------------------------------

def plot_rejection_vs_n_by_k(
    rejection_rate: dict[tuple[str, int, int], float],
    ci: dict[tuple[str, int, int], tuple[float, float]],
    ns: list[int],
    ks: list[int],
) -> None:
    """2x2 faceted line plot: rejection rate vs n, one line per k."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
    colours = sns.color_palette("colorblind", len(ks))
    markers = ["o", "s", "D", "^"]

    for ax, strategy in zip(axes.flat, STRATEGY_ORDER):
        for ki, k in enumerate(ks):
            rates = [rejection_rate.get((strategy, k, n), 0) for n in ns]
            lo = [ci.get((strategy, k, n), (0, 0))[0] for n in ns]
            hi = [ci.get((strategy, k, n), (0, 0))[1] for n in ns]
            yerr_lo = [max(0, r - l) for r, l in zip(rates, lo)]
            yerr_hi = [max(0, h - r) for r, h in zip(rates, hi)]
            ax.errorbar(
                ns, rates,
                yerr=[yerr_lo, yerr_hi],
                label=f"$k={k}$",
                color=colours[ki],
                marker=markers[ki],
                markersize=4,
                linewidth=1.5,
                capsize=2,
            )

        # Theoretical guarantee line
        ax.axhline(0.9, ls="--", color="grey", alpha=0.6, linewidth=1)
        ax.set_title(STRATEGY_LABELS[strategy], fontsize=10)
        ax.set_ylim(-0.05, 1.08)

    axes[1, 0].set_xlabel("$n$")
    axes[1, 1].set_xlabel("$n$")
    axes[0, 0].set_ylabel("Rejection rate")
    axes[1, 0].set_ylabel("Rejection rate")

    # Shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], ls="--", color="grey", alpha=0.6))
    labels.append(r"$1-\delta=0.9$")
    fig.legend(handles, labels, loc="upper center", ncol=len(ks) + 1,
               fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Multi-element soundness: rejection rate vs $n$ by sparsity $k$",
                 y=1.06, fontsize=11)
    fig.tight_layout()
    save(fig, "rejection_vs_n_by_k")


# ---------------------------------------------------------------------------
# Artefact 2: Comparison with single-element soundness
# ---------------------------------------------------------------------------

def plot_comparison_single_vs_multi(
    multi_rate: dict[tuple[str, int, int], float],
    multi_ns: list[int],
    multi_ks: list[int],
    single_rate: dict[tuple[str, int], float],
    single_ns: list[int],
) -> None:
    """Grouped bar chart at representative n values: single vs multi rejection."""
    # Find overlapping n values
    common_ns = sorted(set(multi_ns) & set(single_ns))
    # Pick representative subset
    if len(common_ns) <= 5:
        ns_show = common_ns
    else:
        ns_show = common_ns[::3]
        if common_ns[-1] not in ns_show:
            ns_show.append(common_ns[-1])

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(ns_show))
    n_groups = 1 + len(multi_ks)  # single + one per k
    width = 0.8 / n_groups
    colours = sns.color_palette("colorblind", n_groups)

    # Single-element: pool all 4 strategies into one rejection rate per n
    single_strats = ["random_list", "wrong_parity", "partial_list", "inflated_list"]
    single_pooled = {}
    single_pooled_ci = {}
    for n in ns_show:
        total = 0
        rej = 0
        for s in single_strats:
            r = single_rate.get((s, n), 0)
            total += 1
            rej += r
        single_pooled[n] = rej / total if total else 0
        # For CI: use total trial count (100 trials * 4 strategies = 400 per n)
        rej_count = round(single_pooled[n] * 400)
        single_pooled_ci[n] = wilson_ci(rej_count, 400)

    rates_single = [single_pooled[n] for n in ns_show]
    lo_s = [single_pooled_ci[n][0] for n in ns_show]
    hi_s = [single_pooled_ci[n][1] for n in ns_show]
    yerr_lo_s = [max(0, r - l) for r, l in zip(rates_single, lo_s)]
    yerr_hi_s = [max(0, h - r) for r, h in zip(rates_single, hi_s)]

    offset = (0 - n_groups / 2 + 0.5) * width
    ax.bar(
        x + offset, rates_single, width * 0.9,
        label="Single-element ($k=1$)",
        color=colours[0],
        yerr=[yerr_lo_s, yerr_hi_s],
        capsize=2, edgecolor="white", linewidth=0.5,
    )

    # Multi-element: pool all 4 multi-strategies for each k
    for ki, k in enumerate(multi_ks):
        pooled = {}
        pooled_ci = {}
        for n in ns_show:
            total = 0
            rej = 0
            for s in STRATEGY_ORDER:
                r = multi_rate.get((s, k, n), 0)
                total += 1
                rej += r
            pooled[n] = rej / total if total else 0
            rej_count = round(pooled[n] * 400)
            pooled_ci[n] = wilson_ci(rej_count, 400)

        rates = [pooled[n] for n in ns_show]
        lo = [pooled_ci[n][0] for n in ns_show]
        hi = [pooled_ci[n][1] for n in ns_show]
        yerr_lo = [max(0, r - l) for r, l in zip(rates, lo)]
        yerr_hi = [max(0, h - r) for r, h in zip(rates, hi)]

        offset = (ki + 1 - n_groups / 2 + 0.5) * width
        ax.bar(
            x + offset, rates, width * 0.9,
            label=f"Multi-element ($k={k}$)",
            color=colours[ki + 1],
            yerr=[yerr_lo, yerr_hi],
            capsize=2, edgecolor="white", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"$n={n}$" for n in ns_show])
    ax.set_ylabel("Rejection rate (pooled across strategies)")
    ax.set_ylim(0, 1.08)
    ax.axhline(0.9, ls="--", color="grey", alpha=0.6, label=r"$1-\delta=0.9$")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.set_title("Single-element vs multi-element soundness")
    save(fig, "comparison_single_vs_multi")


# ---------------------------------------------------------------------------
# Artefact 3: Summary table (CSV)
# ---------------------------------------------------------------------------

def write_summary_csv(
    rejection_rate: dict[tuple[str, int, int], float],
    ci: dict[tuple[str, int, int], tuple[float, float]],
    mechanisms: dict[tuple[str, int, int, str], int],
    ns: list[int],
    ks: list[int],
) -> None:
    """Write CSV: per (strategy, k, n) rejection rate + outcome breakdown."""
    path = OUT_DIR / "soundness_multi_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "strategy", "k", "n",
            "rejection_rate", "ci_lower", "ci_upper",
            "reject_weight", "reject_list_size", "accept", "total",
        ])
        for strategy in STRATEGY_ORDER:
            for k in ks:
                for n in ns:
                    rate = rejection_rate.get((strategy, k, n), 0)
                    lo, hi = ci.get((strategy, k, n), (0, 0))
                    rej_weight = mechanisms.get((strategy, k, n, "reject_insufficient_weight"), 0)
                    rej_list = mechanisms.get((strategy, k, n, "reject_list_too_large"), 0)
                    acc = mechanisms.get((strategy, k, n, "accept"), 0)
                    total = rej_weight + rej_list + acc
                    writer.writerow([
                        STRATEGY_LABELS[strategy],
                        k, n,
                        f"{rate:.4f}", f"{lo:.4f}", f"{hi:.4f}",
                        rej_weight, rej_list, acc, total,
                    ])
    print(f"  {path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Analysis / reporting
# ---------------------------------------------------------------------------

def print_report(
    rejection_rate: dict[tuple[str, int, int], float],
    mechanisms: dict[tuple[str, int, int, str], int],
    ns: list[int],
    ks: list[int],
    single_rate: dict[tuple[str, int], float],
    single_ns: list[int],
) -> None:
    """Print analysis report to stdout."""
    print("\n" + "=" * 70)
    print("ANALYSIS REPORT: Multi-Element Soundness")
    print("=" * 70)

    # 1. Overall soundness per (strategy, k)
    print("\n--- Rejection rates per (strategy, k), averaged across n ---")
    any_violation = False
    for strategy in STRATEGY_ORDER:
        for k in ks:
            rates = [rejection_rate.get((strategy, k, n), 0) for n in ns]
            mean_rate = np.mean(rates)
            min_rate = min(rates)
            min_n = ns[rates.index(min_rate)]
            print(f"  {STRATEGY_LABELS[strategy]:25s}  k={k}: "
                  f"mean={mean_rate:.3f}, min={min_rate:.3f} (at n={min_n})")
            if min_rate < 0.9:
                any_violation = True

    if any_violation:
        print("\n  ** Some (strategy, k, n) combinations have rejection < 1-delta=0.9 **")
    else:
        print("\n  All (strategy, k, n) combinations satisfy soundness >= 1-delta=0.9.")

    # 2. Does increasing k weaken rejection?
    print("\n--- Effect of k on rejection (mean across n, per strategy) ---")
    for strategy in STRATEGY_ORDER:
        for i, k1 in enumerate(ks):
            for k2 in ks[i + 1:]:
                rates1 = [rejection_rate.get((strategy, k1, n), 0) for n in ns]
                rates2 = [rejection_rate.get((strategy, k2, n), 0) for n in ns]
                diff = np.mean(rates2) - np.mean(rates1)
                direction = "stronger" if diff > 0 else "weaker" if diff < 0 else "same"
                print(f"  {STRATEGY_LABELS[strategy]:25s}: "
                      f"k={k1}->{k2}: {direction} ({diff:+.4f})")

    # 3. Rejection mechanism per strategy
    print("\n--- Dominant rejection mechanism per (strategy, k) ---")
    for strategy in STRATEGY_ORDER:
        for k in ks:
            total_weight = sum(mechanisms.get((strategy, k, n, "reject_insufficient_weight"), 0) for n in ns)
            total_list = sum(mechanisms.get((strategy, k, n, "reject_list_too_large"), 0) for n in ns)
            total_acc = sum(mechanisms.get((strategy, k, n, "accept"), 0) for n in ns)
            total = total_weight + total_list + total_acc
            if total == 0:
                continue
            print(f"  {STRATEGY_LABELS[strategy]:25s}  k={k}: "
                  f"weight={total_weight/total:.1%}, "
                  f"list-size={total_list/total:.1%}, "
                  f"accept={total_acc/total:.1%}")

    # 4. Boundary strategies at larger k
    print("\n--- Boundary strategies (partial_real, subset_plus_noise) at each k ---")
    for strategy in ["partial_real", "subset_plus_noise"]:
        for k in ks:
            rates = {n: rejection_rate.get((strategy, k, n), 0) for n in ns}
            weak_ns = [n for n, r in rates.items() if r < 0.95]
            if weak_ns:
                print(f"  {STRATEGY_LABELS[strategy]:25s}  k={k}: "
                      f"weaker at n={weak_ns} (rates: {[f'{rates[n]:.2f}' for n in weak_ns]})")
            else:
                print(f"  {STRATEGY_LABELS[strategy]:25s}  k={k}: "
                      f"rejection >= 0.95 at all n")

    # 5. Single vs multi comparison
    print("\n--- Single-element vs multi-element overall ---")
    # Single: pool across all strategies and overlapping n
    common_ns = sorted(set(ns) & set(single_ns))
    single_strats = ["random_list", "wrong_parity", "partial_list", "inflated_list"]

    single_all_rates = []
    for n in common_ns:
        for s in single_strats:
            single_all_rates.append(single_rate.get((s, n), 0))
    single_mean = np.mean(single_all_rates) if single_all_rates else 0

    multi_all_rates = []
    for n in common_ns:
        for k in ks:
            for s in STRATEGY_ORDER:
                multi_all_rates.append(rejection_rate.get((s, k, n), 0))
    multi_mean = np.mean(multi_all_rates) if multi_all_rates else 0

    print(f"  Single-element mean rejection (n in {common_ns[0]}..{common_ns[-1]}): {single_mean:.4f}")
    print(f"  Multi-element  mean rejection (n in {common_ns[0]}..{common_ns[-1]}): {multi_mean:.4f}")
    diff = multi_mean - single_mean
    if abs(diff) < 0.01:
        print("  Difference is negligible (<1 percentage point).")
    elif diff > 0:
        print(f"  Multi-element is HARDER to cheat ({diff:+.4f}).")
    else:
        print(f"  Multi-element is EASIER to cheat ({diff:+.4f}).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_style()

    print("Loading multi-element soundness data...")
    multi_data = load_multi_data()
    multi_trials = multi_data["trials"]
    multi_params = multi_data["parameters"]
    print(f"  {len(multi_trials)} trials, strategies={multi_params['strategies']}, "
          f"k in {multi_params['kRange']}, "
          f"n in {multi_params['nRange'][0]}..{multi_params['nRange'][-1]}")

    print("Loading single-element soundness data...")
    single_data = load_single_data()
    single_trials = single_data["trials"]
    print(f"  {len(single_trials)} trials")

    # Build tables
    multi_rr, multi_ci, multi_mech, multi_ns, multi_ks = build_multi_tables(multi_trials)
    single_rr, single_ci, single_ns = build_single_tables(single_trials)

    print("\nGenerating artefacts:")
    plot_rejection_vs_n_by_k(multi_rr, multi_ci, multi_ns, multi_ks)
    plot_comparison_single_vs_multi(multi_rr, multi_ns, multi_ks, single_rr, single_ns)
    write_summary_csv(multi_rr, multi_ci, multi_mech, multi_ns, multi_ks)

    # Analysis report
    print_report(multi_rr, multi_mech, multi_ns, multi_ks, single_rr, single_ns)

    print("\nDone.")


if __name__ == "__main__":
    main()
