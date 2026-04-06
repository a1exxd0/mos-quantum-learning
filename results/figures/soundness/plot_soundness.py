"""Generate all Exp 1 (Soundness) artefacts from PRESENTATION_PLAN.md.

Artefacts produced:
  1. rejection_by_strategy.{pdf,png}  – Grouped bar chart of rejection rate
  2. rejection_mechanism.{pdf,png}    – Stacked bar: list-size vs weight-check
  3. soundness_summary.csv            – Per (strategy, n) rejection rate + 95% CI

Usage:
    uv run python results/figures/soundness/plot_soundness.py
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
DATA_FILE = PROJECT_ROOT / "results" / "soundness_4_20_100.pb"
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


STRATEGY_LABELS = {
    "random_list": "Random list",
    "wrong_parity": "Wrong parity",
    "partial_list": "Partial list",
    "inflated_list": "Inflated list",
}

STRATEGY_ORDER = ["random_list", "wrong_parity", "partial_list", "inflated_list"]


def load_data() -> dict:
    """Deserialise the soundness .pb file to a Python dict."""
    # Import here so the script can be run standalone
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.decode import decode

    return json.loads(decode(DATA_FILE))


def build_tables(
    trials: list[dict],
) -> tuple[
    dict[tuple[str, int], float],          # rejection_rate[(strategy, n)]
    dict[tuple[str, int], tuple[float, float]],  # ci[(strategy, n)]
    dict[tuple[str, int, str], int],       # mechanism_counts[(strategy, n, outcome)]
    list[int],                              # sorted n values
]:
    """Aggregate trial-level data into per-(strategy, n) summaries."""
    counts: dict[tuple[str, int], int] = defaultdict(int)
    rejects: dict[tuple[str, int], int] = defaultdict(int)
    mechanisms: dict[tuple[str, int, str], int] = defaultdict(int)

    for t in trials:
        strategy = t["phiDescription"].replace("soundness_", "")
        n = t["n"]
        counts[(strategy, n)] += 1
        if not t.get("accepted", False):
            rejects[(strategy, n)] += 1
        mechanisms[(strategy, n, t["outcome"])] += 1

    rejection_rate: dict[tuple[str, int], float] = {}
    ci: dict[tuple[str, int], tuple[float, float]] = {}
    for key, total in counts.items():
        rej = rejects.get(key, 0)
        rejection_rate[key] = rej / total if total else 0
        ci[key] = wilson_ci(rej, total)

    ns = sorted({n for (_, n) in counts})
    return rejection_rate, ci, mechanisms, ns


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
# Artefact 1: Rejection by strategy (grouped bar chart)
# ---------------------------------------------------------------------------

def plot_rejection_by_strategy(
    rejection_rate: dict[tuple[str, int], float],
    ci: dict[tuple[str, int], tuple[float, float]],
    ns: list[int],
) -> None:
    """Grouped bar chart: rejection rate by strategy, bars for select n."""
    # Pick representative n values to keep the chart readable
    if len(ns) <= 6:
        ns_show = ns
    else:
        ns_show = ns[::4]  # every 4th
        if ns[-1] not in ns_show:
            ns_show.append(ns[-1])

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(STRATEGY_ORDER))
    width = 0.8 / len(ns_show)
    colours = sns.color_palette("colorblind", len(ns_show))

    for i, n in enumerate(ns_show):
        rates = [rejection_rate.get((s, n), 0) for s in STRATEGY_ORDER]
        lo = [ci.get((s, n), (0, 0))[0] for s in STRATEGY_ORDER]
        hi = [ci.get((s, n), (0, 0))[1] for s in STRATEGY_ORDER]
        yerr_lo = [max(0, r - l) for r, l in zip(rates, lo)]
        yerr_hi = [max(0, h - r) for r, h in zip(rates, hi)]
        offset = (i - len(ns_show) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            rates,
            width * 0.9,
            label=f"$n={n}$",
            color=colours[i],
            yerr=[yerr_lo, yerr_hi],
            capsize=2,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGY_ORDER])
    ax.set_ylabel("Rejection rate")
    ax.set_ylim(0, 1.08)
    ax.axhline(0.9, ls="--", color="grey", alpha=0.6, label=r"$1-\delta=0.9$")
    ax.legend(fontsize=8, ncol=3, loc="lower right")
    ax.set_title("Soundness: rejection rate by adversarial strategy")
    save(fig, "rejection_by_strategy")


# ---------------------------------------------------------------------------
# Artefact 2: Rejection mechanism breakdown (stacked bar)
# ---------------------------------------------------------------------------

def plot_rejection_mechanism(
    mechanisms: dict[tuple[str, int, str], int],
    ns: list[int],
) -> None:
    """Stacked bar chart: proportion rejected by list-size vs weight check."""
    fig, ax = plt.subplots(figsize=(7, 4))

    # Aggregate across all n for each strategy
    strategy_totals: dict[str, int] = defaultdict(int)
    strategy_list_rej: dict[str, int] = defaultdict(int)
    strategy_weight_rej: dict[str, int] = defaultdict(int)
    strategy_accept: dict[str, int] = defaultdict(int)

    for (strategy, n, outcome), count in mechanisms.items():
        strategy_totals[strategy] += count
        if outcome == "reject_list_too_large":
            strategy_list_rej[strategy] += count
        elif outcome == "reject_insufficient_weight":
            strategy_weight_rej[strategy] += count
        else:
            strategy_accept[strategy] += count

    x = np.arange(len(STRATEGY_ORDER))
    totals = [strategy_totals[s] for s in STRATEGY_ORDER]

    weight_frac = [strategy_weight_rej[s] / totals[i] for i, s in enumerate(STRATEGY_ORDER)]
    list_frac = [strategy_list_rej[s] / totals[i] for i, s in enumerate(STRATEGY_ORDER)]
    accept_frac = [strategy_accept[s] / totals[i] for i, s in enumerate(STRATEGY_ORDER)]

    c = sns.color_palette("colorblind", 3)
    ax.bar(x, weight_frac, label="Reject: insufficient weight", color=c[0], edgecolor="white", linewidth=0.5)
    ax.bar(x, list_frac, bottom=weight_frac, label="Reject: list too large", color=c[1], edgecolor="white", linewidth=0.5)
    bottom2 = [w + l for w, l in zip(weight_frac, list_frac)]
    ax.bar(x, accept_frac, bottom=bottom2, label="Accept", color=c[2], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGY_ORDER])
    ax.set_ylabel("Proportion of trials")
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Soundness: rejection mechanism breakdown (all $n$ pooled)")

    # Annotate percentages
    for i, s in enumerate(STRATEGY_ORDER):
        rej_pct = (1 - accept_frac[i]) * 100
        ax.text(i, 0.5, f"{rej_pct:.1f}%\nrejected", ha="center", va="center", fontsize=8, fontweight="bold")

    save(fig, "rejection_mechanism")


# ---------------------------------------------------------------------------
# Artefact 2b: Rejection mechanism by n (per-strategy faceted)
# ---------------------------------------------------------------------------

def plot_rejection_mechanism_by_n(
    mechanisms: dict[tuple[str, int, str], int],
    ns: list[int],
) -> None:
    """Faceted stacked bar: mechanism breakdown per n for each strategy."""
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5), sharex=True, sharey=True)
    c = sns.color_palette("colorblind", 3)

    for ax, strategy in zip(axes.flat, STRATEGY_ORDER):
        weight_fracs = []
        list_fracs = []
        accept_fracs = []
        for n in ns:
            total = sum(mechanisms.get((strategy, n, o), 0) for o in
                        ["accept", "reject_insufficient_weight", "reject_list_too_large"])
            if total == 0:
                weight_fracs.append(0)
                list_fracs.append(0)
                accept_fracs.append(0)
                continue
            weight_fracs.append(mechanisms.get((strategy, n, "reject_insufficient_weight"), 0) / total)
            list_fracs.append(mechanisms.get((strategy, n, "reject_list_too_large"), 0) / total)
            accept_fracs.append(mechanisms.get((strategy, n, "accept"), 0) / total)

        x = np.arange(len(ns))
        ax.bar(x, weight_fracs, color=c[0], edgecolor="white", linewidth=0.3, width=0.8)
        ax.bar(x, list_fracs, bottom=weight_fracs, color=c[1], edgecolor="white", linewidth=0.3, width=0.8)
        bottom2 = [w + l for w, l in zip(weight_fracs, list_fracs)]
        ax.bar(x, accept_fracs, bottom=bottom2, color=c[2], edgecolor="white", linewidth=0.3, width=0.8)

        ax.set_title(STRATEGY_LABELS[strategy], fontsize=9)
        ax.set_ylim(0, 1.08)
        # Show sparse x-ticks
        tick_idx = list(range(0, len(ns), 4))
        if len(ns) - 1 not in tick_idx:
            tick_idx.append(len(ns) - 1)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([str(ns[i]) for i in tick_idx])

    axes[1, 0].set_xlabel("$n$")
    axes[1, 1].set_xlabel("$n$")
    axes[0, 0].set_ylabel("Proportion")
    axes[1, 0].set_ylabel("Proportion")

    # Shared legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=c[0], label="Reject: weight"),
        Patch(facecolor=c[1], label="Reject: list size"),
        Patch(facecolor=c[2], label="Accept"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Rejection mechanism by $n$ per strategy", y=1.06, fontsize=11)
    fig.tight_layout()
    save(fig, "rejection_mechanism_by_n")


# ---------------------------------------------------------------------------
# Artefact 3: Summary table (CSV)
# ---------------------------------------------------------------------------

def write_summary_table(
    rejection_rate: dict[tuple[str, int], float],
    ci: dict[tuple[str, int], tuple[float, float]],
    ns: list[int],
) -> None:
    """Write CSV summary: per (strategy, n) rejection rate with 95% CI."""
    path = OUT_DIR / "soundness_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "n", "rejection_rate", "ci_lower", "ci_upper"])
        for strategy in STRATEGY_ORDER:
            for n in ns:
                rate = rejection_rate.get((strategy, n), 0)
                lo, hi = ci.get((strategy, n), (0, 0))
                writer.writerow([
                    STRATEGY_LABELS[strategy],
                    n,
                    f"{rate:.4f}",
                    f"{lo:.4f}",
                    f"{hi:.4f}",
                ])
    print(f"  {path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Artefact 3b: LaTeX table
# ---------------------------------------------------------------------------

def write_latex_table(
    rejection_rate: dict[tuple[str, int], float],
    ci: dict[tuple[str, int], tuple[float, float]],
    ns: list[int],
) -> None:
    """Write a LaTeX table: strategies as rows, select n as columns."""
    # Pick representative n values
    if len(ns) <= 8:
        ns_show = ns
    else:
        ns_show = ns[::3]
        if ns[-1] not in ns_show:
            ns_show.append(ns[-1])

    path = OUT_DIR / "soundness_summary.tex"
    with open(path, "w") as f:
        cols = "l" + "c" * len(ns_show)
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Soundness: rejection rate by adversarial strategy and $n$ (95\% Wilson CI).}" + "\n")
        f.write(r"\label{tab:soundness}" + "\n")
        f.write(r"\begin{tabular}{" + cols + "}\n")
        f.write(r"\toprule" + "\n")

        header = "Strategy & " + " & ".join(f"$n={n}$" for n in ns_show) + r" \\" + "\n"
        f.write(header)
        f.write(r"\midrule" + "\n")

        for strategy in STRATEGY_ORDER:
            cells = [STRATEGY_LABELS[strategy]]
            for n in ns_show:
                rate = rejection_rate.get((strategy, n), 0)
                lo, hi = ci.get((strategy, n), (0, 0))
                # Format as percentage
                if rate == 1.0:
                    cells.append("100\\%")
                else:
                    cells.append(f"{rate*100:.0f}\\% [{lo*100:.0f}, {hi*100:.0f}]")
            f.write(" & ".join(cells) + r" \\" + "\n")

        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"  {path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_style()

    print("Loading soundness data...")
    data = load_data()
    trials = data["trials"]
    params = data["parameters"]
    print(f"  {len(trials)} trials, strategies={params['strategies']}, "
          f"n in {params['nRange'][0]}..{params['nRange'][-1]}")

    rejection_rate, ci, mechanisms, ns = build_tables(trials)

    print("\nGenerating artefacts:")
    plot_rejection_by_strategy(rejection_rate, ci, ns)
    plot_rejection_mechanism(mechanisms, ns)
    plot_rejection_mechanism_by_n(mechanisms, ns)
    write_summary_table(rejection_rate, ci, ns)
    write_latex_table(rejection_rate, ci, ns)

    print("\nDone.")


if __name__ == "__main__":
    main()
