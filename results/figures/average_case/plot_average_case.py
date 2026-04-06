"""Generate all Average Case experiment artefacts.

Artefacts produced:
  1. acceptance_by_family.{pdf,png}  -- Line plot of acceptance rate vs n per family
  2. list_size_by_family.{pdf,png}   -- Line plot of median |L| vs n per family
  3. cross_family_summary.csv        -- Per (n, family) acceptance %, correctness %, median |L|, median copies

Usage:
    uv run python results/figures/average_case/plot_average_case.py
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
import statistics
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
DATA_FILE = PROJECT_ROOT / "results" / "average_case_4_16_100.pb"
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


FAMILY_LABELS = {
    "k_sparse_2": "$k$-sparse ($k{=}2$)",
    "k_sparse_4": "$k$-sparse ($k{=}4$)",
    "random_boolean": "Random Boolean",
    "sparse_plus_noise": "Sparse + noise",
}

FAMILY_LABELS_PLAIN = {
    "k_sparse_2": "k-sparse (k=2)",
    "k_sparse_4": "k-sparse (k=4)",
    "random_boolean": "Random Boolean",
    "sparse_plus_noise": "Sparse + noise",
}

FAMILY_ORDER = ["k_sparse_2", "k_sparse_4", "sparse_plus_noise", "random_boolean"]


def parse_family(phi_desc: str) -> str:
    """Extract family key from phiDescription like 'k_sparse_2_n=4'."""
    # Strip the trailing _n=<digits>
    return re.sub(r"_n=\d+$", "", phi_desc)


def load_data() -> dict:
    """Deserialise the average_case .pb file to a Python dict."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.decode import decode
    return json.loads(decode(DATA_FILE))


def build_tables(
    trials: list[dict],
) -> tuple[
    dict[tuple[str, int], float],                    # acceptance_rate[(family, n)]
    dict[tuple[str, int], tuple[float, float]],      # ci[(family, n)]
    dict[tuple[str, int], float],                     # median_list_size[(family, n)]
    dict[tuple[str, int], float],                     # median_copies[(family, n)]
    dict[tuple[str, int], float],                     # prover_found_rate[(family, n)]
    dict[tuple[str, int], float],                     # theta[(family, n)]
    list[int],                                        # sorted n values
]:
    """Aggregate trial-level data into per-(family, n) summaries."""
    counts: dict[tuple[str, int], int] = defaultdict(int)
    accepts: dict[tuple[str, int], int] = defaultdict(int)
    found: dict[tuple[str, int], int] = defaultdict(int)
    list_sizes: dict[tuple[str, int], list[int]] = defaultdict(list)
    copies: dict[tuple[str, int], list[int]] = defaultdict(list)
    theta_map: dict[tuple[str, int], float] = {}

    for t in trials:
        family = parse_family(t["phiDescription"])
        n = t["n"]
        key = (family, n)
        counts[key] += 1
        if t.get("outcome") == "accept":
            accepts[key] += 1
        if t.get("proverFoundTarget", False):
            found[key] += 1
        list_sizes[key].append(t.get("listSize", 0))
        copies[key].append(t.get("totalCopies", 0))
        theta_map[key] = t["theta"]

    acceptance_rate: dict[tuple[str, int], float] = {}
    ci: dict[tuple[str, int], tuple[float, float]] = {}
    median_list: dict[tuple[str, int], float] = {}
    median_cop: dict[tuple[str, int], float] = {}
    found_rate: dict[tuple[str, int], float] = {}

    for key, total in counts.items():
        acc = accepts.get(key, 0)
        acceptance_rate[key] = acc / total if total else 0
        ci[key] = wilson_ci(acc, total)
        median_list[key] = statistics.median(list_sizes[key])
        median_cop[key] = statistics.median(copies[key])
        fnd = found.get(key, 0)
        found_rate[key] = fnd / total if total else 0

    ns = sorted({n for (_, n) in counts})
    return acceptance_rate, ci, median_list, median_cop, found_rate, theta_map, ns


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
# Artefact 1: Acceptance by family (line plot with Wilson CI bands)
# ---------------------------------------------------------------------------

def plot_acceptance_by_family(
    acceptance_rate: dict[tuple[str, int], float],
    ci: dict[tuple[str, int], tuple[float, float]],
    ns: list[int],
) -> None:
    """Line plot of acceptance rate vs n, one line per function family."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colours = sns.color_palette("colorblind", len(FAMILY_ORDER))
    markers = ["o", "s", "D", "^"]

    for i, family in enumerate(FAMILY_ORDER):
        rates = [acceptance_rate.get((family, n), 0) for n in ns]
        lo = [ci.get((family, n), (0, 0))[0] for n in ns]
        hi = [ci.get((family, n), (0, 0))[1] for n in ns]

        ax.plot(ns, rates, marker=markers[i], color=colours[i],
                label=FAMILY_LABELS[family], linewidth=1.5, markersize=5)
        ax.fill_between(ns, lo, hi, color=colours[i], alpha=0.15)

    # Reference line: 1 - delta = 0.9
    ax.axhline(0.9, ls="--", color="grey", alpha=0.6, label=r"$1 - \delta = 0.9$")

    ax.set_xlabel("$n$ (number of qubits)")
    ax.set_ylabel("Acceptance rate")
    ax.set_ylim(-0.05, 1.08)
    ax.set_xticks(ns[::2] if len(ns) > 8 else ns)
    ax.legend(fontsize=8, loc="center right")
    ax.set_title("Average case: acceptance rate by function family")
    fig.tight_layout()
    save(fig, "acceptance_by_family")


# ---------------------------------------------------------------------------
# Artefact 2: List size by family (line plot with 4/theta^2 bound)
# ---------------------------------------------------------------------------

def plot_list_size_by_family(
    median_list: dict[tuple[str, int], float],
    theta_map: dict[tuple[str, int], float],
    ns: list[int],
) -> None:
    """Line plot of median |L| vs n, one line per family, with 4/theta^2 bound."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colours = sns.color_palette("colorblind", len(FAMILY_ORDER))
    markers = ["o", "s", "D", "^"]

    # Collect unique theta values to draw bound lines
    theta_vals = set()
    for family in FAMILY_ORDER:
        for n in ns:
            if (family, n) in theta_map:
                theta_vals.add(theta_map[(family, n)])

    for i, family in enumerate(FAMILY_ORDER):
        sizes = [median_list.get((family, n), 0) for n in ns]
        ax.plot(ns, sizes, marker=markers[i], color=colours[i],
                label=FAMILY_LABELS[family], linewidth=1.5, markersize=5)

    # Draw 4/theta^2 Parseval bound (upper bound on detectable coefficients)
    for theta in sorted(theta_vals):
        bound = 4.0 / theta**2
        ax.axhline(bound, ls=":", color="grey", alpha=0.5)
        ax.text(ns[-1] + 0.3, bound,
                r"$4/\theta^2$" + f" ($\\theta={theta}$, Parseval)",
                va="center", fontsize=7, color="grey")

    ax.set_xlabel("$n$ (number of qubits)")
    ax.set_ylabel("Median list size $|L|$")
    ax.set_xticks(ns[::2] if len(ns) > 8 else ns)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Average case: list size by function family")
    fig.tight_layout()
    save(fig, "list_size_by_family")


# ---------------------------------------------------------------------------
# Artefact 3: Cross-family summary table (CSV)
# ---------------------------------------------------------------------------

def write_summary_table(
    acceptance_rate: dict[tuple[str, int], float],
    ci: dict[tuple[str, int], tuple[float, float]],
    found_rate: dict[tuple[str, int], float],
    median_list: dict[tuple[str, int], float],
    median_cop: dict[tuple[str, int], float],
    ns: list[int],
) -> None:
    """Write CSV: per (n, family) acceptance %, prover found %, median |L|, median copies."""
    path = OUT_DIR / "cross_family_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "family", "n",
            "acceptance_rate", "ci_lower", "ci_upper",
            "prover_found_rate",
            "median_list_size", "median_copies",
        ])
        for family in FAMILY_ORDER:
            for n in ns:
                key = (family, n)
                rate = acceptance_rate.get(key, 0)
                lo, hi = ci.get(key, (0, 0))
                fr = found_rate.get(key, 0)
                ml = median_list.get(key, 0)
                mc = median_cop.get(key, 0)
                writer.writerow([
                    FAMILY_LABELS_PLAIN[family],
                    n,
                    f"{rate:.4f}",
                    f"{lo:.4f}",
                    f"{hi:.4f}",
                    f"{fr:.4f}",
                    f"{ml:.1f}",
                    f"{mc:.0f}",
                ])
    print(f"  {path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_style()

    print("Loading average case data...")
    data = load_data()
    trials = data["trials"]
    params = data["parameters"]
    print(f"  {len(trials)} trials, families={params['families']}, "
          f"n in {params['nRange'][0]}..{params['nRange'][-1]}")

    acceptance_rate, ci, median_list, median_cop, found_rate, theta_map, ns = \
        build_tables(trials)

    print("\nGenerating artefacts:")
    plot_acceptance_by_family(acceptance_rate, ci, ns)
    plot_list_size_by_family(median_list, theta_map, ns)
    write_summary_table(acceptance_rate, ci, found_rate, median_list, median_cop, ns)

    print("\nDone.")


if __name__ == "__main__":
    main()
