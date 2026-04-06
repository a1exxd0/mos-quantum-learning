"""Generate all Theta Sensitivity experiment artefacts.

Artefacts produced:
  1. acceptance_heatmap.{pdf,png}       -- Heatmap (n x theta): acceptance rate
  2. list_size_vs_theta.{pdf,png}       -- Median |L| vs theta with 4/theta^2 bound
  3. tradeoff_curve.{pdf,png}           -- Dual-axis: acceptance + median |L| vs theta
  4. theta_sensitivity_summary.csv      -- Per (n, theta) summary table

Usage:
    uv run python results/figures/theta_sensitivity/plot_theta_sensitivity.py
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
DATA_FILE = PROJECT_ROOT / "results" / "theta_sensitivity_4_16_100.pb"
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
    """Deserialise the theta_sensitivity .pb file to a Python dict."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.decode import decode
    return json.loads(decode(DATA_FILE))


def build_tables(trials: list[dict]) -> dict:
    """Aggregate trial-level data into per-(n, theta) summaries.

    Returns a dict keyed by (n, theta) with fields:
        total, accepted, correct, list_sizes, postselection_rates
    """
    agg: dict[tuple[int, float], dict] = defaultdict(lambda: {
        "total": 0,
        "accepted": 0,
        "correct": 0,
        "list_sizes": [],
        "postselection_rates": [],
    })

    for t in trials:
        key = (t["n"], t["theta"])
        rec = agg[key]
        rec["total"] += 1
        if t.get("accepted", False):
            rec["accepted"] += 1
        if t.get("hypothesisCorrect", False):
            rec["correct"] += 1
        rec["list_sizes"].append(t["listSize"])
        rec["postselection_rates"].append(t["postselectionRate"])

    return dict(agg)


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
# Artefact 1: Acceptance heatmap (n x theta)
# ---------------------------------------------------------------------------

def plot_acceptance_heatmap(
    agg: dict[tuple[int, float], dict],
    ns: list[int],
    thetas: list[float],
) -> None:
    """Heatmap: acceptance rate as colour, rows = n, columns = theta."""
    matrix = np.zeros((len(ns), len(thetas)))
    for i, n in enumerate(ns):
        for j, theta in enumerate(thetas):
            rec = agg.get((n, theta))
            if rec and rec["total"] > 0:
                matrix[i, j] = rec["accepted"] / rec["total"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   origin="lower")

    # Annotate cells
    for i in range(len(ns)):
        for j in range(len(thetas)):
            val = matrix[i, j]
            text_colour = "white" if val < 0.4 or val > 0.85 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=8, color=text_colour, fontweight="bold")

    ax.set_xticks(range(len(thetas)))
    ax.set_xticklabels([f"{t:.2f}" for t in thetas])
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels([str(n) for n in ns])
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("$n$")
    ax.set_title(r"Acceptance rate by $n$ and $\theta$")

    cbar = fig.colorbar(im, ax=ax, label="Acceptance rate", shrink=0.85)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    # Mark extraction boundary at theta=0.20
    boundary_idx = thetas.index(0.20) if 0.20 in thetas else None
    if boundary_idx is not None:
        ax.axvline(boundary_idx, color="navy", ls="--", lw=1.5, alpha=0.7)
        ax.text(boundary_idx, len(ns) - 0.5,
                r"extraction boundary $\theta\!=\!0.20$",
                ha="center", va="bottom", fontsize=7, color="navy",
                fontstyle="italic")

    fig.tight_layout()
    save(fig, "acceptance_heatmap")


# ---------------------------------------------------------------------------
# Artefact 2: List size vs theta with theoretical bound
# ---------------------------------------------------------------------------

def plot_list_size_vs_theta(
    agg: dict[tuple[int, float], dict],
    ns: list[int],
    thetas: list[float],
) -> None:
    """Line plot: median |L| vs theta for representative n, with 4/theta^2 bound."""
    # Pick representative n values
    representative_ns = [4, 8, 16]
    representative_ns = [n for n in representative_ns if n in ns]

    colours = sns.color_palette("colorblind", len(representative_ns) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Theoretical bound
    theta_fine = np.linspace(thetas[0], thetas[-1], 200)
    bound = 4.0 / theta_fine**2
    ax.plot(theta_fine, bound, "k--", lw=1.5, alpha=0.6,
            label=r"$4/\theta^2$ (Parseval bound)")

    for i, n in enumerate(representative_ns):
        medians = []
        q25s = []
        q75s = []
        for theta in thetas:
            rec = agg.get((n, theta))
            if rec:
                sizes = np.array(rec["list_sizes"])
                medians.append(np.median(sizes))
                q25s.append(np.percentile(sizes, 25))
                q75s.append(np.percentile(sizes, 75))
            else:
                medians.append(np.nan)
                q25s.append(np.nan)
                q75s.append(np.nan)

        medians = np.array(medians)
        q25s = np.array(q25s)
        q75s = np.array(q75s)

        ax.plot(thetas, medians, "o-", color=colours[i], lw=1.5, ms=5,
                label=f"$n={n}$ (median)")
        ax.fill_between(thetas, q25s, q75s, alpha=0.15, color=colours[i])

    # Mark extraction boundary
    ax.axvline(0.20, color="navy", ls=":", lw=1.2, alpha=0.7)
    ax.annotate(r"$\theta=0.20$" + "\n(extraction\nboundary)",
                xy=(0.20, ax.get_ylim()[1] * 0.7),
                xytext=(0.28, ax.get_ylim()[1] * 0.6),
                fontsize=7, color="navy",
                arrowprops=dict(arrowstyle="->", color="navy", lw=0.8))

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"List size $|L|$")
    ax.set_yscale("log")
    ax.set_title(r"List size vs resolution threshold $\theta$")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    save(fig, "list_size_vs_theta")


# ---------------------------------------------------------------------------
# Artefact 3: Trade-off curve (dual-axis)
# ---------------------------------------------------------------------------

def plot_tradeoff_curve(
    agg: dict[tuple[int, float], dict],
    ns: list[int],
    thetas: list[float],
) -> None:
    """Dual-axis: acceptance rate (left) and median |L| (right, log) vs theta."""
    # Pick one or two representative n values
    show_ns = [8, 16]
    show_ns = [n for n in show_ns if n in ns]
    if not show_ns:
        show_ns = [ns[len(ns) // 2]]

    colours = sns.color_palette("colorblind", len(show_ns))

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()

    for i, n in enumerate(show_ns):
        acc_rates = []
        medians = []
        for theta in thetas:
            rec = agg.get((n, theta))
            if rec and rec["total"] > 0:
                acc_rates.append(rec["accepted"] / rec["total"])
                medians.append(np.median(rec["list_sizes"]))
            else:
                acc_rates.append(np.nan)
                medians.append(np.nan)

        # Acceptance rate on left axis (solid)
        ax1.plot(thetas, acc_rates, "o-", color=colours[i], lw=2, ms=5,
                 label=f"Acceptance ($n={n}$)")
        # Median |L| on right axis (dashed)
        ax2.plot(thetas, medians, "s--", color=colours[i], lw=1.5, ms=4,
                 alpha=0.7, label=f"Median $|L|$ ($n={n}$)")

    # Mark extraction boundary
    ax1.axvline(0.20, color="navy", ls=":", lw=1.2, alpha=0.7)
    ax1.text(0.205, 0.05, r"$\theta=0.20$", fontsize=7, color="navy",
             transform=ax1.get_xaxis_transform())

    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel("Acceptance rate", color="black")
    ax1.set_ylim(-0.05, 1.1)
    ax1.tick_params(axis="y")

    ax2.set_ylabel(r"Median list size $|L|$", color="grey")
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor="grey")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="center left")

    ax1.set_title(r"Precision--communication trade-off vs $\theta$")
    fig.tight_layout()
    save(fig, "tradeoff_curve")


# ---------------------------------------------------------------------------
# Artefact 4: Summary table (CSV)
# ---------------------------------------------------------------------------

def write_summary_csv(
    agg: dict[tuple[int, float], dict],
    ns: list[int],
    thetas: list[float],
) -> None:
    """CSV: per (n, theta) acceptance %, correctness %, median |L|, median post-sel rate."""
    path = OUT_DIR / "theta_sensitivity_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n", "theta",
            "acceptance_pct", "acc_ci_lo", "acc_ci_hi",
            "correctness_pct", "corr_ci_lo", "corr_ci_hi",
            "median_list_size", "q25_list_size", "q75_list_size",
            "median_postselection_rate",
        ])
        for n in ns:
            for theta in thetas:
                rec = agg.get((n, theta))
                if not rec or rec["total"] == 0:
                    continue
                total = rec["total"]
                acc_pct = rec["accepted"] / total
                corr_pct = rec["correct"] / total
                acc_lo, acc_hi = wilson_ci(rec["accepted"], total)
                corr_lo, corr_hi = wilson_ci(rec["correct"], total)
                sizes = np.array(rec["list_sizes"])
                psrs = np.array(rec["postselection_rates"])

                writer.writerow([
                    n, theta,
                    f"{acc_pct:.4f}", f"{acc_lo:.4f}", f"{acc_hi:.4f}",
                    f"{corr_pct:.4f}", f"{corr_lo:.4f}", f"{corr_hi:.4f}",
                    f"{np.median(sizes):.1f}",
                    f"{np.percentile(sizes, 25):.1f}",
                    f"{np.percentile(sizes, 75):.1f}",
                    f"{np.median(psrs):.4f}",
                ])
    print(f"  {path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Analysis / reporting
# ---------------------------------------------------------------------------

def report_findings(
    agg: dict[tuple[int, float], dict],
    ns: list[int],
    thetas: list[float],
) -> None:
    """Print key findings to stdout."""
    print("\n" + "=" * 70)
    print("THETA SENSITIVITY: KEY FINDINGS")
    print("=" * 70)

    # 1. Does the 4/theta^2 bound hold?
    print("\n1. List-size bound check (|L| <= 4/theta^2):")
    violations = 0
    total_checks = 0
    for n in ns:
        for theta in thetas:
            rec = agg.get((n, theta))
            if not rec:
                continue
            bound = 4.0 / theta**2
            max_l = max(rec["list_sizes"])
            total_checks += 1
            if max_l > bound:
                violations += 1
                print(f"   VIOLATION: n={n}, theta={theta:.2f}: "
                      f"max |L|={max_l}, bound={bound:.0f}")
    if violations == 0:
        print(f"   Bound holds in all {total_checks} (n, theta) cells.")
    else:
        print(f"   {violations}/{total_checks} violations found.")

    # 2. Extraction boundary at theta=0.20
    print("\n2. Extraction boundary (theta ~= 0.20):")
    for n in ns:
        row = []
        for theta in thetas:
            rec = agg.get((n, theta))
            if rec:
                row.append((theta, np.median(rec["list_sizes"])))
        line = ", ".join(f"theta={t:.2f}: |L|={l:.0f}" for t, l in row)
        print(f"   n={n:2d}: {line}")

    # 3. Sweet spot analysis
    print("\n3. Sweet spot analysis (highest acceptance with reasonable |L|):")
    for n in ns:
        best_theta = None
        best_acc = -1
        for theta in thetas:
            rec = agg.get((n, theta))
            if rec and rec["total"] > 0:
                acc = rec["accepted"] / rec["total"]
                if acc > best_acc:
                    best_acc = acc
                    best_theta = theta
        rec = agg[(n, best_theta)]
        med_l = np.median(rec["list_sizes"])
        print(f"   n={n:2d}: best theta={best_theta:.2f} "
              f"(acceptance={best_acc:.0%}, median |L|={med_l:.0f})")

    # 4. Postselection rate vs theta
    print("\n4. Postselection rate vs theta (Thm 5 predicts ~0.5):")
    for theta in thetas:
        rates = []
        for n in ns:
            rec = agg.get((n, theta))
            if rec:
                rates.extend(rec["postselection_rates"])
        if rates:
            print(f"   theta={theta:.2f}: median postsel = {np.median(rates):.3f}, "
                  f"mean = {np.mean(rates):.3f}")

    # 5. Optimal theta shift with n
    print("\n5. Does optimal theta shift with n?")
    for n in ns:
        best_theta = None
        best_score = -1
        for theta in thetas:
            rec = agg.get((n, theta))
            if rec and rec["total"] > 0:
                acc = rec["accepted"] / rec["total"]
                corr = rec["correct"] / rec["total"]
                # Score: joint acceptance and correctness
                score = acc * corr
                if score > best_score:
                    best_score = score
                    best_theta = theta
        rec_best = agg[(n, best_theta)]
        print(f"   n={n:2d}: best theta={best_theta:.2f} "
              f"(acc*corr={best_score:.3f}, median |L|={np.median(rec_best['list_sizes']):.0f})")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_style()

    print("Loading theta sensitivity data...")
    data = load_data()
    trials = data["trials"]
    params = data["parameters"]
    thetas = sorted(params["thetaValues"])
    ns = sorted(params["nRange"])
    print(f"  {len(trials)} trials, theta in {thetas}, n in {ns}")

    agg = build_tables(trials)

    print("\nGenerating artefacts:")
    plot_acceptance_heatmap(agg, ns, thetas)
    plot_list_size_vs_theta(agg, ns, thetas)
    plot_tradeoff_curve(agg, ns, thetas)
    write_summary_csv(agg, ns, thetas)

    report_findings(agg, ns, thetas)

    print("\nDone.")


if __name__ == "__main__":
    main()
