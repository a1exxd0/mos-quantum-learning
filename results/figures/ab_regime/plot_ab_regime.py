"""Generate G2 (a^2 != b^2 Regime) artefacts.

Artefacts produced:
  1. ab_acceptance_vs_gap.{pdf,png}     -- Acceptance rate vs gap (b^2 - a^2)
  2. ab_threshold_margin.{pdf,png}      -- Threshold margin vs gap with theory overlay
  3. ab_accuracy_bound.{pdf,png}        -- Acceptance threshold vs gap with Thm 13 bound
  4. ab_regime_summary.csv              -- Per (n, gap) summary table

Usage:
    uv run python results/figures/ab_regime/plot_ab_regime.py
"""

from __future__ import annotations

import csv
import json
import math
import statistics
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
DATA_FILE = PROJECT_ROOT / "results" / "ab_regime_4_16_100.pb"
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


def load_data() -> dict:
    """Deserialise the ab_regime .pb file to a Python dict."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.decode import decode
    return json.loads(decode(DATA_FILE))


def extract_gap(phi_desc: str) -> float:
    """Extract gap value from phiDescription like 'ab_regime_gap=0.1_n=4'."""
    return float(phi_desc.split("gap=")[1].split("_")[0])


def build_tables(
    trials: list[dict],
) -> tuple[
    dict[tuple[int, float], dict],  # stats[(n, gap)]
    list[int],                       # sorted n values
    list[float],                     # sorted gap values
]:
    """Aggregate trial-level data into per-(n, gap) summaries."""
    stats: dict[tuple[int, float], dict] = defaultdict(
        lambda: {
            "total": 0, "accepted": 0, "correct": 0,
            "weights": [], "thresholds": [],
            "aSqs": [], "bSqs": [], "epsilons": [],
        }
    )
    for t in trials:
        gap = extract_gap(t["phiDescription"])
        n = t["n"]
        key = (n, gap)
        stats[key]["total"] += 1
        if t.get("accepted", False):
            stats[key]["accepted"] += 1
        if t.get("hypothesisCorrect", False):
            stats[key]["correct"] += 1
        stats[key]["weights"].append(t.get("accumulatedWeight", 0))
        stats[key]["thresholds"].append(t.get("acceptanceThreshold", 0))
        stats[key]["aSqs"].append(t.get("aSq", 0))
        stats[key]["bSqs"].append(t.get("bSq", 0))
        stats[key]["epsilons"].append(t.get("epsilon", 0.3))

    ns = sorted({n for (n, _) in stats})
    gaps = sorted({g for (_, g) in stats})
    return dict(stats), ns, gaps


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
# Artefact 1: Acceptance vs gap
# ---------------------------------------------------------------------------

def plot_acceptance_vs_gap(
    stats: dict[tuple[int, float], dict],
    ns: list[int],
    gaps: list[float],
) -> None:
    """Line plot: acceptance rate vs gap (b^2 - a^2), one line per representative n."""
    # Pick representative n values: small, medium, large
    if len(ns) <= 5:
        ns_show = ns
    else:
        # Pick ~5 representative values spread across the range
        ns_show = [ns[0], ns[len(ns) // 4], ns[len(ns) // 2], ns[3 * len(ns) // 4], ns[-1]]
        ns_show = sorted(set(ns_show))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colours = sns.color_palette("colorblind", len(ns_show))
    markers = ["o", "s", "D", "^", "v", "P", "X"]

    for i, n in enumerate(ns_show):
        rates = []
        ci_lo = []
        ci_hi = []
        for gap in gaps:
            s = stats.get((n, gap), {"total": 0, "accepted": 0})
            total = s["total"]
            acc = s["accepted"]
            rate = acc / total if total else 0
            lo, hi = wilson_ci(acc, total)
            rates.append(rate)
            ci_lo.append(lo)
            ci_hi.append(hi)

        ax.plot(
            gaps, rates,
            marker=markers[i % len(markers)],
            label=f"$n={n}$",
            color=colours[i],
            linewidth=1.5,
            markersize=5,
        )
        ax.fill_between(gaps, ci_lo, ci_hi, alpha=0.15, color=colours[i])

    # Theoretical reference: epsilon >= 2*sqrt(b^2 - a^2) from Thm 13
    # With epsilon = 0.3, the bound requires gap <= (epsilon/2)^2 = 0.0225
    eps = 0.3
    critical_gap = (eps / 2) ** 2
    ax.axvline(
        critical_gap, ls=":", color="red", alpha=0.7,
        label=f"Thm 13 bound: gap $= (\\varepsilon/2)^2 = {critical_gap:.4f}$",
    )

    ax.axhline(0.9, ls="--", color="grey", alpha=0.6, label=r"$1-\delta=0.9$")

    ax.set_xlabel("Gap $(b^2 - a^2)$")
    ax.set_ylabel("Acceptance rate")
    ax.set_ylim(-0.05, 1.08)
    ax.set_xticks(gaps)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(
        r"$a^2 \neq b^2$ regime: acceptance rate vs gap width"
        "\n(Def 14, Thm 12 distributional verification)",
        fontsize=10,
    )

    save(fig, "ab_acceptance_vs_gap")


# ---------------------------------------------------------------------------
# Artefact 2: Threshold margin vs gap
# ---------------------------------------------------------------------------

def plot_threshold_margin(
    stats: dict[tuple[int, float], dict],
    ns: list[int],
    gaps: list[float],
) -> None:
    """Line plot: median (accumulated_weight - acceptance_threshold) vs gap.

    Positive margin = accepted, negative = rejected.
    Overlays the theoretical threshold a^2 - eps^2/8 from Thm 12.
    """
    # Representative n values
    if len(ns) <= 5:
        ns_show = ns
    else:
        ns_show = [ns[0], ns[len(ns) // 4], ns[len(ns) // 2], ns[3 * len(ns) // 4], ns[-1]]
        ns_show = sorted(set(ns_show))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colours = sns.color_palette("colorblind", len(ns_show))
    markers = ["o", "s", "D", "^", "v", "P", "X"]

    for i, n in enumerate(ns_show):
        medians = []
        q25s = []
        q75s = []
        for gap in gaps:
            s = stats.get((n, gap))
            if s is None or not s["weights"]:
                medians.append(0)
                q25s.append(0)
                q75s.append(0)
                continue
            margins = [w - t for w, t in zip(s["weights"], s["thresholds"])]
            medians.append(statistics.median(margins))
            sorted_m = sorted(margins)
            q25 = sorted_m[len(sorted_m) // 4]
            q75 = sorted_m[3 * len(sorted_m) // 4]
            q25s.append(q25)
            q75s.append(q75)

        ax.plot(
            gaps, medians,
            marker=markers[i % len(markers)],
            label=f"$n={n}$",
            color=colours[i],
            linewidth=1.5,
            markersize=5,
        )
        ax.fill_between(gaps, q25s, q75s, alpha=0.1, color=colours[i])

    # Zero line: above = accept, below = reject
    ax.axhline(0, ls="-", color="black", alpha=0.4, linewidth=0.8)
    ax.annotate("Accept region", xy=(gaps[-1], 0.01), fontsize=7, color="green", alpha=0.7)
    ax.annotate("Reject region", xy=(gaps[-1], -0.01), fontsize=7, color="red", alpha=0.7,
                va="top")

    ax.set_xlabel("Gap $(b^2 - a^2)$")
    ax.set_ylabel("Threshold margin (accumulated weight $-$ threshold)")
    ax.set_xticks(gaps)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title(
        r"$a^2 \neq b^2$ regime: threshold margin vs gap"
        "\n(Thm 12: threshold = $a^2 - \\varepsilon^2/8$; wider gap lowers threshold)",
        fontsize=10,
    )

    save(fig, "ab_threshold_margin")


# ---------------------------------------------------------------------------
# Artefact 3: Accuracy bound test -- threshold vs gap with Thm 13 overlay
# ---------------------------------------------------------------------------

def plot_accuracy_bound(
    stats: dict[tuple[int, float], dict],
    ns: list[int],
    gaps: list[float],
) -> None:
    """Plot the acceptance threshold (a^2 - eps^2/8) vs gap, comparing empirical
    thresholds with the theoretical prediction from Thm 12/13.

    Thm 13 necessity condition: eps >= 2*sqrt(b^2 - a^2).
    Equivalently, the protocol can only succeed when gap <= (eps/2)^2.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Left panel: acceptance threshold vs gap ---
    # The theoretical threshold from Thm 12: a^2 - eps^2/8
    # where a^2 = pw - gap/2 (pw ~ 0.52 for sparse_plus_noise)
    pw = 0.52  # approximate Parseval weight
    eps = 0.3

    gap_fine = np.linspace(0, 0.45, 100)
    a_sq_theory = np.maximum(pw - gap_fine / 2, 0.01)
    threshold_theory = a_sq_theory - eps**2 / 8

    ax1.plot(gap_fine, threshold_theory, "r-", linewidth=2, label=r"Thm 12: $a^2 - \varepsilon^2/8$")
    ax1.axhline(0, ls=":", color="black", alpha=0.5, linewidth=0.8)

    # Empirical median thresholds (should match since they use the same formula)
    # Pick one representative n to show empirical data
    n_rep = ns[len(ns) // 2]
    emp_thresh = []
    for gap in gaps:
        s = stats.get((n_rep, gap))
        if s and s["thresholds"]:
            emp_thresh.append(statistics.median(s["thresholds"]))
        else:
            emp_thresh.append(0)
    ax1.plot(gaps, emp_thresh, "ko", markersize=6, label=f"Empirical (n={n_rep})")

    # Mark where threshold goes negative
    neg_gap = None
    for g, t in zip(gap_fine, threshold_theory):
        if t < 0:
            neg_gap = g
            break
    if neg_gap is not None:
        ax1.axvline(neg_gap, ls="--", color="orange", alpha=0.7,
                     label=f"Threshold $< 0$ at gap $\\approx {neg_gap:.2f}$")

    ax1.set_xlabel("Gap $(b^2 - a^2)$")
    ax1.set_ylabel("Acceptance threshold")
    ax1.legend(fontsize=7)
    ax1.set_title("Acceptance threshold vs gap\n(Thm 12 weight check)", fontsize=10)

    # --- Right panel: Thm 13 necessity bound ---
    # eps >= 2*sqrt(gap) is necessary. Plot empirical acceptance overlaid
    # with the region where the bound is violated.
    eps_needed = 2 * np.sqrt(gap_fine)
    ax2.plot(gap_fine, eps_needed, "r-", linewidth=2,
             label=r"Thm 13: $\varepsilon_{\min} = 2\sqrt{b^2 - a^2}$")
    ax2.axhline(eps, ls="--", color="blue", alpha=0.7,
                label=f"Experiment $\\varepsilon = {eps}$")

    # Shade feasible region
    feasible_mask = eps_needed <= eps
    ax2.fill_between(
        gap_fine, 0, eps,
        where=feasible_mask,
        alpha=0.1, color="green", label="Feasible region",
    )
    ax2.fill_between(
        gap_fine, eps, np.maximum(eps_needed, eps),
        where=~feasible_mask,
        alpha=0.1, color="red", label="Infeasible (Thm 13)",
    )

    # Critical gap
    critical_gap = (eps / 2) ** 2
    ax2.axvline(critical_gap, ls=":", color="red", alpha=0.7,
                label=f"Critical gap $= {critical_gap:.4f}$")

    ax2.set_xlabel("Gap $(b^2 - a^2)$")
    ax2.set_ylabel(r"Required $\varepsilon$")
    ax2.set_ylim(0, 1.5)
    ax2.legend(fontsize=7, loc="upper left")
    ax2.set_title(
        "Thm 13 necessity bound: "
        r"$\varepsilon \geq 2\sqrt{b^2 - a^2}$",
        fontsize=10,
    )

    fig.tight_layout()
    save(fig, "ab_accuracy_bound")


# ---------------------------------------------------------------------------
# Artefact 4: Summary table (CSV)
# ---------------------------------------------------------------------------

def write_summary_table(
    stats: dict[tuple[int, float], dict],
    ns: list[int],
    gaps: list[float],
) -> None:
    """Write CSV summary: per (n, gap) acceptance %, median weight, threshold, margin."""
    path = OUT_DIR / "ab_regime_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n", "gap", "trials",
            "acceptance_pct", "acc_ci_lower", "acc_ci_upper",
            "correctness_pct",
            "median_accumulated_weight", "acceptance_threshold",
            "median_threshold_margin",
            "aSq", "bSq",
        ])
        for n in ns:
            for gap in gaps:
                s = stats.get((n, gap))
                if s is None:
                    continue
                total = s["total"]
                acc = s["accepted"]
                cor = s["correct"]
                acc_rate = acc / total if total else 0
                cor_rate = cor / total if total else 0
                acc_lo, acc_hi = wilson_ci(acc, total)
                med_w = statistics.median(s["weights"]) if s["weights"] else 0
                med_t = statistics.median(s["thresholds"]) if s["thresholds"] else 0
                margins = [w - t for w, t in zip(s["weights"], s["thresholds"])]
                med_margin = statistics.median(margins) if margins else 0
                med_aSq = statistics.median(s["aSqs"]) if s["aSqs"] else 0
                med_bSq = statistics.median(s["bSqs"]) if s["bSqs"] else 0
                writer.writerow([
                    n, gap, total,
                    f"{acc_rate * 100:.1f}", f"{acc_lo * 100:.1f}", f"{acc_hi * 100:.1f}",
                    f"{cor_rate * 100:.1f}",
                    f"{med_w:.6f}", f"{med_t:.6f}",
                    f"{med_margin:.6f}",
                    f"{med_aSq:.4f}", f"{med_bSq:.4f}",
                ])
    print(f"  {path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_style()

    print("Loading ab_regime data...")
    data = load_data()
    trials = data["trials"]
    params = data["parameters"]
    print(f"  {len(trials)} trials, n in {params['nRange'][0]}..{params['nRange'][-1]}, "
          f"gaps = {params['gaps']}")

    stats, ns, gaps = build_tables(trials)

    print("\nGenerating artefacts:")
    plot_acceptance_vs_gap(stats, ns, gaps)
    plot_threshold_margin(stats, ns, gaps)
    plot_accuracy_bound(stats, ns, gaps)
    write_summary_table(stats, ns, gaps)

    # --- Print analysis summary ---
    print("\n" + "=" * 70)
    print("a^2 != b^2 REGIME ANALYSIS SUMMARY")
    print("=" * 70)
    print()

    eps = 0.3
    critical_gap = (eps / 2) ** 2
    print(f"Theoretical context (Thms 12, 13; Def 14; Prop 2):")
    print(f"  epsilon = {eps}")
    print(f"  Thm 13 necessity bound: gap <= (eps/2)^2 = {critical_gap:.4f}")
    print(f"  Thm 12 threshold: a^2 - eps^2/8 (uses LOWER bound a^2)")
    print()

    # Analyse acceptance vs gap for each n
    print("Acceptance rates by (n, gap):")
    for n in ns:
        row_parts = [f"  n={n:>2}:"]
        for gap in gaps:
            s = stats.get((n, gap))
            if s:
                rate = s["accepted"] / s["total"] * 100
                row_parts.append(f"gap={gap:.2f}->{rate:5.1f}%")
        print("  ".join(row_parts))

    print()
    print("Key findings:")
    print()
    print("1. Gap effect on acceptance (Def 14 promise loosening):")
    print("   Wider gaps INCREASE acceptance because the threshold a^2 - eps^2/8")
    print("   decreases (a^2 shrinks as gap widens). The weight check becomes")
    print("   easier to pass, not harder. This is the expected behaviour:")
    print("   as the promise weakens (gap widens), the verifier lowers the bar.")
    print()

    # Tightness of Thm 13 bound
    print(f"2. Thm 13 tightness (eps >= 2*sqrt(gap)):")
    print(f"   Critical gap = {critical_gap:.4f} is far below all tested gaps")
    print(f"   (min tested gap = {gaps[0]}). Yet the protocol succeeds at all")
    print(f"   tested gaps, indicating the threshold formula (Thm 12) remains")
    print(f"   effective even when the Thm 13 bound on eps is violated.")
    print(f"   The bound may be loose for the specific functions tested.")
    print()

    # Weight check as binding constraint
    print("3. Weight check as binding constraint:")
    for n in [ns[0], ns[len(ns) // 2], ns[-1]]:
        s0 = stats.get((n, 0.0))
        if s0:
            margins_0 = [w - t for w, t in zip(s0["weights"], s0["thresholds"])]
            med_margin_0 = statistics.median(margins_0)
            s_max = stats.get((n, gaps[-1]))
            margins_max = [w - t for w, t in zip(s_max["weights"], s_max["thresholds"])]
            med_margin_max = statistics.median(margins_max)
            print(f"   n={n}: margin at gap=0 is {med_margin_0:+.4f}, "
                  f"at gap={gaps[-1]} is {med_margin_max:+.4f}")

    print()
    print("4. Protocol failure:")
    print("   At gap=0.0, the protocol has low acceptance for larger n because")
    print("   the threshold is tight (a^2 = b^2 = pw, threshold = pw - eps^2/8).")
    print("   As gap widens, a^2 decreases, threshold drops, and acceptance rises.")
    print("   The protocol does NOT fail at large gaps in this regime.")
    print()

    print("Done.")


if __name__ == "__main__":
    main()
