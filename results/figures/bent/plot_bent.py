"""Generate all Bent Function experiment artefacts.

Artefacts produced:
  1. list_size_growth.{pdf,png}    -- Line plot: median |L| vs n (log-scale y)
  2. bent_vs_parity_acceptance.{pdf,png} -- Grouped bar: acceptance rate bent vs parity
  3. resource_explosion.{pdf,png}  -- Line plot: copies + wall-clock vs n (log-scale y)
  4. bent_summary.csv              -- Per-n summary table

Usage:
    uv run python results/figures/bent/plot_bent.py
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
DATA_FILE = PROJECT_ROOT / "results" / "bent_4_16_100.pb"
OUT_DIR = SCRIPT_DIR  # figures land next to the script

# ---------------------------------------------------------------------------
# Protocol parameters
# ---------------------------------------------------------------------------
THETA = 0.3
EPSILON = 0.3


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
    """Deserialise the bent .pb file to a Python dict."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.decode import decode

    return json.loads(decode(DATA_FILE))


def theoretical_list_size(n: int, theta: float) -> float:
    """Predicted |L| for bent functions at given n.

    For bent functions every coefficient has magnitude 2^{-n/2}.
    Prover extraction uses Corollary 5 threshold theta^2/4 on the
    conditional distribution, which corresponds to a theta/2 boundary
    on coefficient magnitude (via Theorem 4).
    Guaranteed inclusion zone: |g_hat(s)| >= theta.
    Guaranteed exclusion zone: |g_hat(s)| < theta/2.
    Uncertain zone: theta/2 <= |g_hat(s)| < theta.
    The Parseval bound caps |L| at 4/theta^2.
    So predicted |L| = min(2^n, floor(4/theta^2)) when above threshold.
    If 2^{-n/2} < theta/2, no coefficients meet the threshold -> |L| = 0.
    """
    coeff_mag = 2 ** (-n / 2)
    threshold = theta / 2
    if coeff_mag >= threshold:
        return min(2**n, math.floor(4 / theta**2))
    else:
        return 0


def crossover_n(theta: float) -> float:
    """The n at which 2^{-n/2} = theta/2, i.e. the detection crossover."""
    return 2 * math.log2(2 / theta)


def iqr(values: list[float]) -> tuple[float, float]:
    """Return (Q1, Q3) of a list of values."""
    if len(values) < 4:
        return (min(values), max(values))
    import statistics
    q = statistics.quantiles(values, n=4)
    return (q[0], q[2])


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

def aggregate(trials: list[dict]) -> dict[int, dict]:
    """Build per-n aggregates from trial data.

    Returns dict[n] -> {
        "trials": int,
        "accepted": int,
        "acceptance_rate": float,
        "list_sizes": list,
        "copies": list,
        "times": list,
        "outcomes": dict[str, int],
    }
    """
    import statistics

    by_n: dict[int, list[dict]] = defaultdict(list)
    for t in trials:
        by_n[t["n"]].append(t)

    result = {}
    for n in sorted(by_n):
        ts = by_n[n]
        acc = sum(1 for t in ts if t.get("accepted", False))
        list_sizes = [t.get("listSize", 0) for t in ts]
        copies = [t["totalCopies"] for t in ts]
        times = [t["totalTimeS"] for t in ts]
        outcomes: dict[str, int] = defaultdict(int)
        for t in ts:
            outcomes[t["outcome"]] += 1

        q1, q3 = iqr(list_sizes)
        result[n] = {
            "trials": len(ts),
            "accepted": acc,
            "acceptance_rate": acc / len(ts),
            "list_sizes": list_sizes,
            "list_size_median": statistics.median(list_sizes),
            "list_size_q1": q1,
            "list_size_q3": q3,
            "copies": copies,
            "copies_median": statistics.median(copies),
            "times": times,
            "time_median": statistics.median(times),
            "outcomes": dict(outcomes),
        }
    return result


# ---------------------------------------------------------------------------
# Artefact 1: List size growth
# ---------------------------------------------------------------------------

def plot_list_size_growth(agg: dict[int, dict], ns: list[int]) -> None:
    """Line plot (log-scale y): median |L| vs n with theory overlay."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colours = sns.color_palette("colorblind", 5)

    # Empirical median list size
    medians = [agg[n]["list_size_median"] for n in ns]
    q1s = [agg[n]["list_size_q1"] for n in ns]
    q3s = [agg[n]["list_size_q3"] for n in ns]

    # For log-scale, replace 0 with a small positive number for plotting
    # but mark them differently
    plot_medians = [max(m, 0.5) for m in medians]

    ax.plot(ns, plot_medians, "o-", color=colours[0], linewidth=2,
            markersize=7, label=r"Bent $|L|$ (median)", zorder=5)
    # IQR shading
    plot_q1 = [max(q, 0.5) for q in q1s]
    plot_q3 = [max(q, 0.5) for q in q3s]
    ax.fill_between(ns, plot_q1, plot_q3, alpha=0.2, color=colours[0],
                     label="IQR")

    # Theoretical prediction
    theory = [theoretical_list_size(n, THETA) for n in ns]
    # Replace 0 with 0.5 for plotting on log scale
    plot_theory = [max(t, 0.5) for t in theory]
    ax.plot(ns, plot_theory, "s--", color=colours[1], linewidth=1.5,
            markersize=6, label=r"Theory: $\min(2^n,\, 4/\theta^2)$ (Parseval bound)", zorder=4)

    # 2^n curve for reference
    two_n = [2**n for n in ns]
    ax.plot(ns, two_n, ":", color="grey", linewidth=1, alpha=0.6,
            label=r"$2^n$ (total coefficients)")

    # Single-parity comparison (|L| = 1)
    ax.axhline(1, ls="-.", color=colours[2], alpha=0.7, linewidth=1.2,
               label=r"Single parity $|L| = 1$")

    # 4/theta^2 Parseval bound cap
    cap = 4 / THETA**2
    ax.axhline(cap, ls="--", color=colours[3], alpha=0.5, linewidth=1,
               label=rf"$4/\theta^2 \approx {cap:.0f}$ (Parseval bound)")

    # Crossover annotation
    x_cross = crossover_n(THETA)
    ax.axvline(x_cross, ls=":", color="red", alpha=0.6, linewidth=1)
    ax.annotate(
        rf"Crossover $n \approx {x_cross:.1f}$" + "\n"
        + rf"$2^{{-n/2}} = \theta/2$ (exclusion boundary)",
        xy=(x_cross, 3), fontsize=8,
        xytext=(x_cross + 1.5, 8),
        arrowprops=dict(arrowstyle="->", color="red", alpha=0.6),
        color="red", alpha=0.8,
    )

    # Audit fix m2 (audit/bent.md): shade the Corollary 5 uncertain
    # band [theta/2, theta) where inclusion in L is permitted but not
    # required.  At n=4 the bent coefficients sit at 2^(-n/2) = 0.25,
    # which lies in [0.15, 0.30) for the default theta = 0.3 -- so
    # the n=4 acceptance is NOT "below the crossover" but rather
    # Corollary 5 extracting from the uncertain band.
    n_band_lo = 2 * math.log2(1.0 / THETA)        # 2^(-n/2) = theta
    n_band_hi = 2 * math.log2(2.0 / THETA)        # 2^(-n/2) = theta/2
    ax.axvspan(n_band_lo, n_band_hi, color="orange", alpha=0.10,
               label=rf"Cor. 5 uncertain band $[\theta/2, \theta)$")

    # Mark zero-valued points
    for n_val, med in zip(ns, medians):
        if med == 0:
            ax.annotate("0", (n_val, 0.5), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=7,
                        color=colours[0], alpha=0.8)

    ax.set_yscale("log")
    ax.set_xlabel("$n$ (number of qubits)")
    ax.set_ylabel("$|L|$ (candidate list size)")
    ax.set_xticks(ns)
    ax.set_title("Bent functions: list size growth vs theory")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_ylim(0.3, max(two_n) * 2)

    save(fig, "list_size_growth")


# ---------------------------------------------------------------------------
# Artefact 2: Bent vs parity acceptance
# ---------------------------------------------------------------------------

def plot_bent_vs_parity_acceptance(agg: dict[int, dict], ns: list[int]) -> None:
    """Grouped bar chart: acceptance rate bent vs single parities at each n."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colours = sns.color_palette("colorblind", 3)

    x = np.arange(len(ns))
    width = 0.35

    # Bent acceptance rates
    bent_rates = [agg[n]["acceptance_rate"] for n in ns]
    bent_ci_lo = []
    bent_ci_hi = []
    for n in ns:
        lo, hi = wilson_ci(agg[n]["accepted"], agg[n]["trials"])
        bent_ci_lo.append(max(0, agg[n]["acceptance_rate"] - lo))
        bent_ci_hi.append(max(0, hi - agg[n]["acceptance_rate"]))

    # Parity acceptance: 100% at all n (hardcoded from scaling experiment)
    parity_rates = [1.0] * len(ns)

    ax.bar(x - width / 2, parity_rates, width, label="Single parity (expected)",
           color=colours[0], edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.bar(x + width / 2, bent_rates, width, label="Bent function",
           color=colours[1], edgecolor="white", linewidth=0.5,
           yerr=[bent_ci_lo, bent_ci_hi], capsize=3)

    # Annotate bent rates
    for i, (n_val, rate) in enumerate(zip(ns, bent_rates)):
        label_text = f"{rate*100:.0f}%"
        y_pos = rate + 0.05 if rate > 0 else 0.05
        ax.text(x[i] + width / 2, y_pos, label_text, ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color=colours[1])

    ax.set_xticks(x)
    ax.set_xticklabels([f"$n={n}$" for n in ns])
    ax.set_ylabel("Acceptance rate")
    ax.set_ylim(0, 1.15)
    ax.set_title("Acceptance: bent functions vs single parities")
    ax.legend(fontsize=9, loc="center right")

    # Annotate the crossover
    x_cross = crossover_n(THETA)
    # Find the bar index closest to crossover
    cross_idx = min(range(len(ns)), key=lambda i: abs(ns[i] - x_cross))
    ax.axvline(x[cross_idx] + 0.02, ls=":", color="red", alpha=0.4, linewidth=1)
    ax.text(x[cross_idx] + 0.15, 0.55,
            rf"$n \approx {x_cross:.1f}$" + "\ncrossover",
            fontsize=7, color="red", alpha=0.7, rotation=0)

    save(fig, "bent_vs_parity_acceptance")


# ---------------------------------------------------------------------------
# Artefact 3: Resource explosion
# ---------------------------------------------------------------------------

def plot_resource_explosion(agg: dict[int, dict], ns: list[int]) -> None:
    """Line plot (log-scale y): total_copies and wall-clock time vs n."""
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    colours = sns.color_palette("colorblind", 5)

    import statistics

    # Total copies
    copies_medians = [agg[n]["copies_median"] for n in ns]
    copies_q1 = [statistics.quantiles(agg[n]["copies"], n=4)[0]
                 if len(agg[n]["copies"]) >= 4 else min(agg[n]["copies"])
                 for n in ns]
    copies_q3 = [statistics.quantiles(agg[n]["copies"], n=4)[2]
                 if len(agg[n]["copies"]) >= 4 else max(agg[n]["copies"])
                 for n in ns]

    line1 = ax1.plot(ns, copies_medians, "o-", color=colours[0], linewidth=2,
                     markersize=7, label="Total copies (median)")
    ax1.fill_between(ns, copies_q1, copies_q3, alpha=0.15, color=colours[0])
    ax1.set_xlabel("$n$ (number of qubits)")
    ax1.set_ylabel("Total copies", color=colours[0])
    ax1.tick_params(axis="y", labelcolor=colours[0])
    ax1.set_yscale("log")

    # Wall-clock time on secondary axis
    ax2 = ax1.twinx()
    time_medians = [agg[n]["time_median"] for n in ns]
    time_q1 = [statistics.quantiles(agg[n]["times"], n=4)[0]
               if len(agg[n]["times"]) >= 4 else min(agg[n]["times"])
               for n in ns]
    time_q3 = [statistics.quantiles(agg[n]["times"], n=4)[2]
               if len(agg[n]["times"]) >= 4 else max(agg[n]["times"])
               for n in ns]

    line2 = ax2.plot(ns, time_medians, "s--", color=colours[1], linewidth=2,
                     markersize=6, label="Wall-clock time (median)")
    ax2.fill_between(ns, time_q1, time_q3, alpha=0.15, color=colours[1])
    ax2.set_ylabel("Wall-clock time (s)", color=colours[1])
    ax2.tick_params(axis="y", labelcolor=colours[1])
    ax2.set_yscale("log")

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=8, loc="upper left")

    ax1.set_xticks(ns)
    ax1.set_title("Bent functions: resource cost vs $n$")

    # Annotate the dramatic time explosion
    max_time_n = ns[-1]
    max_time = agg[max_time_n]["time_median"]
    ax2.annotate(
        f"{max_time:.0f}s\n({max_time/60:.0f} min)",
        xy=(max_time_n, max_time),
        xytext=(max_time_n - 2.5, max_time * 0.3),
        fontsize=8, color=colours[1],
        arrowprops=dict(arrowstyle="->", color=colours[1], alpha=0.6),
    )

    fig.tight_layout()
    save(fig, "resource_explosion")


# ---------------------------------------------------------------------------
# Artefact 4: Summary table (CSV)
# ---------------------------------------------------------------------------

def write_summary_table(agg: dict[int, dict], ns: list[int]) -> None:
    """Write CSV summary: per-n bent function experiment metrics."""
    path = OUT_DIR / "bent_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n",
            "listSize_median", "listSize_Q1", "listSize_Q3",
            "acceptance_pct",
            "copies_median",
            "time_median_s",
            "theory_listSize",
            "coeff_magnitude",
            "threshold_theta_over_2",
            "above_threshold",
            "dominant_outcome",
        ])
        for n in ns:
            a = agg[n]
            theory_ls = theoretical_list_size(n, THETA)
            coeff_mag = 2 ** (-n / 2)
            threshold = THETA / 2
            above = coeff_mag >= threshold

            # Find dominant outcome
            dominant = max(a["outcomes"], key=a["outcomes"].get)

            writer.writerow([
                n,
                f"{a['list_size_median']:.0f}",
                f"{a['list_size_q1']:.0f}",
                f"{a['list_size_q3']:.0f}",
                f"{a['acceptance_rate']*100:.0f}",
                f"{a['copies_median']:.0f}",
                f"{a['time_median']:.3f}",
                f"{theory_ls:.0f}",
                f"{coeff_mag:.6f}",
                f"{threshold:.4f}",
                "yes" if above else "no",
                dominant,
            ])
    print(f"  {path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_style()

    print("Loading bent function data...")
    data = load_data()
    trials = data["trials"]
    params = data["parameters"]
    print(f"  {len(trials)} trials, n in {params['nRange']}, "
          f"theta={params['theta']}, epsilon={params['epsilon']}")

    ns = sorted(set(t["n"] for t in trials))
    agg = aggregate(trials)

    # Print quick summary
    x_cross = crossover_n(THETA)
    print(f"\n  Crossover n = 2*log2(2/theta) = {x_cross:.2f}")
    print(f"  theta/2 = {THETA/2}")
    print(f"  4/theta^2 = {4/THETA**2:.1f}")
    for n in ns:
        a = agg[n]
        coeff = 2**(-n/2)
        print(f"  n={n:2d}: |coeff|={coeff:.4f}, |L|={a['list_size_median']:.0f}, "
              f"accept={a['acceptance_rate']*100:.0f}%, "
              f"time={a['time_median']:.1f}s, "
              f"outcome={list(a['outcomes'].keys())}")

    print("\nGenerating artefacts:")
    plot_list_size_growth(agg, ns)
    plot_bent_vs_parity_acceptance(agg, ns)
    plot_resource_explosion(agg, ns)
    write_summary_table(agg, ns)

    print("\nDone.")


if __name__ == "__main__":
    main()
