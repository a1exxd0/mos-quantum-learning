"""Generate Exp 3 (Gate Noise) artefacts.

Artefacts produced:
  1. gate_noise_acceptance.{pdf,png}  -- Line plot: acceptance rate vs gate error rate p
  2. gate_noise_summary.csv           -- Per (n, p) acceptance %, correctness %

Usage:
    uv run python results/figures/gate_noise/plot_gate_noise.py
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
DATA_FILE = PROJECT_ROOT / "results" / "gate_noise_4_8_50.pb"
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
    """Deserialise the gate_noise .pb file to a Python dict."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.decode import decode
    return json.loads(decode(DATA_FILE))


def extract_error_rate(phi_desc: str) -> float:
    """Extract the gate error rate p from phiDescription like 'gate_noise_p=0.01_s=12'."""
    m = re.search(r"p=([0-9.]+)", phi_desc)
    if m:
        return float(m.group(1))
    raise ValueError(f"Cannot extract error rate from: {phi_desc}")


def build_tables(
    trials: list[dict],
) -> tuple[
    dict[tuple[int, float], dict],  # stats[(n, p)] = {total, accepted, correct}
    list[int],                       # sorted n values
    list[float],                     # sorted p values
]:
    """Aggregate trial-level data into per-(n, p) summaries."""
    stats: dict[tuple[int, float], dict] = defaultdict(
        lambda: {"total": 0, "accepted": 0, "correct": 0}
    )
    for t in trials:
        p = extract_error_rate(t["phiDescription"])
        n = t["n"]
        key = (n, p)
        stats[key]["total"] += 1
        if t.get("accepted", False):
            stats[key]["accepted"] += 1
        if t.get("hypothesisCorrect", False):
            stats[key]["correct"] += 1

    ns = sorted({n for (n, _) in stats})
    ps = sorted({p for (_, p) in stats})
    return dict(stats), ns, ps


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
# Artefact 1: Gate noise acceptance rate line plot
# ---------------------------------------------------------------------------

def plot_gate_noise_acceptance(
    stats: dict[tuple[int, float], dict],
    ns: list[int],
    ps: list[float],
) -> None:
    """Line plot: acceptance rate vs gate error rate p, one line per n, with Wilson CI."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colours = sns.color_palette("colorblind", len(ns))
    markers = ["o", "s", "D", "^", "v", "P", "X"]

    # Filter out p=0.0 (can't appear on log axis; trivially 100% for all n)
    ps_log = [p for p in ps if p > 0]

    for i, n in enumerate(ns):
        rates = []
        ci_lo = []
        ci_hi = []
        for p in ps_log:
            s = stats.get((n, p), {"total": 0, "accepted": 0})
            total = s["total"]
            acc = s["accepted"]
            rate = acc / total if total else 0
            lo, hi = wilson_ci(acc, total)
            rates.append(rate)
            ci_lo.append(lo)
            ci_hi.append(hi)

        rates_arr = np.array(rates)
        ci_lo_arr = np.array(ci_lo)
        ci_hi_arr = np.array(ci_hi)

        ax.plot(
            ps_log, rates_arr,
            marker=markers[i % len(markers)],
            label=f"$n={n}$",
            color=colours[i],
            linewidth=1.5,
            markersize=5,
        )
        ax.fill_between(
            ps_log, ci_lo_arr, ci_hi_arr,
            alpha=0.15, color=colours[i],
        )

    # Threshold reference line
    ax.axhline(0.9, ls="--", color="grey", alpha=0.6, label=r"$1-\delta=0.9$")

    ax.set_xscale("log")
    ax.set_xlabel("Gate error rate $p$")
    ax.set_ylabel("Acceptance rate")
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(ps_log[0] * 0.7, ps_log[-1] * 1.4)
    ax.legend(fontsize=8, loc="center right")
    ax.set_title(
        "Gate noise: acceptance rate vs depolarising error rate\n"
        r"\textit{(No theoretical prediction --- novel empirical contribution)}"
        if plt.rcParams.get("text.usetex", False)
        else "Gate noise: acceptance rate vs depolarising error rate\n"
        "(No theoretical prediction -- novel empirical contribution)",
        fontsize=10,
    )

    ax.set_xticks(ps_log)
    ax.set_xticklabels([f"{p:g}" for p in ps_log], fontsize=7)
    ax.minorticks_off()

    save(fig, "gate_noise_acceptance")


# ---------------------------------------------------------------------------
# Artefact 2: Summary table (CSV)
# ---------------------------------------------------------------------------

def write_summary_table(
    stats: dict[tuple[int, float], dict],
    ns: list[int],
    ps: list[float],
) -> None:
    """Write CSV summary: per (n, p) acceptance %, correctness %, with Wilson CIs."""
    path = OUT_DIR / "gate_noise_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n", "gate_error_rate", "trials",
            "acceptance_pct", "acc_ci_lower", "acc_ci_upper",
            "correctness_pct", "cor_ci_lower", "cor_ci_upper",
        ])
        for n in ns:
            for p in ps:
                s = stats.get((n, p), {"total": 0, "accepted": 0, "correct": 0})
                total = s["total"]
                acc = s["accepted"]
                cor = s["correct"]
                acc_rate = acc / total if total else 0
                cor_rate = cor / total if total else 0
                acc_lo, acc_hi = wilson_ci(acc, total)
                cor_lo, cor_hi = wilson_ci(cor, total)
                writer.writerow([
                    n, p, total,
                    f"{acc_rate * 100:.1f}", f"{acc_lo * 100:.1f}", f"{acc_hi * 100:.1f}",
                    f"{cor_rate * 100:.1f}", f"{cor_lo * 100:.1f}", f"{cor_hi * 100:.1f}",
                ])
    print(f"  {path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_style()

    print("Loading gate noise data...")
    data = load_data()
    trials = data["trials"]
    params = data["parameters"]
    print(f"  {len(trials)} trials, n in {params['nRange'][0]}..{params['nRange'][-1]}, "
          f"gate noise rates = {params['gateNoiseRates']}")

    stats, ns, ps = build_tables(trials)

    print("\nGenerating artefacts:")
    plot_gate_noise_acceptance(stats, ns, ps)
    write_summary_table(stats, ns, ps)

    # --- Print analysis summary ---
    print("\n" + "=" * 70)
    print("GATE NOISE ANALYSIS SUMMARY")
    print("=" * 70)
    print()
    print("Note: No direct theoretical prediction exists in Caro et al. for")
    print("depolarising circuit noise; this is a novel empirical contribution.")
    print()

    # Identify threshold behaviour per n
    for n in ns:
        # Find lowest p where acceptance drops below 50%
        threshold_p = None
        for p in ps:
            s = stats.get((n, p), {"total": 0, "accepted": 0})
            rate = s["accepted"] / s["total"] if s["total"] else 0
            if rate < 0.5:
                threshold_p = p
                break
        if threshold_p is not None:
            # What was rate at previous p?
            prev_idx = ps.index(threshold_p) - 1
            if prev_idx >= 0:
                prev_p = ps[prev_idx]
                prev_s = stats.get((n, prev_p), {"total": 0, "accepted": 0})
                prev_rate = prev_s["accepted"] / prev_s["total"] if prev_s["total"] else 0
                curr_s = stats[(n, threshold_p)]
                curr_rate = curr_s["accepted"] / curr_s["total"]
                print(f"  n={n}: sharp drop at p={threshold_p} "
                      f"(prev p={prev_p}: {prev_rate*100:.0f}% -> {curr_rate*100:.0f}%)")
            else:
                print(f"  n={n}: drops below 50% at p={threshold_p}")
        else:
            all_rates = []
            for p in ps:
                s = stats.get((n, p), {"total": 0, "accepted": 0})
                rate = s["accepted"] / s["total"] if s["total"] else 0
                all_rates.append(rate)
            min_rate = min(all_rates)
            print(f"  n={n}: remains above 50% for all p (min rate = {min_rate*100:.0f}%)")

    print()
    print("Key findings:")
    print("  - Depolarising gate noise produces a sharp accept->reject")
    print("    transition with minimal intermediate degradation.")
    print()
    print("CAVEATS (audit/gate_noise.md, exploratory experiment):")
    print("  - Caro et al. only analyse classical label-flip noise (Def 5(iii)).")
    print("    The paper makes NO predictions about per-gate depolarising noise;")
    print("    this experiment is exploratory, NOT a validation of any theorem.")
    print("  - The observed 'threshold' is dominated by truth-table oracle")
    print("    synthesis cost in mos.MoSState._circuit_oracle_f, which emits")
    print("    up to 2^n multi-controlled-X gates per shot.  Errors-per-shot")
    print("    scale as ~p*n*2^n -- exponential in n, which the paper's prover")
    print("    complexity does NOT predict.  The 'larger n is more sensitive'")
    print("    finding is therefore a synthesis-cost artefact, not a property")
    print("    of the verification protocol.")
    print("  - p in {0.2, 0.5} is unphysical; real 2q gate error rates are")
    print("    ~1e-4 to ~1e-2.")
    print("  - Noise is applied only to h, x, cx; transpiled basis gates")
    print("    (sx, rz, u3, t, tdg) carry no error, so the effective per-")
    print("    logical-op error is smaller than nominal p.")
    print()

    print("Done.")


if __name__ == "__main__":
    main()
