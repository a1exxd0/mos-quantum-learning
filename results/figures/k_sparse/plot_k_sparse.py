"""Generate all k-Sparse experiment artefacts.

Artefacts produced:
  1. acceptance_vs_n_by_k.{pdf,png}     - Line plot: acceptance rate vs n per k
  2. list_size_vs_k.{pdf,png}           - Grouped bar: median |L| by k at select n
  3. misclassification_heatmap.{pdf,png} - Heatmap: empirical error rate (n x k)
  4. k_sparse_summary.csv               - Per (n, k) summary table

Usage:
    uv run python results/figures/k_sparse/plot_k_sparse.py
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
DATA_FILE = PROJECT_ROOT / "results" / "k_sparse_4_16_100.pb"
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


def extract_k(phi: str) -> int:
    """Extract k from phiDescription like 'k_sparse_k=4_n=8'."""
    m = re.search(r"k=(\d+)", phi)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot extract k from phiDescription: {phi}")


def load_data() -> dict:
    """Deserialise the k_sparse .pb file to a Python dict."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.decode import decode

    return json.loads(decode(DATA_FILE))


def build_tables(trials: list[dict]) -> dict:
    """Aggregate trial-level data into per-(k, n) summaries."""
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for t in trials:
        k = extract_k(t["phiDescription"])
        n = t["n"]
        groups[(k, n)].append(t)

    ks = sorted({k for k, _ in groups})
    ns = sorted({n for _, n in groups})

    summaries: dict[tuple[int, int], dict] = {}
    for (k, n), ts in groups.items():
        total = len(ts)
        accepted = sum(1 for t in ts if t.get("accepted", False))
        correct = sum(1 for t in ts if t.get("hypothesisCorrect", False))
        list_sizes = [t["listSize"] for t in ts]
        copies = [t["totalCopies"] for t in ts]
        weights = [t["accumulatedWeight"] for t in ts]
        thresholds = [t["acceptanceThreshold"] for t in ts]
        # Audit fix m3 (audit/k_sparse.md): expose b_sq so the
        # list-size bound overlay can use the actually-enforced
        # 64*b^2/theta^2 (Theorem 10/15) instead of 4/theta^2
        # (Theorem 7).  Dirichlet draws give per-trial b_sq, take
        # the mean across trials in this cell.
        b_sq_vals = [t.get("bSq", 1.0) for t in ts]

        # Misclassification: use misclassificationRate if available,
        # otherwise use 1 - correctness as proxy
        misclass_trials = [t for t in ts if "misclassificationRate" in t]
        if misclass_trials:
            misclass_rates = [t["misclassificationRate"] for t in misclass_trials]
            avg_misclass = sum(misclass_rates) / len(misclass_rates)
        else:
            # proxy: fraction of trials where hypothesis was incorrect
            avg_misclass = 1.0 - correct / total if total > 0 else float("nan")

        # Rejection mechanism breakdown
        rej_list = sum(1 for t in ts if t["outcome"] == "reject_list_too_large")
        rej_weight = sum(
            1 for t in ts if t["outcome"] == "reject_insufficient_weight"
        )

        acc_lo, acc_hi = wilson_ci(accepted, total)
        corr_lo, corr_hi = wilson_ci(correct, total)

        summaries[(k, n)] = {
            "total": total,
            "accepted": accepted,
            "correct": correct,
            "acc_rate": accepted / total if total else 0,
            "acc_ci": (acc_lo, acc_hi),
            "corr_rate": correct / total if total else 0,
            "corr_ci": (corr_lo, corr_hi),
            "median_list": float(np.median(list_sizes)),
            "mean_list": float(np.mean(list_sizes)),
            "median_copies": float(np.median(copies)),
            "mean_copies": float(np.mean(copies)),
            "avg_misclass": avg_misclass,
            "median_weight": float(np.median(weights)),
            "mean_threshold": float(np.mean(thresholds)),
            "rej_list": rej_list,
            "rej_weight": rej_weight,
            "theta": ts[0]["theta"],
            "mean_b_sq": float(np.mean(b_sq_vals)),
        }

    return {"summaries": summaries, "ks": ks, "ns": ns}


# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------


def setup_style() -> None:
    sns.set_context("paper", font_scale=1.1)
    sns.set_palette("colorblind")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path)
        print(f"  {path.relative_to(PROJECT_ROOT)}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Artefact 1: Acceptance vs n by k (line plot with Wilson CI bands)
# ---------------------------------------------------------------------------


def plot_acceptance_vs_n(summaries: dict, ks: list[int], ns: list[int]) -> None:
    """Line plot: acceptance rate vs n, one line per k, with 95% Wilson CI."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colours = sns.color_palette("colorblind", len(ks))
    markers = ["o", "s", "D", "^"]

    for i, k in enumerate(ks):
        rates = []
        ci_lo = []
        ci_hi = []
        for n in ns:
            s = summaries.get((k, n))
            if s is None:
                rates.append(float("nan"))
                ci_lo.append(float("nan"))
                ci_hi.append(float("nan"))
            else:
                rates.append(s["acc_rate"])
                ci_lo.append(s["acc_ci"][0])
                ci_hi.append(s["acc_ci"][1])

        rates_arr = np.array(rates)
        ci_lo_arr = np.array(ci_lo)
        ci_hi_arr = np.array(ci_hi)

        ax.plot(
            ns,
            rates_arr,
            marker=markers[i % len(markers)],
            label=f"$k={k}$",
            color=colours[i],
            linewidth=1.5,
            markersize=5,
        )
        ax.fill_between(ns, ci_lo_arr, ci_hi_arr, alpha=0.15, color=colours[i])

    ax.axhline(
        0.9,
        ls="--",
        color="grey",
        alpha=0.6,
        label=r"$1-\delta=0.9$ (nominal Thm 10/15 target)",
    )
    ax.set_xlabel("$n$ (number of qubits)")
    ax.set_ylabel("Acceptance rate")
    ax.set_ylim(-0.02, 1.08)
    ax.set_xticks(ns)
    ax.legend(fontsize=8, loc="lower left")
    # Audit fix M1 (audit/k_sparse.md): the experiment intentionally
    # operates outside the Definition 11 promise -- Dirichlet(1,...,1)
    # routinely produces coefficients below theta -- so the
    # 1-delta=0.9 target is not enforced for all cells.  This
    # acceptance ceiling is a structural consequence of being
    # off-promise, not a protocol failure.
    ax.set_title(
        "$k$-Sparse completeness: acceptance rate vs $n$\n"
        "(off-promise: Dirichlet$(1,\\dots,1)$ targets often have "
        "$c_{\\min} < \\theta$)"
    )
    fig.tight_layout()
    save(fig, "acceptance_vs_n_by_k")


# ---------------------------------------------------------------------------
# Artefact 2: List size vs k (grouped bar with theoretical bound overlay)
# ---------------------------------------------------------------------------


def plot_list_size_vs_k(summaries: dict, ks: list[int], ns: list[int]) -> None:
    """Grouped bar chart at representative n values, bars grouped by k,
    showing median |L|. Overlay 4/theta^2 upper bound."""
    # Pick representative n values
    ns_show = [n for n in [4, 8, 12, 16] if n in ns]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(ks))
    width = 0.8 / len(ns_show)
    colours = sns.color_palette("colorblind", len(ns_show))

    for i, n in enumerate(ns_show):
        medians = []
        for k in ks:
            s = summaries.get((k, n))
            medians.append(s["median_list"] if s else 0)
        offset = (i - len(ns_show) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            medians,
            width * 0.9,
            label=f"$n={n}$",
            color=colours[i],
            edgecolor="white",
            linewidth=0.5,
        )

    # Overlay actually-enforced bound 64*b_sq/theta^2 per k.
    # Audit fix m3 (audit/k_sparse.md): the previous version overlaid
    # 4/theta^2 (from Theorem 7, the SQ verifier) but the experiment
    # actually runs the random-example verifier from Theorem 10/15
    # which uses 64*b^2/theta^2.  ``b_sq`` is read from the trial
    # summaries; for the experiment's Dirichlet targets, b_sq varies
    # per trial so we take the mean over the displayed cells.
    for j, k in enumerate(ks):
        theta = None
        b_sq_vals: list[float] = []
        for n in ns:
            s = summaries.get((k, n))
            if s:
                if theta is None:
                    theta = s["theta"]
                if "mean_b_sq" in s:
                    b_sq_vals.append(s["mean_b_sq"])
        if theta:
            mean_b_sq = float(np.mean(b_sq_vals)) if b_sq_vals else 1.0
            bound = 64.0 * mean_b_sq / (theta**2)
            ax.plot(
                [j - 0.45, j + 0.45],
                [bound, bound],
                color="red",
                linewidth=1.5,
                linestyle="--",
                zorder=5,
            )
            # Only label on first one
            if j == 0:
                ax.plot(
                    [],
                    [],
                    color="red",
                    linewidth=1.5,
                    linestyle="--",
                    label=r"$64\,b^2/\theta^2$ (Thm 10/15 list bound)",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([f"$k={k}$" for k in ks])
    ax.set_ylabel("Median list size $|L|$")
    ax.set_xlabel("Sparsity parameter $k$")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("List size vs sparsity $k$ at representative $n$")

    # Use log scale if range is large
    all_vals = [
        summaries[(k, n)]["median_list"]
        for k in ks
        for n in ns
        if (k, n) in summaries
    ]
    if max(all_vals) / max(min(all_vals), 1) > 50:
        ax.set_yscale("log")
        ax.set_ylabel("Median list size $|L|$ (log scale)")

    fig.tight_layout()
    save(fig, "list_size_vs_k")


# ---------------------------------------------------------------------------
# Artefact 3: Misclassification rate heatmap (n x k)
# ---------------------------------------------------------------------------


def plot_misclassification_heatmap(
    summaries: dict, ks: list[int], ns: list[int]
) -> None:
    """Heatmap of empirical misclassification rate (rows=n, cols=k).
    For accepted trials with misclassificationRate, uses that; otherwise
    uses 1-correctness as proxy."""
    matrix = np.full((len(ns), len(ks)), np.nan)

    for i, n in enumerate(ns):
        for j, k in enumerate(ks):
            s = summaries.get((k, n))
            if s:
                matrix[i, j] = s["avg_misclass"]

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0)

    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"$k={k}$" for k in ks])
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels([f"$n={n}$" for n in ns])
    ax.set_xlabel("Sparsity parameter $k$")
    ax.set_ylabel("Number of qubits $n$")

    # Annotate cells
    for i in range(len(ns)):
        for j in range(len(ks)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_colour = "white" if val > 0.3 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_colour,
                )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Misclassification rate")

    # Reference line for epsilon
    ax.set_title(
        r"Empirical misclassification rate ($\varepsilon=0.3$, Thm 15: $\leq 2\cdot\mathrm{opt}+\varepsilon$)"
    )
    fig.tight_layout()
    save(fig, "misclassification_heatmap")


# ---------------------------------------------------------------------------
# Artefact 4: Summary CSV
# ---------------------------------------------------------------------------


def write_summary_csv(summaries: dict, ks: list[int], ns: list[int]) -> None:
    """Write CSV: per (n, k) acceptance%, correctness%, median |L|, median copies."""
    path = OUT_DIR / "k_sparse_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "n",
                "k",
                "theta",
                "acceptance_pct",
                "acc_ci_lower",
                "acc_ci_upper",
                "correctness_pct",
                "corr_ci_lower",
                "corr_ci_upper",
                "median_list_size",
                "median_copies",
                "avg_misclassification",
                "median_weight",
                "mean_threshold",
                "rej_list_count",
                "rej_weight_count",
                "total_trials",
            ]
        )
        for n in ns:
            for k in ks:
                s = summaries.get((k, n))
                if s is None:
                    continue
                writer.writerow(
                    [
                        n,
                        k,
                        f"{s['theta']:.4f}",
                        f"{s['acc_rate'] * 100:.1f}",
                        f"{s['acc_ci'][0] * 100:.1f}",
                        f"{s['acc_ci'][1] * 100:.1f}",
                        f"{s['corr_rate'] * 100:.1f}",
                        f"{s['corr_ci'][0] * 100:.1f}",
                        f"{s['corr_ci'][1] * 100:.1f}",
                        f"{s['median_list']:.0f}",
                        f"{s['median_copies']:.0f}",
                        f"{s['avg_misclass']:.4f}",
                        f"{s['median_weight']:.4f}",
                        f"{s['mean_threshold']:.4f}",
                        s["rej_list"],
                        s["rej_weight"],
                        s["total"],
                    ]
                )
    print(f"  {path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    setup_style()

    print("Loading k-sparse data...")
    data = load_data()
    trials = data["trials"]
    params = data["parameters"]
    print(
        f"  {len(trials)} trials, k in {params['kValues']}, "
        f"n in {params['nRange'][0]}..{params['nRange'][-1]}"
    )

    tables = build_tables(trials)
    summaries = tables["summaries"]
    ks = tables["ks"]
    ns = tables["ns"]

    print("\nGenerating artefacts:")
    plot_acceptance_vs_n(summaries, ks, ns)
    plot_list_size_vs_k(summaries, ks, ns)
    plot_misclassification_heatmap(summaries, ks, ns)
    write_summary_csv(summaries, ks, ns)

    # Print analysis summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    print("\n1. Completeness (acceptance rate):")
    for k in ks:
        rates = [summaries[(k, n)]["acc_rate"] for n in ns if (k, n) in summaries]
        avg = sum(rates) / len(rates) if rates else 0
        print(f"   k={k}: mean acceptance = {avg*100:.1f}%  (Thm 15 predicts >= 90%)")

    print("\n2. List size growth:")
    for k in ks:
        meds = [summaries[(k, n)]["median_list"] for n in ns if (k, n) in summaries]
        theta = summaries[(k, ns[0])]["theta"]
        bound = 4 / theta**2
        print(
            f"   k={k} (theta={theta:.4f}): median |L| range [{min(meds):.0f}, {max(meds):.0f}], "
            f"4/theta^2 = {bound:.1f} (Parseval bound)"
        )

    print("\n3. Misclassification:")
    for k in ks:
        miscs = [
            summaries[(k, n)]["avg_misclass"] for n in ns if (k, n) in summaries
        ]
        avg = sum(miscs) / len(miscs) if miscs else 0
        print(
            f"   k={k}: mean misclassification = {avg:.4f}  "
            f"(Thm 15: <= 2*opt + eps = {0.3:.1f} for opt~0)"
        )

    print("\n4. Weight threshold binding constraint:")
    for k in ks:
        rej_w = sum(summaries[(k, n)]["rej_weight"] for n in ns if (k, n) in summaries)
        total = sum(summaries[(k, n)]["total"] for n in ns if (k, n) in summaries)
        print(
            f"   k={k}: {rej_w}/{total} trials rejected by weight check "
            f"({rej_w/total*100:.1f}%)"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
