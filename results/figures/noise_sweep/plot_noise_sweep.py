"""Generate all Noise Sweep (label-flip noise) artefacts.

Artefacts produced:
  1. noise_heatmap.{pdf,png}               – Heatmap (n x eta): acceptance rate
  2. acceptance_correctness_vs_eta.{pdf,png} – Lines for representative n values
  3. fourier_weight_attenuation.{pdf,png}   – Median weight vs eta with theory curve
  4. breakdown_points.csv                   – Per-n empirical & theoretical breakdown

Usage:
    uv run python results/figures/noise_sweep/plot_noise_sweep.py
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DATA_FILE = PROJECT_ROOT / "results" / "noise_sweep_4_16_100.pb"
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
    """Deserialise the noise_sweep .pb file to a Python dict."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.decode import decode
    return json.loads(decode(DATA_FILE))


def parse_eta(phi_description: str) -> float:
    """Extract eta value from phiDescription like 'noisy_parity_eta=0.2_s=12'."""
    m = re.search(r"eta=([\d.]+)", phi_description)
    return float(m.group(1)) if m else 0.0


def build_tables(trials: list[dict], epsilon: float) -> dict:
    """Aggregate trial-level data into per-(n, eta) summaries.

    Proto3 JSON encoding omits zero/False defaults, so post-breakdown trials
    where the prover emitted an empty list and the verifier vacuously accepted
    a negative threshold appear with ``listSize``, ``accumulatedWeight`` and
    ``hypothesisCorrect`` all missing (``None``).  This function separates
    three things readers of the figures need to distinguish:

    * ``acceptance_rate`` — raw ``accepted`` fraction.  Includes vacuous
      accepts at ``eta > eta_max`` where the threshold ``a^2 - eps^2/8`` is
      negative and any empty list trivially passes.
    * ``correctness_rate`` — fraction whose hypothesis parity equals the
      target.  Drops to 0 in the vacuous regime because the degenerate
      hypothesis ``s = 0`` is never the target.
    * ``joint_rate`` — the operationally honest "protocol worked" event
      ``accepted AND hypothesisCorrect``.  This is the only metric that
      collapses both the vacuous-accept spike and the mid-eta variance dip
      into a single signal.
    * ``filtered_median_weight`` — median accumulated weight computed only
      over trials where the prover returned a non-empty list AND the verifier
      did not reject for ``reject_list_too_large``.  This filter is essential:
      without it, the high-eta median collapses to 0 (because a majority of
      trials have ``accumulatedWeight = 0`` from empty-list handling) and the
      figure spuriously appears to diverge from Lemma 6's ``(1-2*eta)^2``.
    * ``empty_list_fraction`` — fraction of trials where the honest prover
      emitted an empty list (``listSize == 0``) or the verifier rejected for
      list-too-large.  This is the honest prover-side breakdown signal; it
      rises sharply at eta ~ (1 - theta/2) / 2 ~ 0.425 when the prover's
      extraction threshold ``theta**2 / 4`` exceeds the QFS mass ``(1-2*eta)**2``.
    """
    # Group trials by (n, eta)
    groups: dict[tuple[int, float], list[dict]] = defaultdict(list)
    for t in trials:
        eta = parse_eta(t["phiDescription"])
        groups[(t["n"], eta)].append(t)

    ns = sorted({n for (n, _) in groups})
    etas = sorted({eta for (_, eta) in groups})

    # Extract theta (fixed across the sweep by the Tier-2 fix; verify).
    thetas = {float(t.get("theta", 0.0)) for t in trials}
    if len(thetas) != 1:
        raise RuntimeError(
            f"expected a single theta across the sweep, got {sorted(thetas)}"
        )
    theta = next(iter(thetas))

    # Per-(n, eta) aggregates
    acceptance_rate: dict[tuple[int, float], float] = {}
    correctness_rate: dict[tuple[int, float], float] = {}
    joint_rate: dict[tuple[int, float], float] = {}
    acceptance_ci: dict[tuple[int, float], tuple[float, float]] = {}
    correctness_ci: dict[tuple[int, float], tuple[float, float]] = {}
    joint_ci: dict[tuple[int, float], tuple[float, float]] = {}
    median_weight: dict[tuple[int, float], float] = {}
    filtered_median_weight: dict[tuple[int, float], float] = {}
    median_threshold: dict[tuple[int, float], float] = {}
    empty_list_fraction: dict[tuple[int, float], float] = {}

    for key, group in groups.items():
        total = len(group)
        accepted = sum(1 for t in group if t.get("accepted", False))
        correct = sum(1 for t in group if t.get("hypothesisCorrect", False))
        joint = sum(
            1 for t in group
            if t.get("accepted", False) and t.get("hypothesisCorrect", False)
        )

        acceptance_rate[key] = accepted / total
        correctness_rate[key] = correct / total
        joint_rate[key] = joint / total
        acceptance_ci[key] = wilson_ci(accepted, total)
        correctness_ci[key] = wilson_ci(correct, total)
        joint_ci[key] = wilson_ci(joint, total)

        # Raw median: treats missing fields as 0.0 (the proto3 default).
        # This is the collapse-to-zero pathway the fix replaces; kept only for
        # the per-cell report diagnostics.
        weights = [t.get("accumulatedWeight") or 0.0 for t in group]
        thresholds = [t.get("acceptanceThreshold") or 0.0 for t in group]
        median_weight[key] = float(np.median(weights))
        median_threshold[key] = float(np.median(thresholds))

        # Filtered median: only trials where the prover returned a non-empty
        # list AND the verifier did not reject for list-too-large.  These are
        # the trials whose accumulated weight is a real Lemma-6 measurement.
        informative = [
            t for t in group
            if (t.get("listSize") or 0) > 0
            and t.get("outcome") != "reject_list_too_large"
        ]
        if informative:
            filt_weights = [t.get("accumulatedWeight") or 0.0 for t in informative]
            filtered_median_weight[key] = float(np.median(filt_weights))
        else:
            filtered_median_weight[key] = float("nan")

        # Empty-list / list-too-large fraction.  This is the prover-side
        # breakdown signal.
        empty = sum(
            1 for t in group
            if (t.get("listSize") or 0) == 0
            or t.get("outcome") == "reject_list_too_large"
        )
        empty_list_fraction[key] = empty / total

    return {
        "ns": ns,
        "etas": etas,
        "acceptance_rate": acceptance_rate,
        "correctness_rate": correctness_rate,
        "joint_rate": joint_rate,
        "acceptance_ci": acceptance_ci,
        "correctness_ci": correctness_ci,
        "joint_ci": joint_ci,
        "median_weight": median_weight,
        "filtered_median_weight": filtered_median_weight,
        "median_threshold": median_threshold,
        "empty_list_fraction": empty_list_fraction,
        "epsilon": epsilon,
        "theta": theta,
    }


def breakdown_points(epsilon: float, theta: float) -> tuple[float, float]:
    """Return the two distinct breakdown eta values in the sweep.

    Returns a pair ``(eta_prover, eta_max)`` where:

    * ``eta_prover = (1 - theta/2) / 2`` is the prover-side effective
      breakdown, i.e. the smallest eta at which the honest prover's QFS mass
      at the target ``(1-2*eta)^2`` falls below the extraction threshold
      ``theta**2 / 4``, so ``_extract_heavy_list`` starts emitting empty lists
      with high probability.  This is where Lemma 6 stops being
      operationally observable end-to-end, even though the lemma itself still
      holds at the quantum state level.
    * ``eta_max = (1 - eps/(2*sqrt(2))) / 2`` is the verifier-side algebraic
      breakdown, i.e. the smallest eta at which the Theorem 12 acceptance
      threshold ``(1-2*eta)^2 - eps^2/8`` becomes non-positive.  At
      ``eta > eta_max`` the check ``Sigma xi_hat^2 >= threshold`` is
      vacuously satisfied by any non-negative weight, so the verifier accepts
      the degenerate empty-list hypothesis ``s = 0`` and correctness collapses
      to 0 while acceptance spikes to 1.
    """
    eta_prover = (1.0 - theta / 2.0) / 2.0
    eta_max = (1.0 - epsilon / (2.0 * math.sqrt(2.0))) / 2.0
    return eta_prover, eta_max


BREAKDOWN_FOOTNOTE = (
    "Breakdown mechanisms: (a) mid-eta variance vs slack (eps^2/8 slack vs "
    "O(1/sqrt(m)) squared-estimator noise with m=3000 verifier samples); "
    "(b) prover-side extraction threshold theta^2/4 > (1-2*eta)^2 at "
    "eta > eta_prover; (c) verifier-side threshold inversion at eta > eta_max."
)


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
# Artefact 1: Noise heatmap (n x eta) – acceptance rate
# ---------------------------------------------------------------------------

def _overlay_vacuous_hatch(ax, etas, ns, epsilon: float) -> None:
    """Draw a grey cross-hatch overlay on cells where the Theorem 12 weight
    threshold ``(1-2*eta)**2 - eps**2/8`` is non-positive, i.e. the verifier
    is in the vacuous-accept regime.  Any non-negative accumulated weight
    (in particular the empty-list ``0.0``) satisfies the check, so acceptance
    rate in these cells is not a protocol signal.
    """
    for j, eta in enumerate(etas):
        threshold = (1.0 - 2.0 * eta) ** 2 - epsilon**2 / 8.0
        if threshold <= 0.0:
            for i in range(len(ns)):
                rect = mpatches.Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    facecolor="none",
                    edgecolor="black",
                    hatch="///",
                    linewidth=0.0,
                    alpha=0.55,
                )
                ax.add_patch(rect)


def plot_noise_heatmap(tables: dict) -> None:
    """Two-panel heatmap: raw acceptance (left) and correctness (right).

    The left panel is the quantity the existing audit plotted.  Its
    post-breakdown cells (``eta >= 0.46``) are coloured bright green for
    ``accepted = 100%`` even though the verifier is vacuously accepting
    empty lists against a negative threshold.  To flag this, those cells are
    overlaid with a grey cross-hatch.

    The right panel plots ``correctness = Pr[hypothesisCorrect]`` — i.e. the
    operationally honest "protocol recovered the target parity" event.  In
    the vacuous-accept regime this collapses to 0 because the degenerate
    empty-list hypothesis ``s = 0`` is never the target (``make_random_parity``
    draws from ``{1, ..., 2^n - 1}``).  The divergence between the two panels
    at ``eta in {0.46, 0.48}`` is the finding.
    """
    ns = tables["ns"]
    etas = tables["etas"]
    acc = tables["acceptance_rate"]
    corr = tables["correctness_rate"]
    epsilon = tables["epsilon"]
    theta = tables["theta"]

    acc_matrix = np.zeros((len(ns), len(etas)))
    corr_matrix = np.zeros((len(ns), len(etas)))
    for i, n in enumerate(ns):
        for j, eta in enumerate(etas):
            acc_matrix[i, j] = acc.get((n, eta), 0.0)
            corr_matrix[i, j] = corr.get((n, eta), 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    eta_prover, eta_max = breakdown_points(epsilon, theta)

    for ax, matrix, title in (
        (axes[0], acc_matrix, r"Acceptance rate"),
        (axes[1], corr_matrix, r"Correctness rate (hypothesis matches target)"),
    ):
        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            origin="lower",
        )

        for i in range(len(ns)):
            for j in range(len(etas)):
                val = matrix[i, j]
                text_colour = "black" if 0.3 < val < 0.8 else "white"
                ax.text(j, i, f"{val*100:.0f}%", ha="center", va="center",
                        fontsize=7, color=text_colour, fontweight="bold")

        _overlay_vacuous_hatch(ax, etas, ns, epsilon)

        ax.set_xticks(range(len(etas)))
        ax.set_xticklabels([f"{e:.2f}" for e in etas], rotation=45)
        ax.set_xlabel(r"Noise rate $\eta$")
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label(title.split("(")[0].strip())

    axes[0].set_yticks(range(len(ns)))
    axes[0].set_yticklabels([str(n) for n in ns])
    axes[0].set_ylabel(r"Dimension $n$")

    # Legend proxy for the hatch overlay
    hatch_patch = mpatches.Patch(
        facecolor="white",
        edgecolor="black",
        hatch="///",
        label=(
            rf"Vacuous regime $\eta > \eta_{{\max}} = {eta_max:.3f}$"
            "\n"
            rf"(threshold $(1-2\eta)^2 - \varepsilon^2/8 \leq 0$)"
        ),
    )
    axes[1].legend(handles=[hatch_patch], loc="upper left", fontsize=7,
                   framealpha=0.9)

    fig.suptitle(
        r"Label-flip noise sweep: acceptance vs correctness",
        fontsize=13,
        y=1.02,
    )
    fig.text(
        0.5,
        -0.04,
        BREAKDOWN_FOOTNOTE,
        ha="center",
        va="top",
        fontsize=7,
        style="italic",
        color="#333",
        wrap=True,
    )
    fig.tight_layout()
    save(fig, "noise_heatmap")


# ---------------------------------------------------------------------------
# Artefact 2: Acceptance & correctness vs eta for representative n
# ---------------------------------------------------------------------------

def plot_acceptance_correctness_vs_eta(tables: dict) -> None:
    """Line plot: acceptance, correctness, and joint rate vs eta.

    Three curves per selected n:

    * Solid ``acceptance`` (o) — raw ``accepted`` rate.  Spikes back to 1.0
      at ``eta >= 0.46`` for ``n >= 7`` because the Theorem 12 threshold has
      become non-positive and the verifier trivially accepts empty lists.
    * Dashed ``correctness`` (s) — fraction with ``hypothesisCorrect = True``.
      Drops to 0 in the vacuous regime.
    * Dotted ``joint = accepted AND correct`` (^) — the operationally
      honest "protocol worked" event.  Collapses both the vacuous-accept
      spike and the mid-eta variance dip into a single signal.

    Two vertical breakdown markers:

    * ``eta_prover = (1 - theta/2) / 2`` — prover-side extraction floor.
      The honest prover's post-selected QFS mass at the target is
      ``(1-2*eta)**2``; below the extraction threshold ``theta**2/4`` the
      prover emits empty lists.  This is the earlier of the two events.
    * ``eta_max = (1 - eps/(2*sqrt(2))) / 2`` — verifier-side algebraic
      threshold inversion.  Above this ``a^2 - eps^2/8`` is negative and
      the acceptance check is vacuous.

    The ``eta > eta_max`` region is shaded red (``vacuous threshold``) so
    the ``accept`` spike at ``eta in {0.46, 0.48}`` is visually marked as a
    degenerate regime, not a protocol recovery.
    """
    ns = tables["ns"]
    etas = tables["etas"]
    acc = tables["acceptance_rate"]
    corr = tables["correctness_rate"]
    joint = tables["joint_rate"]
    epsilon = tables["epsilon"]
    theta = tables["theta"]

    # Pick representative n values: small, medium, large
    n_choices = []
    if 4 in ns:
        n_choices.append(4)
    if 8 in ns:
        n_choices.append(8)
    if 14 in ns:
        n_choices.append(14)
    elif ns[-1] >= 12:
        n_choices.append(ns[-1])

    eta_prover, eta_max = breakdown_points(epsilon, theta)

    colours = sns.color_palette("colorblind", len(n_choices))
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    eta_arr = np.array(etas)

    for idx, n in enumerate(n_choices):
        acc_vals = [acc.get((n, e), 0.0) for e in etas]
        corr_vals = [corr.get((n, e), 0.0) for e in etas]
        joint_vals = [joint.get((n, e), 0.0) for e in etas]

        ax.plot(eta_arr, acc_vals, "-o", color=colours[idx], markersize=4,
                linewidth=1.2, label=f"Acceptance ($n={n}$)")
        ax.plot(eta_arr, corr_vals, "--s", color=colours[idx], markersize=4,
                linewidth=1.2, alpha=0.75, label=f"Correctness ($n={n}$)")
        ax.plot(eta_arr, joint_vals, ":^", color=colours[idx], markersize=5,
                linewidth=1.6,
                label=f"Accept $\\wedge$ correct ($n={n}$)")

    x_max = max(etas) + 0.01
    # Shade the three distinct regimes
    if 0 < eta_prover < 0.5:
        ax.axvspan(0, eta_prover, alpha=0.06, color="green")
    if 0 < eta_prover < eta_max < 0.5:
        ax.axvspan(eta_prover, eta_max, alpha=0.08, color="orange")
    if eta_max < x_max:
        ax.axvspan(eta_max, x_max, alpha=0.10, color="red")

    # Vertical breakdown markers
    if 0 < eta_prover < 0.5:
        ax.axvline(
            eta_prover, ls="-.", color="#d95f02", alpha=0.8, linewidth=1.5,
            label=(
                rf"$\eta_{{\mathrm{{prover}}}} = (1-\vartheta/2)/2 = "
                rf"{eta_prover:.3f}$"
                "\n"
                r"(prover extraction floor $\theta^2/4 > (1-2\eta)^2$)"
            ),
        )
    if 0 < eta_max < 0.5:
        ax.axvline(
            eta_max, ls=":", color="#7570b3", alpha=0.8, linewidth=1.5,
            label=(
                rf"$\eta_{{\max}} = (1 - \varepsilon/(2\sqrt{{2}}))/2 = "
                rf"{eta_max:.3f}$"
                "\n"
                r"(vacuous threshold $(1-2\eta)^2 - \varepsilon^2/8 \leq 0$)"
            ),
        )

    # Label the three regimes with small italic text.  The middle regime
    # (prover signal lost) is narrow -- about 0.022 wide -- so its label is
    # drawn above the plot area with an arrow pointing into the band.
    ax.text((eta_prover) / 2, 0.03, "safe",
            ha="center", va="bottom",
            fontsize=9, color="green", alpha=0.7, fontstyle="italic")
    narrow_midpoint = (eta_prover + eta_max) / 2
    ax.annotate(
        "prover signal\nlost",
        xy=(narrow_midpoint, 0.48),
        xytext=(narrow_midpoint - 0.12, 0.78),
        ha="center", va="center",
        fontsize=8, color="#d95f02", alpha=0.9, fontstyle="italic",
        arrowprops=dict(arrowstyle="-", color="#d95f02", alpha=0.6, lw=0.8),
    )
    ax.text((eta_max + x_max) / 2, 0.45, "vacuous\nthreshold",
            ha="center", va="center",
            fontsize=9, color="#b22222", alpha=0.85, fontstyle="italic")

    ax.set_xlabel(r"Noise rate $\eta$")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(-0.01, x_max)
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.01, 0.5),
              ncol=1, frameon=True)
    ax.set_title(
        r"Acceptance, correctness, and joint success vs label-flip noise $\eta$"
    )
    fig.text(
        0.5,
        -0.03,
        BREAKDOWN_FOOTNOTE,
        ha="center",
        va="top",
        fontsize=7,
        style="italic",
        color="#333",
        wrap=True,
    )
    fig.tight_layout()
    save(fig, "acceptance_correctness_vs_eta")


# ---------------------------------------------------------------------------
# Artefact 3: Fourier weight attenuation
# ---------------------------------------------------------------------------

def plot_fourier_weight_attenuation(tables: dict) -> None:
    """Two-panel figure:

    * Top: filtered median accumulated weight vs ``eta``, overlaid on the
      Lemma 6 theory curve ``(1-2*eta)^2``.  The filter excludes trials
      where the prover returned an empty list and ``reject_list_too_large``
      trials — i.e. only trials where the verifier actually computed a
      ``Sigma xi_hat^2`` over a real list contribute to the median.  Without
      this filter the median collapses to 0 at ``eta >= 0.44`` because a
      majority of trials have ``accumulatedWeight = 0``, making Lemma 6
      appear to break down when in fact the lemma still holds exactly at
      the quantum state level — the collapse is a *downstream pipeline*
      artefact of the honest prover's extraction threshold.
    * Bottom: fraction of trials per cell where the prover returned an
      empty list (or the verifier rejected for list-too-large).  This is
      the honest prover-side breakdown signal: it starts rising sharply
      near ``eta_prover = (1-theta/2)/2`` when the extraction threshold
      ``theta**2/4`` exceeds the QFS mass ``(1-2*eta)**2``.
    """
    ns = tables["ns"]
    etas = tables["etas"]
    fmw = tables["filtered_median_weight"]
    elf = tables["empty_list_fraction"]
    epsilon = tables["epsilon"]
    theta = tables["theta"]

    eta_arr = np.array(etas)
    eta_prover, eta_max = breakdown_points(epsilon, theta)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.5, 6.8), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )

    # ---- Top: filtered median weight vs theory ----
    eta_fine = np.linspace(0, 0.5, 200)
    theory_weight = (1 - 2 * eta_fine) ** 2
    theory_threshold = (1 - 2 * eta_fine) ** 2 - epsilon**2 / 8
    ax_top.plot(eta_fine, theory_weight, "-", color="grey",
                alpha=0.7, linewidth=2,
                label=r"Lemma 6: $(1-2\eta)^2$")
    ax_top.plot(eta_fine, theory_threshold, "--", color="grey",
                alpha=0.6, linewidth=1.5,
                label=r"Thm 12 threshold: $(1-2\eta)^2 - \varepsilon^2/8$")
    ax_top.axhline(0, ls="-", color="black", alpha=0.3, linewidth=0.5)

    n_choices = [ns[0], ns[len(ns) // 2], ns[-1]]
    palette = sns.color_palette("colorblind", len(n_choices))
    markers = ["o", "s", "^"]

    for n, colour, marker in zip(n_choices, palette, markers):
        weights = np.array([fmw.get((n, e), float("nan")) for e in etas])
        mask = ~np.isnan(weights)
        ax_top.plot(
            eta_arr[mask], weights[mask],
            f"{marker}-",
            color=colour, markersize=5, linewidth=1.5, alpha=0.9,
            label=rf"Filtered median weight ($n={n}$, non-empty $L$)",
        )

    # Vertical breakdown markers
    if 0 < eta_prover < 0.5:
        ax_top.axvline(
            eta_prover, ls="-.", color="#d95f02", alpha=0.8, linewidth=1.3,
            label=(
                rf"$\eta_{{\mathrm{{prover}}}} = (1-\vartheta/2)/2 ="
                rf" {eta_prover:.3f}$"
            ),
        )
    if 0 < eta_max < 0.5:
        ax_top.axvline(
            eta_max, ls=":", color="#7570b3", alpha=0.8, linewidth=1.3,
            label=(
                rf"$\eta_{{\max}} = (1-\varepsilon/(2\sqrt{{2}}))/2 ="
                rf" {eta_max:.3f}$"
            ),
        )

    ax_top.set_ylabel(r"Fourier weight $\Sigma_{s \in L}\, \hat\xi(s)^2$")
    ax_top.set_ylim(-0.05, 1.1)
    ax_top.legend(fontsize=6, loc="upper right", ncol=1, framealpha=0.9)
    ax_top.set_title(
        r"Lemma 6 $(1-2\eta)^2$ attenuation (filtered to non-empty prover lists)"
    )

    # ---- Bottom: empty-list fraction vs eta ----
    for n, colour, marker in zip(n_choices, palette, markers):
        empty_vals = np.array([elf.get((n, e), 0.0) for e in etas])
        ax_bot.plot(
            eta_arr, empty_vals,
            f"{marker}-",
            color=colour, markersize=4, linewidth=1.2, alpha=0.9,
            label=rf"$n={n}$",
        )
    if 0 < eta_prover < 0.5:
        ax_bot.axvline(eta_prover, ls="-.", color="#d95f02",
                       alpha=0.8, linewidth=1.3)
    if 0 < eta_max < 0.5:
        ax_bot.axvline(eta_max, ls=":", color="#7570b3",
                       alpha=0.8, linewidth=1.3)

    ax_bot.set_xlabel(r"Noise rate $\eta$")
    ax_bot.set_ylabel(
        "Pr[prover returned\nempty list]",
        fontsize=9,
    )
    ax_bot.set_ylim(-0.05, 1.08)
    ax_bot.set_xlim(-0.01, max(etas) + 0.01)
    ax_bot.legend(fontsize=7, loc="upper left", ncol=len(n_choices))
    ax_bot.set_title(
        r"Prover-side breakdown: empty-list fraction vs $\eta$",
        fontsize=10,
    )

    fig.text(
        0.5,
        -0.02,
        BREAKDOWN_FOOTNOTE,
        ha="center",
        va="top",
        fontsize=7,
        style="italic",
        color="#333",
        wrap=True,
    )
    fig.tight_layout()
    save(fig, "fourier_weight_attenuation")


# ---------------------------------------------------------------------------
# Artefact 4: Breakdown points table (CSV)
# ---------------------------------------------------------------------------

def write_breakdown_table(tables: dict) -> None:
    """CSV: per-n empirical eta where acceptance < 50%, theoretical eta threshold."""
    ns = tables["ns"]
    etas = tables["etas"]
    acc = tables["acceptance_rate"]
    epsilon = tables["epsilon"]

    # Theoretical: (1-2*eta)^2 = eps^2/8 => eta = (1 - eps/(2*sqrt(2))) / 2
    eta_theory = (1 - epsilon / (2 * math.sqrt(2))) / 2

    path = OUT_DIR / "breakdown_points.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n", "empirical_eta_50pct", "theoretical_eta_threshold",
            "acceptance_at_max_eta", "match_within_0.05",
        ])
        for n in ns:
            # Find empirical breakdown: first eta where acceptance < 0.5
            empirical_eta = None
            for eta in etas:
                rate = acc.get((n, eta), 1.0)
                if rate < 0.5:
                    empirical_eta = eta
                    break

            acc_at_max = acc.get((n, etas[-1]), 0.0)
            match = ""
            if empirical_eta is not None:
                match = "yes" if abs(empirical_eta - eta_theory) <= 0.05 else "no"
            else:
                match = "no_breakdown"

            writer.writerow([
                n,
                f"{empirical_eta:.2f}" if empirical_eta is not None else "none",
                f"{eta_theory:.4f}",
                f"{acc_at_max:.2f}",
                match,
            ])
    print(f"  {path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(tables: dict) -> None:
    """Print a summary of key findings to stdout."""
    ns = tables["ns"]
    etas = tables["etas"]
    acc = tables["acceptance_rate"]
    corr = tables["correctness_rate"]
    mw = tables["median_weight"]
    epsilon = tables["epsilon"]

    eta_theory = (1 - epsilon / (2 * math.sqrt(2))) / 2

    print("\n" + "=" * 70)
    print("NOISE SWEEP ANALYSIS REPORT")
    print("=" * 70)

    print(f"\nExperiment: n in {ns[0]}..{ns[-1]}, "
          f"eta in {{{', '.join(f'{e:.2f}' for e in etas)}}}")
    print(f"Epsilon = {epsilon}, theoretical eta_max = {eta_theory:.4f}")
    print(f"  (from (1-2*eta)^2 = eps^2/8)")

    # 1. Breakdown analysis
    print("\n--- Breakdown Analysis ---")
    for n in ns:
        breakdown_eta = None
        for eta in etas:
            if acc.get((n, eta), 1.0) < 0.5:
                breakdown_eta = eta
                break
        if breakdown_eta is not None:
            print(f"  n={n:2d}: acceptance < 50% at eta={breakdown_eta:.2f}"
                  f"  (theory: {eta_theory:.3f},"
                  f" diff={abs(breakdown_eta - eta_theory):.3f})")
        else:
            print(f"  n={n:2d}: acceptance never drops below 50%"
                  f" (min acc={min(acc.get((n, e), 1.0) for e in etas)*100:.1f}%)")

    # 2. N-dependence
    print("\n--- N-dependence at fixed eta ---")
    for eta in [0.2, 0.3, 0.4]:
        rates = [acc.get((n, eta), 0.0) for n in ns]
        print(f"  eta={eta:.1f}: acceptance range = "
              f"{min(rates)*100:.0f}%--{max(rates)*100:.0f}%"
              f" (spread={abs(max(rates)-min(rates))*100:.1f}pp)")

    # 3. Degradation character
    print("\n--- Degradation Character ---")
    n_mid = ns[len(ns) // 2]
    prev_rate = 1.0
    for eta in etas:
        rate = acc.get((n_mid, eta), 0.0)
        drop = prev_rate - rate
        marker = " <-- large drop" if drop > 0.2 else ""
        print(f"  n={n_mid}, eta={eta:.2f}: acceptance={rate*100:5.1f}%"
              f"  (delta={-drop*100:+.1f}pp){marker}")
        prev_rate = rate

    # 4. Weight attenuation
    print("\n--- Weight Attenuation vs Theory ---")
    for n in [ns[0], ns[len(ns)//2], ns[-1]]:
        print(f"  n={n}:")
        for eta in etas:
            observed = mw.get((n, eta), 0.0)
            theory = (1 - 2 * eta) ** 2
            rel_err = abs(observed - theory) / theory if theory > 0.01 else float("inf")
            print(f"    eta={eta:.2f}: weight={observed:.4f},"
                  f" theory={(1-2*eta)**2:.4f},"
                  f" rel_err={rel_err*100:.1f}%")

    # 5. Correctness
    print("\n--- Correctness Under Noise ---")
    for n in [ns[0], ns[len(ns)//2], ns[-1]]:
        for eta in [0.0, 0.2, 0.4]:
            c = corr.get((n, eta), 0.0)
            a = acc.get((n, eta), 0.0)
            print(f"  n={n:2d}, eta={eta:.1f}: correctness={c*100:.0f}%,"
                  f" acceptance={a*100:.0f}%")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_style()

    print("Loading noise sweep data...")
    data = load_data()
    trials = data["trials"]
    params = data["parameters"]
    epsilon = params["epsilon"]
    print(f"  {len(trials)} trials, n in {params['nRange'][0]}..{params['nRange'][-1]}, "
          f"eta in {params['noiseRates']}, epsilon={epsilon}")

    tables = build_tables(trials, epsilon)

    print("\nGenerating artefacts:")
    plot_noise_heatmap(tables)
    plot_acceptance_correctness_vs_eta(tables)
    plot_fourier_weight_attenuation(tables)
    write_breakdown_table(tables)

    print_report(tables)
    print("\nDone.")


if __name__ == "__main__":
    main()
