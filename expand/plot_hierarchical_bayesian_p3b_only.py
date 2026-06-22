import os

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


matplotlib.use("Agg")

BASE = "base_dir"
EXP = os.path.join(BASE, "expand")

SUMMARY_CSV = os.path.join(EXP, "hierarchical_bayesian_stage_twostep_summary.csv")
OUT_PNG = os.path.join(EXP, "figure_hierarchical_bayesian_p3b_transition_only.png")
OUT_PDF = os.path.join(EXP, "figure_hierarchical_bayesian_p3b_transition_only.pdf")

COLORS = {
    "early_1_2": "#d95f02",
    "late_7_8": "#1b9e77",
    "all_sets": "#4b5563",
}
SET_LABELS = {
    "early_1_2": "Early sets 1-2",
    "late_7_8": "Late sets 7-8",
    "all_sets": "All sets 1-8",
}

STAGE_LABELS = {
    "transition_first_correct_minus_search_error": "FC - search\nerror",
    "transition_first_correct_minus_transition_pre_core_correct": "FC - transition\npre-core",
    "transition_first_correct_minus_acquired_correct": "FC - acquired\ncorrect",
}

TWOSTEP_LABELS = {
    "step1_transition_fc_minus_search": "Step 1\nFC - search",
    "step2a_transition_precore_minus_fc": "Step 2a\nPre-core - FC",
    "step2b_acquired_core_minus_transition_precore": "Step 2b\nAcquired - pre-core",
}


def set_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.linewidth"] = 0.9
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8


def draw_interval(ax, x, y, lo, hi, color, label=None) -> None:
    ax.plot([lo, hi], [y, y], color=color, linewidth=2.4, solid_capstyle="round")
    ax.scatter([x], [y], color=color, s=48, zorder=3, label=label, edgecolor="white", linewidth=0.75)


def plot_stage(ax, d: pd.DataFrame) -> None:
    contrasts = list(STAGE_LABELS.keys())
    set_order = ["early_1_2", "late_7_8", "all_sets"]
    offsets = {"early_1_2": 0.18, "late_7_8": 0.0, "all_sets": -0.18}
    y_base = np.arange(len(contrasts))[::-1]

    sub = d[
        (d["analysis_family"] == "p300_frn_stage_changes")
        & (d["feature"] == "feedback_locked_P3b")
    ].copy()

    for set_name in set_order:
        ss = sub[sub["set_bin"] == set_name].set_index("contrast")
        first = True
        for i, cname in enumerate(contrasts):
            if cname not in ss.index:
                continue
            row = ss.loc[cname]
            draw_interval(
                ax,
                float(row["posterior_mean"]),
                y_base[i] + offsets[set_name],
                float(row["ci95_low"]),
                float(row["ci95_high"]),
                COLORS[set_name],
                SET_LABELS[set_name] if first else None,
            )
            first = False

    ax.axvline(0, color="#222222", linewidth=1.0, linestyle="--", alpha=0.75)
    ax.set_yticks(y_base)
    ax.set_yticklabels([STAGE_LABELS[c] for c in contrasts], rotation=8, va="center")
    ax.set_xlim(-5.0, 5.0)
    ax.set_title("A. P3b stage contrasts", loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel("Posterior mean difference (uV)")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.tick_params(axis="both", labelsize=9)


def plot_twostep(ax, d: pd.DataFrame) -> None:
    contrasts = list(TWOSTEP_LABELS.keys())
    set_order = ["early_1_2", "late_7_8", "all_sets"]
    x = np.arange(len(contrasts))

    sub = d[
        (d["analysis_family"] == "two_step_transition_confirmation")
        & (d["feature"] == "feedback_locked_P3b")
    ].copy()

    for set_name in set_order:
        ss = sub[sub["set_bin"] == set_name].set_index("contrast")
        means, lows, highs = [], [], []
        for cname in contrasts:
            row = ss.loc[cname]
            means.append(float(row["posterior_mean"]))
            lows.append(float(row["ci95_low"]))
            highs.append(float(row["ci95_high"]))
        means = np.array(means)
        yerr = np.vstack([means - np.array(lows), np.array(highs) - means])
        ax.errorbar(
            x,
            means,
            yerr=yerr,
            fmt="o-",
            color=COLORS[set_name],
            linewidth=2.2,
            capsize=3.8,
            markersize=6.0,
            label=SET_LABELS[set_name],
        )

    ax.axhline(0, color="#222222", linewidth=1.0, linestyle="--", alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels([TWOSTEP_LABELS[c] for c in contrasts], fontsize=9)
    ax.set_ylim(-4.2, 5.0)
    ax.set_ylabel("Posterior mean difference (uV)")
    ax.set_title("B. P3b two-step transition confirmation", loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.tick_params(axis="y", labelsize=9)


def main() -> None:
    set_style()
    d = pd.read_csv(SUMMARY_CSV)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), constrained_layout=False)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.86, bottom=0.18, hspace=0.48)

    plot_stage(axes[0], d)
    plot_twostep(axes[1], d)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.8, bbox_to_anchor=(0.5, 0.045))

    fig.suptitle("Hierarchical Bayesian P3b transition effects", fontsize=14, y=0.965)
    fig.text(
        0.5,
        0.105,
        "Points show posterior means; intervals show 95% credible intervals. Estimates come from block-pair hierarchical models with subject random effects and trial-count-weighted observation noise.",
        ha="center",
        va="center",
        fontsize=8.2,
    )

    fig.savefig(OUT_PNG, dpi=340, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.06)
    print(f"saved: {OUT_PNG}")
    print(f"saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
