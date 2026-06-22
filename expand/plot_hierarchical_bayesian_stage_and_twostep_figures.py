import os

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


matplotlib.use("Agg")

BASE = "base_dir"
EXP = os.path.join(BASE, "expand")

SUMMARY_CSV = os.path.join(EXP, "hierarchical_bayesian_stage_twostep_summary.csv")

OUT_STAGE_PNG = os.path.join(EXP, "figure_hierarchical_bayesian_p3b_frn_stage_comparison.png")
OUT_STAGE_PDF = os.path.join(EXP, "figure_hierarchical_bayesian_p3b_frn_stage_comparison.pdf")
OUT_TWOSTEP_PNG = os.path.join(EXP, "figure_hierarchical_bayesian_p3b_twostep_by_set.png")
OUT_TWOSTEP_PDF = os.path.join(EXP, "figure_hierarchical_bayesian_p3b_twostep_by_set.pdf")

COLORS = {
    "all_sets": "#4b5563",
    "early_1_2": "#d95f02",
    "late_7_8": "#1b9e77",
}
SET_LABELS = {
    "all_sets": "Overall sets 1-8",
    "early_1_2": "Early sets 1-2",
    "late_7_8": "Late sets 7-8",
}
FEATURE_LABELS = {
    "feedback_locked_P3b": "P3b",
    "feedback_locked_FRN": "FRN",
}
STAGE_LABELS = {
    "transition_first_correct_minus_search_error": "FC - search\nerror",
    "transition_first_correct_minus_transition_pre_core_correct": "FC - transition\ncorrect",
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
    ax.plot([lo, hi], [y, y], color=color, linewidth=2.25, solid_capstyle="round")
    ax.scatter([x], [y], color=color, s=42, zorder=3, label=label, edgecolor="white", linewidth=0.7)


def plot_stage_comparison(summary: pd.DataFrame) -> None:
    d = summary[
        (summary["analysis_family"] == "p300_frn_stage_changes")
        & (summary["feature"].isin(["feedback_locked_P3b", "feedback_locked_FRN"]))
    ].copy()

    contrasts = list(STAGE_LABELS.keys())
    set_order = ["all_sets", "early_1_2", "late_7_8"]
    offsets = {"all_sets": 0.18, "early_1_2": 0.0, "late_7_8": -0.18}
    y_base = np.arange(len(contrasts))[::-1]

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.2), sharex=True, constrained_layout=False)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.85, bottom=0.22, hspace=0.34)

    for ax, feature in zip(axes, ["feedback_locked_P3b", "feedback_locked_FRN"]):
        sub = d[d["feature"] == feature].copy()
        for set_name in set_order:
            ss = sub[sub["set_bin"] == set_name].set_index("contrast")
            first = True
            for i, contrast in enumerate(contrasts):
                if contrast not in ss.index:
                    continue
                row = ss.loc[contrast]
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
        ax.set_title(FEATURE_LABELS[feature], fontsize=12, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.set_xlim(-5.0, 5.0)
        ax.tick_params(axis="both", labelsize=9)

    axes[-1].set_xlabel("Posterior mean difference (uV)", labelpad=6)
    fig.text(0.045, 0.53, "Stage contrast", rotation=90, ha="center", va="center", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.8, bbox_to_anchor=(0.5, 0.045))
    fig.suptitle("Hierarchical Bayesian stage comparisons for P3b and FRN", fontsize=14, y=0.965)
    fig.text(
        0.5,
        0.135,
        "Points show posterior means; intervals show 95% credible intervals from block-pair hierarchical models.",
        ha="center",
        va="center",
        fontsize=8.4,
    )

    fig.savefig(OUT_STAGE_PNG, dpi=340, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT_STAGE_PDF, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"saved: {OUT_STAGE_PNG}")
    print(f"saved: {OUT_STAGE_PDF}")


def plot_twostep_by_set(summary: pd.DataFrame) -> None:
    d = summary[
        (summary["analysis_family"] == "two_step_transition_confirmation")
        & (summary["feature"] == "feedback_locked_P3b")
    ].copy()

    contrasts = list(TWOSTEP_LABELS.keys())
    panels = [
        ("A. Overall", "all_sets"),
        ("B. Early sets", "early_1_2"),
        ("C. Late sets", "late_7_8"),
    ]
    x = np.arange(len(contrasts))

    fig, axes = plt.subplots(3, 1, figsize=(7.8, 8.4), sharex=True, sharey=True, constrained_layout=False)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.18, hspace=0.34)

    for ax, (title, set_name) in zip(axes, panels):
        ss = d[d["set_bin"] == set_name].set_index("contrast")
        means, lows, highs = [], [], []
        for contrast in contrasts:
            row = ss.loc[contrast]
            means.append(float(row["posterior_mean"]))
            lows.append(float(row["ci95_low"]))
            highs.append(float(row["ci95_high"]))
        means = np.array(means)
        lows = np.array(lows)
        highs = np.array(highs)
        yerr = np.vstack([means - lows, highs - means])

        ax.errorbar(
            x,
            means,
            yerr=yerr,
            fmt="o-",
            color=COLORS[set_name],
            linewidth=2.25,
            capsize=3.8,
            markersize=6.2,
        )
        ax.axhline(0, color="#222222", linewidth=1.0, linestyle="--", alpha=0.75)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("P3b difference (uV)")
        ax.set_ylim(-4.2, 5.0)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.tick_params(axis="both", labelsize=9)
        ax.text(
            0.99,
            0.88,
            SET_LABELS[set_name],
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
            color=COLORS[set_name],
            fontweight="bold",
        )

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([TWOSTEP_LABELS[c] for c in contrasts], fontsize=9)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    fig.suptitle("Hierarchical Bayesian P3b two-step transition confirmation", fontsize=14, y=0.965)
    fig.text(
        0.5,
        0.085,
        "Points show posterior means; intervals show 95% credible intervals. Overall, early, and late panels were estimated separately.",
        ha="center",
        va="center",
        fontsize=8.4,
    )

    fig.savefig(OUT_TWOSTEP_PNG, dpi=340, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT_TWOSTEP_PDF, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"saved: {OUT_TWOSTEP_PNG}")
    print(f"saved: {OUT_TWOSTEP_PDF}")


def main() -> None:
    set_style()
    summary = pd.read_csv(SUMMARY_CSV)
    plot_stage_comparison(summary)
    plot_twostep_by_set(summary)


if __name__ == "__main__":
    main()
