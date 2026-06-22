import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel


BASE = "base_dir"
COMP_TAG = "1s_comp"

CUE_TRIAL = os.path.join(BASE, "trials_trialwise", COMP_TAG, "eeg_tables", "eeg_trial_cue_locked.csv")
FB_TRIAL = os.path.join(BASE, "trials_trialwise", COMP_TAG, "eeg_tables", "eeg_trial_long.csv")

OUT_DIR = os.path.join(BASE, "trials_trialwise", COMP_TAG, "eeg_paper_results", "erp_component_reanalysis")
OUT_STATS = os.path.join(OUT_DIR, "erp_component_paired_tests.csv")
OUT_SUBJ = os.path.join(OUT_DIR, "erp_component_subject_means_long.csv")
OUT_FIG = os.path.join(OUT_DIR, "erp_component_summary.png")

SKIP_SUBJ = {"reknow011", "reknow020", "reknow023"}


def _subject_means(df: pd.DataFrame, group_col: str, value_col: str, keep_levels: List[str]) -> pd.DataFrame:
    d = df.copy()
    d = d[~d["subj"].isin(SKIP_SUBJ)]
    d[group_col] = d[group_col].astype(str).str.strip()
    d = d[d[group_col].isin(keep_levels)]
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=["subj", group_col, value_col])
    out = d.groupby(["subj", group_col], as_index=False)[value_col].mean()
    return out


def _paired_test(subj_means: pd.DataFrame, group_col: str, value_col: str, a: str, b: str) -> Dict[str, float]:
    aa = subj_means[subj_means[group_col] == a][["subj", value_col]].rename(columns={value_col: "a"})
    bb = subj_means[subj_means[group_col] == b][["subj", value_col]].rename(columns={value_col: "b"})
    m = aa.merge(bb, on="subj", how="inner").dropna()
    if len(m) < 6:
        return {
            "n_pairs": int(len(m)),
            "mean_a": np.nan,
            "mean_b": np.nan,
            "mean_diff_a_minus_b": np.nan,
            "t": np.nan,
            "p": np.nan,
        }
    t, p = ttest_rel(m["a"], m["b"], nan_policy="omit")
    return {
        "n_pairs": int(len(m)),
        "mean_a": float(m["a"].mean()),
        "mean_b": float(m["b"].mean()),
        "mean_diff_a_minus_b": float((m["a"] - m["b"]).mean()),
        "t": float(t),
        "p": float(p),
    }


def _plot_grouped_subject_bars(ax, subj_means: pd.DataFrame, group_col: str, value_col: str, order: List[str], title: str, ylabel: str):
    x = np.arange(len(order))
    means = []
    sems = []
    for lev in order:
        vals = subj_means[subj_means[group_col] == lev][value_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        means.append(np.nanmean(vals) if len(vals) else np.nan)
        sems.append(np.nanstd(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan)

    ax.bar(x, means, yerr=sems, color=["#A8C9E8", "#F5C2CE", "#BFE3C0"][: len(order)], edgecolor="#666666", linewidth=0.8, capsize=4)

    rng = np.random.default_rng(20260404)
    for i, lev in enumerate(order):
        vals = subj_means[subj_means[group_col] == lev][value_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        jit = rng.uniform(-0.06, 0.06, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jit, vals, s=20, color="black", alpha=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=12)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cue = pd.read_csv(CUE_TRIAL)
    fb = pd.read_csv(FB_TRIAL)

    stats_rows: List[Dict[str, float]] = []
    subj_rows: List[pd.DataFrame] = []

    # 1) Cue-locked switch-related positivity proxy (P3a_cue)
    cue_order = ["within_level", "cross_level"]
    cue_p3a = _subject_means(cue, "shift_type", "P3a_cue", cue_order)
    cue_p3a = cue_p3a.rename(columns={"shift_type": "condition", "P3a_cue": "value"})
    cue_p3a["component"] = "P3a_cue"
    subj_rows.append(cue_p3a[["subj", "component", "condition", "value"]])
    s = _paired_test(cue_p3a, "condition", "value", "cross_level", "within_level")
    s.update(
        {
            "analysis": "cue_shift_type",
            "component": "P3a_cue",
            "contrast_a": "cross_level",
            "contrast_b": "within_level",
        }
    )
    stats_rows.append(s)

    # 2) Feedback-locked ERP components across HMM phases (search vs acquired)
    phase_order = ["search", "acquired"]
    fb_components = ["feedback_locked_FRN", "feedback_locked_P3a", "feedback_locked_P3b"]

    for comp in fb_components:
        d = _subject_means(fb, "phase", comp, phase_order)
        d = d.rename(columns={"phase": "condition", comp: "value"})
        d["component"] = comp
        subj_rows.append(d[["subj", "component", "condition", "value"]])

        for a, b in [("search", "acquired")]:
            s = _paired_test(d, "condition", "value", a, b)
            s.update(
                {
                    "analysis": "feedback_phase",
                    "component": comp,
                    "contrast_a": a,
                    "contrast_b": b,
                }
            )
            stats_rows.append(s)

    subj_long = pd.concat(subj_rows, ignore_index=True)
    stats_df = pd.DataFrame(stats_rows)
    stats_df = stats_df[
        ["analysis", "component", "contrast_a", "contrast_b", "n_pairs", "mean_a", "mean_b", "mean_diff_a_minus_b", "t", "p"]
    ]
    stats_df.to_csv(OUT_STATS, index=False)
    subj_long.to_csv(OUT_SUBJ, index=False)

    # Plot summary figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    _plot_grouped_subject_bars(
        axes[0, 0],
        cue_p3a,
        "condition",
        "value",
        cue_order,
        "Cue P3a by Shift Type",
        "P3a (uV)",
    )
    axes[0, 0].set_xticklabels(["within", "cross"])

    for ax, comp, ttl in zip(
        [axes[0, 1], axes[1, 0], axes[1, 1]],
        fb_components,
        ["Feedback FRN by Phase", "Feedback P3a by Phase", "Feedback P3b by Phase"],
    ):
        d = subj_long[subj_long["component"] == comp].copy()
        _plot_grouped_subject_bars(ax, d, "condition", "value", phase_order, ttl, f"{comp} (uV)")

    fig.savefig(OUT_FIG, dpi=320, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print("saved ->", OUT_DIR)
    print("stats:", OUT_STATS)
    print("subject means:", OUT_SUBJ)
    print("figure:", OUT_FIG)


if __name__ == "__main__":
    main()

