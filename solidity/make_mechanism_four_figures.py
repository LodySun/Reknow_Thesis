import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
from scipy.stats import norm
import statsmodels.formula.api as smf
from patsy import dmatrix
from figure_style_and_metrics import p_to_sig, set_helvetica_font, update_metrics_file


BASE = "base_dir"
OUT_DIR = BASE
set_helvetica_font()

PATH_BLOCK = os.path.join(BASE, "hmm_unified_block_metrics.csv")
PATH_TRIAL = os.path.join(BASE, "hmm_unified_trial_posteriors.csv")
PATH_STAGE = os.path.join(BASE, "eeg_acq_sequence_stage_means_long.csv")
PATH_PAIR = os.path.join(BASE, "eeg_acq_sequence_stage_pair_tests.csv")
PATH_SPEAR = os.path.join(BASE, "eeg_transition_firstcorrect_predicts_hmm_metrics_spearman.csv")
PATH_MIXED = os.path.join(BASE, "eeg_transition_firstcorrect_predicts_hmm_metrics_mixedlm.csv")
PATH_SENS = os.path.join(BASE, "figure23_leave2out_sensitivity_mixedlm.csv")
PATH_COUNT = os.path.join(BASE, "figure23_count_model_alternatives.csv")


def _bootstrap_mean_ci(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 2026) -> Tuple[float, float, float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        s = rng.choice(x, size=len(x), replace=True)
        boots.append(np.nanmean(s))
    boots = np.array(boots, dtype=float)
    lo = float(np.nanpercentile(boots, 100 * alpha / 2))
    hi = float(np.nanpercentile(boots, 100 * (1 - alpha / 2)))
    return float(np.nanmean(x)), lo, hi


def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n <= 0:
        return np.nan, np.nan, np.nan
    phat = k / n
    den = 1 + z**2 / n
    cen = (phat + z**2 / (2 * n)) / den
    half = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / den
    return float(phat), float(cen - half), float(cen + half)


def _add_panel_label(ax, label: str):
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")


def _prep():
    b = pd.read_csv(PATH_BLOCK)
    t = pd.read_csv(PATH_TRIAL)
    s = pd.read_csv(PATH_STAGE)
    p = pd.read_csv(PATH_PAIR)
    sp = pd.read_csv(PATH_SPEAR)
    mx = pd.read_csv(PATH_MIXED)

    for c in ["acquisition_lag_core", "transition_width_20_to_80", "set_index", "acquisition_trial_core", "acquisition_trial_viterbi", "first_correct_trial"]:
        if c in b.columns:
            b[c] = pd.to_numeric(b[c], errors="coerce")
    for c in ["trial_index_1based", "p_acq", "rel_trial_fc", "set_index"]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")
    s["value_mean"] = pd.to_numeric(s["value_mean"], errors="coerce")
    return b, t, s, p, sp, mx


def make_figure1_behavior(b: pd.DataFrame):
    fig_title = "Stabilization Typically Follows First Correct Rapidly, with Broader Early Confirmation Windows"
    fig_caption = (
        "Panel A shows acquisition lag distribution. Panel B shows transition width by set with full-range inset. "
        "Panel C shows setwise probability of non-zero confirmation window."
    )
    fig = plt.figure(figsize=(12.4, 8.1), constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.05], hspace=0.36, wspace=0.24)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    # Panel A: lag distribution
    lag = b["acquisition_lag_core"].dropna().to_numpy(dtype=float)
    # Integer-aligned bins; main x-range focuses on the informative bulk.
    bins = np.arange(-0.5, 10.5 + 1e-9, 1.0)
    lag_main = lag[(lag >= 0) & (lag <= 10)]
    axA.hist(lag_main, bins=bins, color="#9ecae1", edgecolor="white", alpha=0.9, density=True)
    if len(lag) >= 20:
        xs = np.linspace(0, 10, 300)
        try:
            kde = gaussian_kde(lag_main if len(lag_main) >= 8 else lag)
            axA.plot(xs, kde(xs), color="#08519c", linewidth=2.0)
        except Exception:
            pass
    prop_03 = float(((lag >= 0) & (lag <= 3)).mean()) if len(lag) else np.nan
    n_tail = int(np.sum(lag > 10))
    txt = f"mean={np.nanmean(lag):.2f}\nmedian={np.nanmedian(lag):.1f}\n0-3 trials={prop_03*100:.1f}%\n>10 trials={n_tail}"
    axA.text(0.98, 0.95, txt, transform=axA.transAxes, ha="right", va="top", fontsize=10, bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.8"))
    axA.set_title("Acquisition Lag Distribution")
    axA.set_xlabel("acquisition_lag_core (trials)")
    axA.set_ylabel("Density")
    axA.set_xlim(-0.5, 10.5)
    _add_panel_label(axA, "A")

    # Panel B: width by set (jitter + mean/bootCI)
    d = b[["set_index", "transition_width_20_to_80"]].dropna().copy()
    rng = np.random.default_rng(42)
    y_main_max = 6.0
    for si in sorted(d["set_index"].dropna().unique()):
        y = d.loc[d["set_index"] == si, "transition_width_20_to_80"].to_numpy(dtype=float)
        y_main = y[y <= y_main_max]
        x = np.full(len(y_main), si, dtype=float) + rng.uniform(-0.12, 0.12, size=len(y_main))
        axB.scatter(x, y_main, s=9, color="#bdbdbd", alpha=0.30, linewidths=0)
        m, lo, hi = _bootstrap_mean_ci(y)
        axB.errorbar([si], [m], yerr=[[m - lo], [hi - m]], fmt="o", color="#d62728", capsize=3, markersize=5)
        axB.text(si, min(m + 0.10, y_main_max - 0.08), f"{m:.2f}", ha="center", va="bottom", fontsize=8, color="#8c2d04")
    axB.set_xlim(0.5, 8.5)
    axB.set_xticks(np.arange(1, 9))
    axB.set_ylim(-0.1, y_main_max)
    axB.set_title("Transition Width by Set (main range)")
    axB.set_xlabel("set_index")
    axB.set_ylabel("transition_width_20_to_80")
    axB.grid(axis="y", linestyle="--", alpha=0.35)
    axB.text(0.03, 0.95, "red points = mean width", transform=axB.transAxes, ha="left", va="top", fontsize=8, color="#8c2d04")
    # Full-range inset for rare extreme blocks.
    axBins = axB.inset_axes([0.58, 0.50, 0.38, 0.34])
    for si in sorted(d["set_index"].dropna().unique()):
        y = d.loc[d["set_index"] == si, "transition_width_20_to_80"].to_numpy(dtype=float)
        x = np.full(len(y), si, dtype=float) + rng.uniform(-0.10, 0.10, size=len(y))
        axBins.scatter(x, y, s=5, color="#bdbdbd", alpha=0.22, linewidths=0)
    axBins.set_xlim(0.5, 8.5)
    axBins.set_ylim(-0.1, max(8.0, float(np.nanpercentile(d["transition_width_20_to_80"], 99.9) * 1.03)))
    axBins.set_xticks([])
    axBins.set_yticks([])
    axBins.set_title("full range", fontsize=7)
    _add_panel_label(axB, "B")

    # Panel C: proportion width>0 by set
    rows = []
    for si in range(1, 9):
        dd = d[d["set_index"] == si]
        n = len(dd)
        k = int((dd["transition_width_20_to_80"] > 0).sum())
        p, lo, hi = _wilson_ci(k, n)
        rows.append((si, p, lo, hi))
    rr = pd.DataFrame(rows, columns=["set_index", "prop", "lo", "hi"])
    axC.errorbar(rr["set_index"], rr["prop"], yerr=[rr["prop"] - rr["lo"], rr["hi"] - rr["prop"]], fmt="o-", color="#2ca25f", capsize=3, linewidth=2, markersize=5)
    axC.set_xlim(0.5, 8.5)
    axC.set_xticks(np.arange(1, 9))
    axC.set_ylim(0, max(0.23, float(np.nanmax(rr["hi"]) + 0.03)))
    axC.set_title("Non-zero Confirmation Windows Are More Common in Early Sets")
    axC.set_xlabel("set_index")
    axC.set_ylabel("Pr(width > 0)")
    axC.grid(axis="y", linestyle="--", alpha=0.35)
    _add_panel_label(axC, "C")
    fig.suptitle(fig_title, fontsize=15, y=0.98)
    out = os.path.join(OUT_DIR, "figure20_behavior_lag_width_summary.png")
    fig.savefig(out, dpi=320, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    # Supplementary robustness panel (former D): core vs viterbi concordance.
    dd = b[["acquisition_trial_core", "acquisition_trial_viterbi"]].dropna().copy()
    x = dd["acquisition_trial_viterbi"].to_numpy(dtype=float)
    y = dd["acquisition_trial_core"].to_numpy(dtype=float)
    r = pd.Series(x).corr(pd.Series(y), method="spearman")
    fig_sup, ax_sup = plt.subplots(1, 1, figsize=(5.4, 5.0), constrained_layout=False)
    ax_sup.scatter(x, y, s=10, alpha=0.22, color="#756bb1")
    mn = min(np.nanmin(x), np.nanmin(y))
    mx = max(np.nanmax(x), np.nanmax(y))
    ax_sup.plot([mn, mx], [mn, mx], linestyle="--", color="black", linewidth=1)
    ax_sup.text(0.97, 0.04, f"Spearman rho={r:.3f}", transform=ax_sup.transAxes, ha="right", va="bottom", fontsize=10)
    ax_sup.set_title("Core vs Viterbi Acquisition Concordance")
    ax_sup.set_xlabel("acquisition_trial_viterbi")
    ax_sup.set_ylabel("acquisition_trial_core")
    out_sup = os.path.join(OUT_DIR, "figure20s_core_vs_viterbi_concordance.png")
    fig_sup.savefig(out_sup, dpi=320, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig_sup)
    update_metrics_file(
        figure_id="figure20",
        figure_title=fig_title,
        figure_caption=fig_caption,
        rows=[
            {
                "panel": "supplement",
                "metric_name": "core_vs_viterbi_spearman_rho",
                "p_value": np.nan,
                "significance": "NA",
                "effect_or_stat": f"rho={r:.3f}",
                "notes": "No p-value displayed in figure.",
            }
        ],
    )
    return out


def make_figure2_time_series(b: pd.DataFrame, t: pd.DataFrame):
    fig_title = "Acquired-state Probability Rises Sharply After First Correct"
    fig_caption = (
        "Panel A shows early vs late aligned trajectories around the transition first-correct trial. "
        "Panel B shows four exemplar block-level p_acq trajectories."
    )
    fig = plt.figure(figsize=(13.2, 7.8), constrained_layout=False)
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.2, 1], hspace=0.38, wspace=0.28)
    axA = fig.add_subplot(gs[0, :])
    axsB = [fig.add_subplot(gs[1, i]) for i in range(4)]

    # Panel A: early extreme vs late extreme around the transition first-correct event
    d = t.dropna(subset=["rel_trial_fc", "p_acq", "set_index"]).copy()
    d = d[(d["rel_trial_fc"] >= -3) & (d["rel_trial_fc"] <= 4)]
    d["group"] = np.where(d["set_index"].isin([1, 2]), "set1_2", np.where(d["set_index"].isin([7, 8]), "set7_8", "other"))
    d = d[d["group"].isin(["set1_2", "set7_8"])]

    for grp, col in [("set1_2", "#d95f0e"), ("set7_8", "#1b9e77")]:
        g = d[d["group"] == grp]
        if g.empty:
            continue
        # average at block level first, then grand mean/CI
        blk = g.groupby(["subj", "block_id", "rel_trial_fc"], as_index=False)["p_acq"].mean()
        pivot = blk.pivot_table(index=["subj", "block_id"], columns="rel_trial_fc", values="p_acq")
        xs = sorted([int(v) for v in pivot.columns.tolist()])
        mat = pivot[xs].to_numpy(dtype=float)
        mu = np.nanmean(mat, axis=0)
        se = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.maximum(1, np.sum(np.isfinite(mat), axis=0)))
        axA.plot(xs, mu, color=col, linewidth=2.6, label=grp.replace("_", "-"))
        axA.fill_between(xs, mu - 1.96 * se, mu + 1.96 * se, color=col, alpha=0.22)
    axA.axvline(0, color="black", linestyle="--", linewidth=1)
    axA.set_ylim(0, 1.02)
    axA.set_xlabel("Relative trial to transition first-correct")
    axA.set_ylabel("p_acq")
    axA.set_title("Group Trajectory: Early (Sets 1-2) vs Late (Sets 7-8)")
    axA.legend(frameon=False)
    axA.grid(axis="y", linestyle="--", alpha=0.35)
    _add_panel_label(axA, "A")

    # Panel B: exemplars
    m = b[["subj", "block_id", "set_index", "acquisition_lag_core", "transition_width_20_to_80", "first_correct_trial", "acquisition_trial_core", "acquisition_trial_viterbi"]].dropna().copy()
    m["block_id"] = m["block_id"].astype(str)
    t2 = t.copy()
    t2["block_id"] = t2["block_id"].astype(str)

    def _is_acquired_after_core(row, min_tail=3, mean_thr=0.85, min_thr=0.70):
        sub = row["subj"]
        bid = str(row["block_id"])
        core = float(row["acquisition_trial_core"])
        z = t2[(t2["subj"] == sub) & (t2["block_id"] == bid)].copy()
        if z.empty or not np.isfinite(core):
            return False
        z = z.sort_values("trial_index_1based")
        tail = z[z["trial_index_1based"] >= core]
        if len(tail) < min_tail:
            return False
        p = pd.to_numeric(tail["p_acq"], errors="coerce").to_numpy(dtype=float)
        p = p[np.isfinite(p)]
        if len(p) < min_tail:
            return False
        return (np.nanmean(p) >= mean_thr) and (np.nanmin(p) >= min_thr)

    used_keys = set()

    def _pick_unique(cond, sort_cols, require_acquired=True):
        d2 = m[cond].copy()
        if d2.empty:
            return None
        d2["key"] = d2["subj"].astype(str) + "|" + d2["block_id"].astype(str)
        d2 = d2[~d2["key"].isin(used_keys)].copy()
        if d2.empty:
            return None
        if require_acquired:
            d2 = d2[d2.apply(_is_acquired_after_core, axis=1)].copy()
            if d2.empty:
                return None
        row = d2.sort_values(sort_cols).iloc[0]
        used_keys.add(str(row["subj"]) + "|" + str(row["block_id"]))
        return row

    ex = [
        (
            "fast (lag=0,width=0)",
            _pick_unique(
                (m["acquisition_lag_core"] == 0) & (m["transition_width_20_to_80"] == 0),
                ["set_index", "acquisition_lag_core", "transition_width_20_to_80"],
                require_acquired=True,
            ),
        ),
        (
            "early lag>0 (acquired)",
            _pick_unique(
                (m["set_index"] <= 2)
                & (m["acquisition_lag_core"] > 0)
                & (m["acquisition_lag_core"] <= 6)
                & (m["transition_width_20_to_80"] <= 6),
                ["set_index", "acquisition_lag_core", "transition_width_20_to_80"],
                require_acquired=True,
            ),
        ),
        (
            "early width>0 (acquired)",
            _pick_unique(
                (m["set_index"] <= 2)
                & (m["transition_width_20_to_80"] >= 2)
                & (m["transition_width_20_to_80"] <= 8),
                ["set_index", "transition_width_20_to_80", "acquisition_lag_core"],
                require_acquired=True,
            ),
        ),
        (
            "late fast",
            _pick_unique(
                (m["set_index"] >= 7)
                & (m["acquisition_lag_core"] == 0)
                & (m["transition_width_20_to_80"] == 0),
                ["set_index", "acquisition_lag_core", "transition_width_20_to_80"],
                require_acquired=True,
            ),
        ),
    ]

    for ax, (ttl, row) in zip(axsB, ex):
        if row is None:
            ax.text(0.5, 0.5, "No exemplar", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        sub = row["subj"]
        bid = str(row["block_id"])
        z = t2[(t2["subj"] == sub) & (t2["block_id"] == bid)].copy()
        z = z.sort_values("trial_index_1based")
        ax.plot(z["trial_index_1based"], z["p_acq"], color="#3182bd", linewidth=2)
        ax.axvline(row["first_correct_trial"], color="black", linestyle="--", linewidth=1, label="transition first-correct")
        ax.axvline(row["acquisition_trial_core"], color="#e6550d", linestyle="-", linewidth=1.2, label="acquired core")
        ax.axvline(row["acquisition_trial_viterbi"], color="#31a354", linestyle=":", linewidth=1.2, label="viterbi")
        ax.set_ylim(0, 1.02)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("Trial index")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    axsB[0].set_ylabel("p_acq")
    axsB[-1].legend(frameon=False, fontsize=8, loc="lower right")
    _add_panel_label(axsB[0], "B")

    fig.suptitle(fig_title, fontsize=15, y=0.98)
    out = os.path.join(OUT_DIR, "figure21_pacq_aligned_early_late_and_exemplars.png")
    fig.savefig(out, dpi=320, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    update_metrics_file(
        figure_id="figure21",
        figure_title=fig_title,
        figure_caption=fig_caption,
        rows=[
            {
                "panel": "A-B",
                "metric_name": "descriptive_trajectory_plot",
                "p_value": np.nan,
                "significance": "NA",
                "effect_or_stat": "",
                "notes": "Descriptive plot; no p-value in figure.",
            }
        ],
    )
    return out


def make_figure3_erp_stages(s: pd.DataFrame, p: pd.DataFrame):
    fig_title = "Feedback ERP Stage Profile: Discovery/Update vs Stabilization"
    fig_caption = (
        "Panels A/B show P3b stage means for early sets and all blocks. "
        "Panel C shows supportive early-set P3a stage means. Brackets use significance stars."
    )
    metric_rows = []
    fig = plt.figure(figsize=(12.6, 7.2), constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], hspace=0.36, wspace=0.28)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    order = ["search_error", "transition_first_correct", "transition_pre_core", "acquired_core"]
    labels = ["Search\nerror", "Transition\nfirst correct", "Transition\npre-core", "Acquired\ncore onset"]

    def _plot_stage(ax, subset_name: str, feat: str, title: str, color: str, drop_empty_first: bool = False):
        d = s[(s["subset"] == subset_name) if "subset" in s.columns else np.ones(len(s), dtype=bool)]
        if "subset" in s.columns:
            d = s[s["subset"] == subset_name].copy()
        else:
            d = s.copy()
            if subset_name == "early_sets":
                d = d[d["is_early_set"] == 1]
            elif subset_name == "all_blocks":
                pass
        d = d[d["feature"] == feat]
        # subject-level mean in each category
        g = d.groupby(["subj", "category"], as_index=False)["value_mean"].mean()
        piv = g.pivot_table(index="subj", columns="category", values="value_mean")
        order_use = list(order)
        if drop_empty_first and len(order_use) > 0 and order_use[0] == "search_error":
            # For panel B (all-block P3b), always suppress the first stage column.
            order_use = order_use[1:]

        xs = np.arange(len(order_use))
        mu, se = [], []
        for c in order_use:
            arr = piv[c].to_numpy(dtype=float) if c in piv.columns else np.array([])
            mu.append(np.nanmean(arr) if len(arr) else np.nan)
            se.append(np.nanstd(arr, ddof=1) / np.sqrt(np.sum(np.isfinite(arr))) if np.sum(np.isfinite(arr)) > 1 else np.nan)
        mu = np.array(mu, dtype=float)
        se = np.array(se, dtype=float)
        ax.errorbar(xs, mu, yerr=1.96 * se, color=color, marker="o", linewidth=2.2, capsize=3)
        ax.set_xticks(xs)
        label_map = dict(zip(order, labels))
        ax.set_xticklabels([label_map.get(c, c) for c in order_use])
        ax.set_title(title)
        ax.set_ylabel("Amplitude (uV)")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        mu_hi = mu + 1.96 * np.nan_to_num(se, nan=0.0)
        mu_hi = mu_hi[np.isfinite(mu_hi)]
        ytop = float(np.nanmax(mu_hi)) if len(mu_hi) else 0.0
        return ytop

    def _add_sig_top(ax, x1: float, x2: float, sig: str, level: int, yref: float):
        # Keep significance marks at top with extra headroom.
        y0 = yref + 0.16 + level * 0.22
        ax.plot([x1, x1, x2, x2], [y0 - 0.06, y0, y0, y0 - 0.06], color="black", linewidth=1)
        ax.text((x1 + x2) / 2, y0 + 0.006, sig, ha="center", va="bottom", fontsize=10.5, fontweight="bold")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, max(hi, y0 + 0.16))

    # Panel A: early P3b
    yrefA = _plot_stage(axA, "early_sets", "feedback_locked_P3b", "P3b Stage Means (Early Sets)", "#1f78b4")
    # significance bracket: transition first-correct vs transition pre-core
    q = p[
        (p["subset"] == "early_sets")
        & (p["feature"] == "feedback_locked_P3b")
        & (p["comparison"] == "transition_first_correct_minus_transition_pre_core")
    ]
    if not q.empty:
        pv = float(q.iloc[0]["p"])
        _add_sig_top(axA, 1, 2, p_to_sig(pv), level=0, yref=yrefA)
        metric_rows.append(
            {
                "panel": "A",
                "metric_name": "P3b early transition_first_correct_vs_transition_pre_core",
                "p_value": pv,
                "significance": p_to_sig(pv),
                "effect_or_stat": "",
                "notes": "",
            }
        )
    _add_panel_label(axA, "A")

    # Panel B: all P3b
    _plot_stage(
        axB,
        "all_blocks",
        "feedback_locked_P3b",
        "P3b Stage Means (All Blocks)",
        "#6baed6",
        drop_empty_first=True,
    )
    _add_panel_label(axB, "B")

    # Panel C: early P3a supportive panel
    yrefC = _plot_stage(axC, "early_sets", "feedback_locked_P3a", "P3a Stage Means (Early Sets, Supportive)", "#f28e2b")
    q2 = p[
        (p["subset"] == "early_sets")
        & (p["feature"] == "feedback_locked_P3a")
        & (p["comparison"] == "transition_first_correct_minus_transition_pre_core")
    ]
    if not q2.empty:
        pv = float(q2.iloc[0]["p"])
        _add_sig_top(axC, 1, 2, p_to_sig(pv), level=0, yref=yrefC)
        metric_rows.append(
            {
                "panel": "C",
                "metric_name": "P3a early transition_first_correct_vs_transition_pre_core",
                "p_value": pv,
                "significance": p_to_sig(pv),
                "effect_or_stat": "",
                "notes": "",
            }
        )
    d_count = s.copy()
    if "subset" in d_count.columns:
        d_count = d_count[d_count["subset"] == "early_sets"]
    else:
        d_count = d_count[d_count["is_early_set"] == 1]
    d_count = d_count[d_count["feature"] == "feedback_locked_P3a"]
    if "block_id" in d_count.columns:
        c_map = d_count.groupby("category")["block_id"].nunique().to_dict()
    else:
        c_map = d_count.groupby("category").size().to_dict()
    cnt_txt = (
        f"n blocks: search={int(c_map.get('search_error', 0))}, "
        f"transition-fc={int(c_map.get('transition_first_correct', 0))}, "
        f"transition-precore={int(c_map.get('transition_pre_core', 0))}, "
        f"acquired-core={int(c_map.get('acquired_core', 0))}"
    )
    axC.text(
        0.01,
        0.02,
        cnt_txt,
        transform=axC.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        bbox=dict(facecolor="white", alpha=0.78, edgecolor="0.85"),
    )
    _add_panel_label(axC, "C")

    fig.suptitle(fig_title, fontsize=15, y=0.98)
    out = os.path.join(OUT_DIR, "figure22_erp_stage_roles_p3b_focus.png")
    fig.savefig(out, dpi=320, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    if not metric_rows:
        metric_rows.append(
            {
                "panel": "A-C",
                "metric_name": "descriptive_stage_plot",
                "p_value": np.nan,
                "significance": "NA",
                "effect_or_stat": "",
                "notes": "No paired comparison available.",
            }
        )
    update_metrics_file(
        figure_id="figure22",
        figure_title=fig_title,
        figure_caption=fig_caption,
        rows=metric_rows,
    )
    return out


def make_figure4_predictive(s: pd.DataFrame, sp: pd.DataFrame, mx: pd.DataFrame):
    fig_title = "Transition first-correct FRN is associated with subsequent acquisition metrics"
    fig_caption = (
        "Panels A/B show binned means plus mixed-model marginal fits for FRN associations with lag and width. "
        "Panel C summarizes FRN robustness across model families with 95% CIs."
    )
    metric_rows = []
    # Use transition first-correct FRN block values from stage table.
    d = s[(s["category"] == "transition_first_correct") & (s["feature"].isin(["feedback_locked_FRN", "feedback_locked_P3a", "feedback_locked_P3b"]))].copy()
    piv = d.pivot_table(
        index=[
            "subj",
            "block_id",
            "lag_core",
            "width_20_80",
            "set_index",
            "rule_level",
            "shift_type",
            "is_early_set",
            "is_lag_pos",
            "is_width_pos",
        ],
        columns="feature",
        values="value_mean",
        aggfunc="first",
    ).reset_index()
    piv.columns.name = None
    sens = pd.read_csv(PATH_SENS) if os.path.exists(PATH_SENS) else pd.DataFrame()
    cnt = pd.read_csv(PATH_COUNT) if os.path.exists(PATH_COUNT) else pd.DataFrame()

    fig = plt.figure(figsize=(12.8, 7.6), constrained_layout=False)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1.1], hspace=0.35, wspace=0.28)
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[:, 1])
    axC = fig.add_subplot(gs[:, 2])

    def _fit_mixed_for_x(dd: pd.DataFrame, ycol: str, xcol: str):
        dfit = dd[[ycol, xcol, "set_index", "shift_type", "rule_level", "subj"]].dropna().copy()
        dfit = dfit[dfit["shift_type"] != "first_block"].copy()
        if len(dfit) < 80:
            return None, None, None
        formula = f"{ycol} ~ {xcol} + set_index + C(shift_type) + C(rule_level)"
        try:
            m = smf.mixedlm(formula, dfit, groups=dfit["subj"], re_formula="1")
            r = m.fit(method="lbfgs", reml=False, maxiter=200, disp=False)
        except Exception:
            return None, None, None

        # Build marginal prediction curve over x with covariates fixed at typical values.
        xg = np.linspace(np.nanpercentile(dfit[xcol], 5), np.nanpercentile(dfit[xcol], 95), 120)
        mode_shift = dfit["shift_type"].mode().iloc[0]
        mode_rule = dfit["rule_level"].mode().iloc[0]
        med_set = float(np.nanmedian(dfit["set_index"]))
        exog_df = pd.DataFrame(
            {
                xcol: xg,
                "set_index": med_set,
                "shift_type": mode_shift,
                "rule_level": mode_rule,
            }
        )
        rhs = formula.split("~", 1)[1]
        Xnew = dmatrix(rhs, exog_df, return_type="dataframe")

        beta = r.fe_params.copy()
        cov = r.cov_params().loc[beta.index, beta.index].copy()
        Xuse = Xnew.reindex(columns=beta.index, fill_value=0.0).to_numpy(dtype=float)
        b = beta.to_numpy(dtype=float)
        yhat = Xuse @ b
        var = np.einsum("ij,jk,ik->i", Xuse, cov.to_numpy(dtype=float), Xuse)
        se = np.sqrt(np.clip(var, 0, np.inf))
        lo = yhat - 1.96 * se
        hi = yhat + 1.96 * se
        return r, xg, (yhat, lo, hi)

    def _panel_marginal(ax, xcol: str, ycol: str, title: str, key_metric: str, panel_id: str, y_main_max: float = 10.0):
        dd = piv[[xcol, ycol, "set_index", "shift_type", "rule_level", "subj"]].dropna().copy()
        x = dd[xcol].to_numpy(dtype=float)
        y = dd[ycol].to_numpy(dtype=float)
        n_out = int(np.sum(y > y_main_max))

        # 1) Binned means (deciles) with CI.
        q = pd.qcut(dd[xcol], q=min(10, max(4, len(dd) // 80)), duplicates="drop")
        bstat = (
            dd.assign(bin=q)
            .groupby("bin", as_index=False)
            .agg(
                x_mid=(xcol, "median"),
                y_mean=(ycol, "mean"),
                n=(ycol, "count"),
                y_sd=(ycol, "std"),
            )
        )
        bstat["se"] = bstat["y_sd"] / np.sqrt(np.maximum(1, bstat["n"]))
        ax.errorbar(
            bstat["x_mid"],
            bstat["y_mean"],
            yerr=1.96 * bstat["se"],
            fmt="o",
            color="#756bb1",
            markersize=5,
            capsize=3,
            alpha=0.95,
            label="binned mean +/- 95% CI",
        )

        # 2) Mixed-model marginal prediction + 95% CI.
        fit, xg, pred = _fit_mixed_for_x(dd, ycol=ycol, xcol=xcol)
        if fit is not None and pred is not None:
            yhat, lo, hi = pred
            ax.plot(xg, yhat, color="#e6550d", linewidth=2.2, label="mixed-model marginal fit")
            ax.fill_between(xg, lo, hi, color="#e6550d", alpha=0.18, linewidth=0)

        # Main panel + inset
        ax.set_ylim(-0.5, y_main_max)
        axins = ax.inset_axes([0.58, 0.58, 0.40, 0.38])
        axins.scatter(x, y, s=7, alpha=0.20, color="#6a51a3")
        if fit is not None and pred is not None:
            axins.plot(xg, yhat, color="#e6550d", linewidth=1.3)
        y_hi = float(np.nanmax(y)) if len(y) else y_main_max
        axins.set_ylim(-0.5, max(y_main_max, y_hi * 1.03))
        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_title("full range", fontsize=7)

        # annotate stats
        q1 = sp[(sp["subset"] == "all_blocks") & (sp["x_feature"] == xcol) & (sp["y_metric"] == key_metric)]
        q2 = mx[(mx["x_feature"] == xcol) & (mx["y_metric"] == key_metric)]
        txt = []
        if not q1.empty:
            p1 = float(q1.iloc[0]["p"])
            metric_rows.append(
                {
                    "panel": panel_id,
                    "metric_name": f"{key_metric}_spearman",
                    "p_value": p1,
                    "significance": p_to_sig(p1),
                    "effect_or_stat": f"rho={float(q1.iloc[0]['spearman_rho']):.3f}",
                    "notes": "",
                }
            )
        if not q2.empty:
            p2 = float(q2.iloc[0]["p_x"])
            txt.append(f"MixedLM(full): {p_to_sig(p2)}")
            metric_rows.append(
                {
                    "panel": panel_id,
                    "metric_name": f"{key_metric}_mixedlm_full",
                    "p_value": p2,
                    "significance": p_to_sig(p2),
                    "effect_or_stat": f"beta={float(q2.iloc[0]['beta_x']):.3f}" if "beta_x" in q2.columns else "",
                    "notes": "",
                }
            )
        # Add robustness summaries (drop2 + count models).
        if not sens.empty:
            srow = sens[sens["model"] == ("lag_drop2" if key_metric == "lag_core" else "width_drop2")]
            if not srow.empty:
                p3 = float(srow.iloc[0]["p_FRN"])
                txt.append(f"MixedLM(drop2): {p_to_sig(p3)}")
                metric_rows.append(
                    {
                        "panel": panel_id,
                        "metric_name": f"{key_metric}_mixedlm_drop2",
                        "p_value": p3,
                        "significance": p_to_sig(p3),
                        "effect_or_stat": f"beta={float(srow.iloc[0]['beta_FRN']):.3f}",
                        "notes": "",
                    }
                )
        if not cnt.empty:
            crow1 = cnt[cnt["model"] == ("gee_poisson_lag" if key_metric == "lag_core" else "gee_poisson_width")]
            crow2 = cnt[cnt["model"] == ("glm_nb_lag" if key_metric == "lag_core" else "glm_nb_width")]
            if not crow1.empty:
                p4 = float(crow1.iloc[0]["p_FRN"])
                metric_rows.append(
                    {
                        "panel": panel_id,
                        "metric_name": f"{key_metric}_poisson_gee",
                        "p_value": p4,
                        "significance": p_to_sig(p4),
                        "effect_or_stat": f"beta={float(crow1.iloc[0]['beta_FRN']):.3f}",
                        "notes": "",
                    }
                )
            if not crow2.empty:
                p5 = float(crow2.iloc[0]["p_FRN"])
                txt.append(f"NB robust: {p_to_sig(p5)}")
                metric_rows.append(
                    {
                        "panel": panel_id,
                        "metric_name": f"{key_metric}_nb_robust",
                        "p_value": p5,
                        "significance": p_to_sig(p5),
                        "effect_or_stat": f"beta={float(crow2.iloc[0]['beta_FRN']):.3f}",
                        "notes": "",
                    }
                )
        if n_out > 0:
            txt.append(f"out-of-range points: {n_out}")
        if txt:
            ax.text(
                0.03,
                0.97,
                "\n".join(txt),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.8"),
            )
        ax.set_title(title)
        ax.set_xlabel("First-correct FRN amplitude (uV)")
        ax.set_ylabel(ycol)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        y0, y1 = ax.get_ylim()
        y_target = 3.0
        y_frac = (y_target - y0) / (y1 - y0) if y1 > y0 else 0.30
        y_frac = float(np.clip(y_frac, 0.12, 0.88))
        ax.legend(
            frameon=False,
            fontsize=8,
            loc="center right",
            bbox_to_anchor=(0.96, y_frac),
        )

    _panel_marginal(
        axA,
        "feedback_locked_FRN",
        "lag_core",
        "FRN vs lag\n(association)",
        "lag_core",
        "A",
        y_main_max=10.0,
    )
    _add_panel_label(axA, "A")

    _panel_marginal(
        axB,
        "feedback_locked_FRN",
        "width_20_80",
        "FRN vs width\n(association)",
        "width_20_80",
        "B",
        y_main_max=10.0,
    )
    _add_panel_label(axB, "B")

    # Panel C: FRN robustness summary across models.
    robust_rows = []
    if not sens.empty:
        for mname, metric in [("lag_full", "lag_core"), ("lag_drop2", "lag_core"), ("width_full", "width_20_80"), ("width_drop2", "width_20_80")]:
            q = sens[sens["model"] == mname]
            if not q.empty:
                robust_rows.append(
                    {
                        "model_label": mname.replace("_", " "),
                        "metric": metric,
                        "beta": float(q.iloc[0]["beta_FRN"]),
                        "se": float(q.iloc[0]["se_FRN"]),
                    }
                )
    if not cnt.empty:
        for mname, metric in [("gee_poisson_lag", "lag_core"), ("gee_poisson_width", "width_20_80"), ("glm_nb_lag", "lag_core"), ("glm_nb_width", "width_20_80")]:
            q = cnt[cnt["model"] == mname]
            if not q.empty:
                robust_rows.append(
                    {
                        "model_label": mname.replace("_", " "),
                        "metric": metric,
                        "beta": float(q.iloc[0]["beta_FRN"]),
                        "se": float(q.iloc[0]["se_FRN"]),
                    }
                )

    rr = pd.DataFrame(robust_rows)
    # fixed order for readability
    order = [
        "lag full",
        "lag drop2",
        "gee poisson lag",
        "glm nb lag",
        "width full",
        "width drop2",
        "gee poisson width",
        "glm nb width",
    ]
    rr["model_label"] = rr["model_label"].str.lower()
    rr = rr.set_index("model_label").reindex(order).reset_index()
    xbase = np.arange(len(rr))
    colors = np.where(rr["metric"] == "lag_core", "#1f78b4", "#33a02c")
    axC.scatter(xbase, rr["beta"], s=46, c=colors, zorder=3)
    for xi, yi, se, col in zip(xbase, rr["beta"], rr["se"], colors):
        if np.isfinite(yi) and np.isfinite(se):
            axC.plot([xi, xi], [yi - 1.96 * se, yi + 1.96 * se], color=col, linewidth=1.8, zorder=2)
    axC.axhline(0, color="black", linewidth=1)
    axC.set_xticks(xbase)
    axC.set_xticklabels(
        [
            "lag\nfull",
            "lag\ndrop2",
            "lag\nPoisson GEE",
            "lag\nNB",
            "width\nfull",
            "width\ndrop2",
            "width\nPoisson GEE",
            "width\nNB",
        ],
        fontsize=8,
        rotation=30,
        ha="right",
    )
    axC.set_ylabel("FRN coefficient")
    axC.set_title("FRN Robustness Across Model Families")
    axC.grid(axis="y", linestyle="--", alpha=0.35)
    _add_panel_label(axC, "C")

    fig.suptitle(fig_title, fontsize=15, y=0.98)
    fig.text(
        0.50,
        0.02,
        "Note: Points represent block-level observations; inferential statistics are from mixed-effects models with repeated measures within subjects.",
        fontsize=9,
        ha="center",
        va="bottom",
    )
    out = os.path.join(OUT_DIR, "figure23_frn_predicts_acquisition_speed.png")
    fig.savefig(out, dpi=320, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    if not metric_rows:
        metric_rows.append(
            {
                "panel": "A-C",
                "metric_name": "descriptive_predictive_plot",
                "p_value": np.nan,
                "significance": "NA",
                "effect_or_stat": "",
                "notes": "No inferential rows available.",
            }
        )
    update_metrics_file(
        figure_id="figure23",
        figure_title=fig_title,
        figure_caption=fig_caption,
        rows=metric_rows,
    )
    return out


def main():
    b, t, s, p, sp, mx = _prep()
    outs = []
    outs.append(make_figure1_behavior(b))
    outs.append(make_figure2_time_series(b, t))
    outs.append(make_figure3_erp_stages(s, p))
    outs.append(make_figure4_predictive(s, sp, mx))
    for o in outs:
        print(f"saved: {o}")


if __name__ == "__main__":
    main()
