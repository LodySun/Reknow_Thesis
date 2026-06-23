import os
import shutil
import string
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, ttest_rel
import mne


BASE = "/Users/lodysun/Desktop/Thesis"
COMP_TAG = "1s_comp"

ALL_TRIAL = os.path.join(BASE, "trials_trialwise", "trialwise", "all_subjects_trialwise.csv")
HMM_TRIAL = os.path.join(BASE, "trials_trialwise", "hmm_long_tables", "hmm_trial_long.csv")
TRANS_TRIAL = os.path.join(BASE, "trials_trialwise", "hmm_mixture", "hmm_mixture_transition_trials_used.csv")
EEG_TRIAL = os.path.join(BASE, "trials_trialwise", COMP_TAG, "eeg_tables", "eeg_trial_long.csv")
EEG_CUE = os.path.join(BASE, "trials_trialwise", COMP_TAG, "eeg_tables", "eeg_trial_cue_locked.csv")
EEG_DIR = os.path.join(BASE, "prc", "ctap", "base_wcst", "correct_wcst_5", "tppd", "4_ADJUST_IC_CORR")

PRIORITY_DATA = os.path.join(
    BASE, "trials_trialwise", COMP_TAG, "eeg_paper_results", "priority_regressions", "priority_regression_data.csv"
)
PRIORITY_STATS = os.path.join(
    BASE, "trials_trialwise", COMP_TAG, "eeg_paper_results", "priority_regressions", "priority_regressions_spearman_no_resid.csv"
)
ISRSA_TRIPTYCH = os.path.join(
    BASE,
    "trials_trialwise",
    COMP_TAG,
    "eeg_paper_results",
    "is_rsa_model3_robustness",
    "model3_drop_mixture_is_rsa_triptych_1s_comp.png",
)
HMM_CV_SUBJ = os.path.join(BASE, "trials_trialwise", "hmm_baseline_ppc", "hmm_baseline_cv_subject_summary.csv")
HMM_CV_FOLD = os.path.join(BASE, "trials_trialwise", "hmm_baseline_ppc", "hmm_baseline_cv_by_fold.csv")
SOFT_BLOCK = os.path.join(BASE, "trials_trialwise", "hmm_mixture", "hmm_mixture_soft_block_labels.csv")
HMM_SET_BY_SET = os.path.join(BASE, "trials_trialwise", "hmm_set", "hmm_set_by_set_fullonly.csv")

OUT_DIR = os.path.join(BASE, "trials_trialwise", COMP_TAG, "eeg_paper_results", "main_text_figures_bundle")
OUT_DIR = os.environ.get("OUT_DIR", OUT_DIR)

# Panel titles off by default (caption in LaTeX). Set FIG_SUBPLOT_TITLES=1 to restore.
SUBPLOT_TITLES = os.environ.get("FIG_SUBPLOT_TITLES", "").lower() in ("1", "true", "yes")

ROI_FRONT = ["Fz", "FC1", "FC2", "Cz", "F3", "F4"]
FEEDBACK_CODE = "60"
CANDY = ["#9FC5E8", "#F4B6C2", "#B6D7A8", "#FFE599", "#C9B6E4", "#A7D8DE"]

# G1/G2 (global): object vs color; L1/L2 (local): object vs orientation — same as trialwise_parser.RULE_TYPE_MAP.
TRUE_RULE_TO_STIMULUS_FEATURE = {"G1": "obj", "G2": "col", "L1": "obj", "L2": "ori"}

plt.rcParams.update(
    {
        "font.family": "Arial",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    }
)

# Global style knobs for compact bar panels
BAR_WIDTH = 0.42
BAR_X_STEP = 0.78
BAR_JITTER = 0.045
BAR_BOX_ASPECT = 0.72
SAVE_PAD_INCHES = 0.05

# Filled by grouped_bar / waveform annotations; saved at end of main() as figure_inferential_tests.csv.
FIGURE_INFERENCE_ROWS: List[Dict[str, Any]] = []


def clear_figure_inference_log() -> None:
    FIGURE_INFERENCE_ROWS.clear()


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    idx = np.where(ok)[0]
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adj = np.empty(m, dtype=float)
    adj[-1] = ranked[-1]
    for i in range(m - 2, -1, -1):
        adj[i] = min(ranked[i] * m / (i + 1), adj[i + 1])
    restored = np.empty(m, dtype=float)
    restored[order] = np.clip(adj, 0.0, 1.0)
    out[idx] = restored
    return out


def _append_inference_row(row: Dict[str, Any]) -> None:
    FIGURE_INFERENCE_ROWS.append(row)


def _save_figure_compact(fig, filename: str) -> None:
    fig.savefig(
        os.path.join(OUT_DIR, filename),
        dpi=300,
        bbox_inches="tight",
        pad_inches=SAVE_PAD_INCHES,
    )


def save_figure_inference_table(out_dir: str) -> str:
    path = os.path.join(out_dir, "figure_inferential_tests.csv")
    cols = [
        "figure_file",
        "panel",
        "subplot_title",
        "test",
        "predictor",
        "outcome",
        "level_a",
        "level_b",
        "n_pairs",
        "t_statistic",
        "p_value",
        "mean_diff_a_minus_b",
        "note",
        "p_fdr",
    ]
    if FIGURE_INFERENCE_ROWS:
        df = pd.DataFrame(FIGURE_INFERENCE_ROWS).reindex(columns=cols)
        # BH FDR within S2 hypothesis families:
        # figure1_behavior_rule_dimensions rows grouped by outcome (logRT family: 3 tests; accuracy family: 3 tests).
        fig1 = df["figure_file"] == "figure1_behavior_rule_dimensions.png"
        for _, grp_idx in df[fig1].groupby("outcome").groups.items():
            df.loc[grp_idx, "p_fdr"] = _bh_fdr(df.loc[grp_idx, "p_value"].to_numpy(dtype=float))
        df.to_csv(path, index=False)
    else:
        pd.DataFrame(columns=cols).to_csv(path, index=False)
    return path


def _panel_label(ax, letter: str) -> None:
    ax.text(
        0.02,
        0.98,
        letter,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
        ha="left",
        color="black",
    )


def _label_panels(axes: Union[np.ndarray, Sequence], labels: Optional[str] = None) -> None:
    """Assign A, B, C, ... to axes in row-major (flattened) order."""
    flat = np.atleast_1d(axes).ravel().tolist()
    if labels is None:
        labels = string.ascii_uppercase[: len(flat)]
    else:
        labels = list(labels)
    for ax, lab in zip(flat, labels):
        _panel_label(ax, lab)


def to_num(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_behavior_master() -> pd.DataFrame:
    a = pd.read_csv(ALL_TRIAL)
    h = pd.read_csv(HMM_TRIAL)
    t = pd.read_csv(TRANS_TRIAL)

    if "true_rule" in a.columns:
        tr = a["true_rule"].astype(str).str.strip()
        a = a.copy()
        a["rule_type"] = tr.map(TRUE_RULE_TO_STIMULUS_FEATURE)

    a = to_num(a, ["block_id", "trial_id", "trial_since_rule_switch", "correctness", "rt"])
    h = to_num(
        h,
        [
            "block_id",
            "trial_id",
            "trial_in_block",
            "correctness",
            "rt",
            "first_correct_flag",
            "post_first_correct_flag",
            "stable_flag",
            "search_flag",
        ],
    )
    t = to_num(t, ["block_id", "trial_id", "prev_correct", "time_since_collapse"])

    m = a.merge(
        h[
            [
                "subj",
                "block_id",
                "trial_id",
                "trial_in_block",
                "first_correct_flag",
                "post_first_correct_flag",
                "stable_flag",
                "search_flag",
            ]
        ],
        on=["subj", "block_id", "trial_id"],
        how="left",
    )
    m = m.merge(
        t[["subj", "block_id", "trial_id", "prev_correct", "time_since_collapse"]],
        on=["subj", "block_id", "trial_id"],
        how="left",
    )

    if "trial_in_block" not in m.columns:
        m["trial_in_block"] = m["trial_since_rule_switch"]
    m["trial_in_block"] = pd.to_numeric(m["trial_in_block"], errors="coerce")
    m["rt"] = pd.to_numeric(m["rt"], errors="coerce")
    m["correctness"] = pd.to_numeric(m["correctness"], errors="coerce")
    m = m[(m["rt"] > 0) & m["correctness"].isin([0, 1])].copy()
    m["log_rt"] = np.log(m["rt"])

    fc = (
        m[m["first_correct_flag"] == 1]
        .groupby(["subj", "block_id"], as_index=False)["trial_in_block"]
        .min()
        .rename(columns={"trial_in_block": "first_correct_latency"})
    )
    block_meta = (
        m.groupby(["subj", "block_id"], as_index=False)
        .agg(rule_level=("rule_level", "first"), rule_type=("rule_type", "first"))
        .merge(fc, on=["subj", "block_id"], how="left")
    )
    block_meta["search_len"] = block_meta["first_correct_latency"] - 1
    block_meta = block_meta.sort_values(["subj", "block_id"]).copy()
    block_meta["prev_rule_level"] = block_meta.groupby("subj")["rule_level"].shift(1)
    block_meta["shift_type"] = np.where(
        block_meta["prev_rule_level"].isna(),
        "missing",
        np.where(block_meta["prev_rule_level"] == block_meta["rule_level"], "within_level", "cross_level"),
    )

    def _short_rule_lv(v) -> Optional[str]:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        s = str(v).strip().lower()
        if s == "global":
            return "G"
        if s == "local":
            return "L"
        return None

    pls = block_meta["prev_rule_level"].map(_short_rule_lv)
    cls = block_meta["rule_level"].map(_short_rule_lv)
    st = block_meta["shift_type"].astype(str)
    detail = []
    for i in range(len(block_meta)):
        p, c, s = pls.iloc[i], cls.iloc[i], st.iloc[i]
        if s == "missing" or p is None or c is None:
            detail.append("missing")
        elif s == "within_level":
            detail.append(f"{c}-{c}")
        else:
            detail.append(f"{p}-{c}")
    block_meta["shift_detail"] = detail

    m = m.merge(
        block_meta[
            ["subj", "block_id", "first_correct_latency", "search_len", "shift_type", "shift_detail"]
        ],
        on=["subj", "block_id"],
        how="left",
    )

    m["position_bin"] = np.where(
        m["trial_in_block"] == 1,
        "t1",
        np.where(m["trial_in_block"].isin([2, 3]), "t2_3", "t4plus"),
    )
    m["phase3"] = np.where(
        m["stable_flag"] == 1,
        "stable",
        np.where(
            (m["post_first_correct_flag"] == 1) | (m["first_correct_flag"] == 1),
            "first_correct_to_pre_stable",
            "pre_first_correct",
        ),
    )
    m["post_collapse"] = np.where(m["time_since_collapse"].notna() & (m["time_since_collapse"] >= 0), 1, 0)

    m = m.sort_values(["subj", "block_id", "trial_in_block"], kind="mergesort").reset_index(drop=True)
    m["next_correct"] = m.groupby(["subj", "block_id"], sort=False)["correctness"].shift(-1)

    m["prev_correct_bin"] = np.where(
        pd.to_numeric(m["prev_correct"], errors="coerce").isna(),
        np.nan,
        (pd.to_numeric(m["prev_correct"], errors="coerce") > 0).astype(float),
    )

    tsc_all = pd.to_numeric(m["time_since_collapse"], errors="coerce")
    post_ok = (m["post_collapse"] == 1) & tsc_all.notna()
    kf = np.floor(tsc_all[post_ok].astype(float)).astype(int)
    collapse_vals = np.full(len(m), np.nan, dtype=object)
    lab = np.where(kf >= 3, "3+", np.array([str(int(i)) for i in kf]))
    collapse_vals[np.where(post_ok)[0]] = lab
    m["collapse_bin"] = collapse_vals

    return m


def _p_to_star(p: float) -> str:
    if not np.isfinite(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _p_report_with_threshold(p: float) -> str:
    """Single line for figures: p≤ bins when significant, else p>0.05 (no exact p on plots)."""
    if not np.isfinite(p):
        return "p=n/a"
    if p > 0.05:
        return "p>0.05"
    if p <= 0.001:
        return "p≤0.001"
    if p <= 0.01:
        return "p≤0.01"
    return "p≤0.05"


def _bracket_label_from_p(p: float) -> str:
    star = _p_to_star(p)
    if not np.isfinite(p):
        return f"{star}\n(n<6 paired subjects)"
    return f"{star}\n{_p_report_with_threshold(p)}"


def _paired_ttest_rel_details(subj_level: pd.DataFrame, xcol: str, ycol: str, a, b) -> Optional[dict]:
    aa = subj_level[subj_level[xcol] == a][["subj", ycol]].rename(columns={ycol: "a"})
    bb = subj_level[subj_level[xcol] == b][["subj", ycol]].rename(columns={ycol: "b"})
    m = aa.merge(bb, on="subj", how="inner").dropna()
    if len(m) < 6:
        return None
    diff = m["a"].to_numpy(dtype=float) - m["b"].to_numpy(dtype=float)
    if np.allclose(diff, 0.0, rtol=0.0, atol=1e-12):
        return {
            "n_pairs": int(len(m)),
            "t_statistic": 0.0,
            "p_value": 1.0,
            "mean_diff_a_minus_b": 0.0,
        }
    res = ttest_rel(m["a"], m["b"], nan_policy="omit")
    return {
        "n_pairs": int(len(m)),
        "t_statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "mean_diff_a_minus_b": float(np.mean(diff)),
    }


def _paired_p(subj_level: pd.DataFrame, xcol: str, ycol: str, a, b) -> float:
    det = _paired_ttest_rel_details(subj_level, xcol, ycol, a, b)
    if det is None:
        return np.nan
    return float(det["p_value"])


def grouped_bar(
    ax,
    df,
    xcol,
    ycol,
    order,
    title,
    ylabel,
    sig_pairs=None,
    stat_meta: Optional[Tuple[str, str]] = None,
    bar_width: Optional[float] = None,
    jitter_width: Optional[float] = None,
    box_aspect: Optional[float] = None,
):
    bw = float(BAR_WIDTH if bar_width is None else bar_width)
    jw = float(BAR_JITTER if jitter_width is None else jitter_width)
    ba = float(BAR_BOX_ASPECT if box_aspect is None else box_aspect)

    sub = df[df[xcol].isin(order)].groupby(["subj", xcol], as_index=False)[ycol].mean()
    g = sub.groupby(xcol)[ycol].agg(["mean", "sem"]).reindex(order)
    x = np.arange(len(order), dtype=float) * BAR_X_STEP
    x_pos = {cat: float(xx) for cat, xx in zip(order, x)}
    bar_colors = (CANDY * ((len(order) // len(CANDY)) + 1))[: len(order)]
    ax.bar(
        x,
        g["mean"].values,
        yerr=g["sem"].values,
        width=bw,
        capsize=4,
        alpha=0.9,
        color=bar_colors,
        edgecolor="#666666",
        linewidth=0.8,
    )
    # Subject-level jittered points
    rng = np.random.default_rng(20260320)
    for i, cat in enumerate(order):
        vals = sub[sub[xcol] == cat][ycol].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-jw, jw, size=len(vals))
        ax.scatter(
            np.full(len(vals), x[i]) + jitter,
            vals,
            s=28,
            alpha=0.55,
            marker="o",
            facecolors="black",
            edgecolors="black",
            linewidths=0.35,
            zorder=3,
        )

    # y-axis: must include all subject points and error bars (do not clip scatter).
    pts = sub[sub[xcol].isin(order)][ycol].dropna().to_numpy(dtype=float)
    bar_low = float(np.nanmin(g["mean"].values - np.nan_to_num(g["sem"].values, nan=0.0)))
    bar_high = float(np.nanmax(g["mean"].values + np.nan_to_num(g["sem"].values, nan=0.0)))
    if len(pts):
        pmin = float(np.nanmin(pts))
        pmax = float(np.nanmax(pts))
    else:
        pmin, pmax = bar_low, bar_high
    ymin_raw = float(min(bar_low, pmin))
    ymax_raw = float(max(bar_high, pmax))
    yspan = max(ymax_raw - ymin_raw, 1e-9)
    pad = 0.05 * yspan
    y_bottom = ymin_raw - pad
    y_top = ymax_raw + pad

    # Significance brackets (only extend upper limit); stars + p≤ / p>0.05 (no exact p).
    # Anchor above *all* displayed y (mean±SEM and subject means) so log-RT panels match accuracy spacing.
    if sig_pairs:
        ymax_b = float(np.nanmax(g["mean"].values + np.nan_to_num(g["sem"].values, nan=0.0)))
        ymin_b = float(np.nanmin(g["mean"].values - np.nan_to_num(g["sem"].values, nan=0.0)))
        data_top = float(max(ymax_b, pmax))
        data_bot = float(min(ymin_b, pmin))
        yr = max(data_top - data_bot, 1e-6)
        valid_pairs = [(a, b) for (a, b) in sig_pairs if a in order and b in order]
        n_br = len(valid_pairs)
        step = (0.13 + 0.06 * max(0, n_br - 1)) * yr
        base = data_top + 0.10 * yr
        upper = y_top
        fig_stem, panel_letter = stat_meta if stat_meta else ("", "")
        for k, (a, b) in enumerate(valid_pairs):
            i1 = order.index(a)
            i2 = order.index(b)
            xa = x_pos[a]
            xb = x_pos[b]
            det = _paired_ttest_rel_details(sub, xcol, ycol, a, b)
            p = float(det["p_value"]) if det else np.nan
            label = _bracket_label_from_p(p)
            y = base + k * step
            drop = 0.012 * yr
            ax.plot([xa, xa, xb, xb], [y - drop, y, y, y - drop], color="black", linewidth=1.4)
            ax.text((xa + xb) / 2, y + 0.02 * yr, label, ha="center", va="bottom", fontsize=10, color="#7A1F1F", linespacing=1.05)
            upper = max(upper, y + 0.34 * yr)
            if stat_meta and det:
                _append_inference_row(
                    {
                        "figure_file": f"{fig_stem}.png",
                        "panel": panel_letter,
                        "subplot_title": title,
                        "test": "paired_ttest_rel_subject_means",
                        "predictor": xcol,
                        "outcome": ycol,
                        "level_a": str(a),
                        "level_b": str(b),
                        "n_pairs": det["n_pairs"],
                        "t_statistic": det["t_statistic"],
                        "p_value": det["p_value"],
                        "mean_diff_a_minus_b": det["mean_diff_a_minus_b"],
                        "note": "",
                    }
                )
        if n_br > 0:
            upper = max(upper, base + max(n_br - 1, 0) * step + 0.36 * yr)
            y_top = max(y_top, upper)

    ax.set_ylim(y_bottom, y_top)
    # Slightly tighter x margins so bars don't look sparse.
    if len(x) > 0:
        ax.set_xlim(float(np.min(x) - 0.32), float(np.max(x) + 0.32))
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=15)
    ax.set_box_aspect(ba)
    if SUBPLOT_TITLES:
        ax.set_title(title)
    ax.set_ylabel(ylabel)


def scatter_with_fit(ax, x, y, title, xlabel, ylabel):
    ax.scatter(x, y, alpha=0.8, s=36, c="black", edgecolors="black", linewidths=0.4)
    if len(x) > 1:
        p = np.polyfit(x, y, 1)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xx, p[0] * xx + p[1], linewidth=2)
    if SUBPLOT_TITLES:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def make_behavior_figures(df: pd.DataFrame) -> None:
    # Figure 1 (6 panels): within-Global feature, within-Local feature, marginal Global vs Local.
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 9.2), constrained_layout=False)
    fig1_bar_width = BAR_WIDTH * 0.60
    fig1_jitter = BAR_JITTER * 0.60
    fig1_box_aspect = BAR_BOX_ASPECT * 0.90

    global_type_order = ["col", "obj"]
    d_global = df.copy()
    if "rule_level" in d_global.columns and "rule_type" in d_global.columns:
        d_global["rule_level"] = d_global["rule_level"].astype(str).str.strip().str.lower()
        d_global["rule_type"] = d_global["rule_type"].astype(str).str.strip().str.lower()
        d_global = d_global[
            (d_global["rule_level"] == "global") & d_global["rule_type"].isin(global_type_order)
        ].copy()
    else:
        d_global = d_global.iloc[0:0].copy()

    if len(d_global) > 0:
        grouped_bar(
            axes[0, 0],
            d_global,
            "rule_type",
            "log_rt",
            global_type_order,
            "Log-RT (Global: Color vs Object)",
            "Log(RT)",
            sig_pairs=[("col", "obj")],
            stat_meta=("figure1_behavior_rule_dimensions", "A"),
            bar_width=fig1_bar_width,
            jitter_width=fig1_jitter,
            box_aspect=fig1_box_aspect,
        )
        axes[0, 0].set_xticklabels(["Color", "Object"], rotation=15)
        grouped_bar(
            axes[0, 1],
            d_global,
            "rule_type",
            "correctness",
            global_type_order,
            "Accuracy (Global: Color vs Object)",
            "Accuracy",
            sig_pairs=[("col", "obj")],
            stat_meta=("figure1_behavior_rule_dimensions", "B"),
            bar_width=fig1_bar_width,
            jitter_width=fig1_jitter,
            box_aspect=fig1_box_aspect,
        )
        axes[0, 1].set_xticklabels(["Color", "Object"], rotation=15)
    else:
        axes[0, 0].text(
            0.5, 0.5, "global rule_type missing", ha="center", va="center", transform=axes[0, 0].transAxes
        )
        axes[0, 1].text(0.5, 0.5, "global rule_type missing", ha="center", va="center", transform=axes[0, 1].transAxes)

    local_type_order = ["ori", "obj"]
    d_local = df.copy()
    if "rule_level" in d_local.columns and "rule_type" in d_local.columns:
        d_local["rule_level"] = d_local["rule_level"].astype(str).str.strip().str.lower()
        d_local["rule_type"] = d_local["rule_type"].astype(str).str.strip().str.lower()
        d_local = d_local[
            (d_local["rule_level"] == "local") & d_local["rule_type"].isin(local_type_order)
        ].copy()
    else:
        d_local = d_local.iloc[0:0].copy()

    if len(d_local) > 0:
        grouped_bar(
            axes[0, 2],
            d_local,
            "rule_type",
            "log_rt",
            local_type_order,
            "Log-RT (Local: Orientation vs Object)",
            "Log(RT)",
            sig_pairs=[("ori", "obj")],
            stat_meta=("figure1_behavior_rule_dimensions", "C"),
            bar_width=fig1_bar_width,
            jitter_width=fig1_jitter,
            box_aspect=fig1_box_aspect,
        )
        axes[0, 2].set_xticklabels(["Orientation", "Object"], rotation=15)
        grouped_bar(
            axes[1, 0],
            d_local,
            "rule_type",
            "correctness",
            local_type_order,
            "Accuracy (Local: Orientation vs Object)",
            "Accuracy",
            sig_pairs=[("ori", "obj")],
            stat_meta=("figure1_behavior_rule_dimensions", "D"),
            bar_width=fig1_bar_width,
            jitter_width=fig1_jitter,
            box_aspect=fig1_box_aspect,
        )
        axes[1, 0].set_xticklabels(["Orientation", "Object"], rotation=15)
    else:
        axes[0, 2].text(0.5, 0.5, "local rule_type missing", ha="center", va="center", transform=axes[0, 2].transAxes)
        axes[1, 0].text(0.5, 0.5, "local rule_type missing", ha="center", va="center", transform=axes[1, 0].transAxes)

    # E–F: object rule only (G1 vs L1) — Global obj vs Local obj.
    level_order = ["global", "local"]
    d_obj_gl = df.copy()
    if "rule_level" in d_obj_gl.columns and "rule_type" in d_obj_gl.columns:
        d_obj_gl["rule_level"] = d_obj_gl["rule_level"].astype(str).str.strip().str.lower()
        d_obj_gl["rule_type"] = d_obj_gl["rule_type"].astype(str).str.strip().str.lower()
        d_obj_gl = d_obj_gl[
            (d_obj_gl["rule_type"] == "obj") & d_obj_gl["rule_level"].isin(level_order)
        ].copy()
    else:
        d_obj_gl = d_obj_gl.iloc[0:0].copy()

    if len(d_obj_gl) > 0:
        grouped_bar(
            axes[1, 1],
            d_obj_gl,
            "rule_level",
            "log_rt",
            level_order,
            "Log-RT (Object Rule: Global vs Local)",
            "Log(RT)",
            sig_pairs=[("global", "local")],
            stat_meta=("figure1_behavior_rule_dimensions", "E"),
            bar_width=fig1_bar_width,
            jitter_width=fig1_jitter,
            box_aspect=fig1_box_aspect,
        )
        axes[1, 1].set_xticklabels(["Global", "Local"], rotation=15)
        grouped_bar(
            axes[1, 2],
            d_obj_gl,
            "rule_level",
            "correctness",
            level_order,
            "Accuracy (Object Rule: Global vs Local)",
            "Accuracy",
            sig_pairs=[("global", "local")],
            stat_meta=("figure1_behavior_rule_dimensions", "F"),
            bar_width=fig1_bar_width,
            jitter_width=fig1_jitter,
            box_aspect=fig1_box_aspect,
        )
        axes[1, 2].set_xticklabels(["Global", "Local"], rotation=15)
    else:
        axes[1, 1].text(0.5, 0.5, "object-rule global/local missing", ha="center", va="center", transform=axes[1, 1].transAxes)
        axes[1, 2].text(0.5, 0.5, "object-rule global/local missing", ha="center", va="center", transform=axes[1, 2].transAxes)

    _label_panels(axes, "ABCDEF")
    # Slightly increase spacing (~1.15x) while keeping compact 2x3 layout.
    axes[0, 1].set_ylabel("")
    axes[0, 2].set_ylabel("")
    axes[1, 1].set_ylabel("")
    axes[1, 2].set_ylabel("")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.08, top=0.955, wspace=0.07, hspace=0.10)
    # Full caption in LaTeX; avoid duplicating on PNG.
    _save_figure_compact(fig, "figure1_behavior_rule_dimensions.png")
    plt.close(fig)

    # Shift + position (Figure 2).
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), constrained_layout=True)
    d_block = df[["subj", "block_id", "shift_type", "search_len", "first_correct_latency"]].drop_duplicates()
    d_block = d_block[d_block["shift_type"].isin(["within_level", "cross_level"])]
    grouped_bar(
        axes[0, 0],
        d_block,
        "shift_type",
        "search_len",
        ["within_level", "cross_level"],
        "Search Length By Shift Type",
        "Search Length (Trials)",
        sig_pairs=[("within_level", "cross_level")],
        stat_meta=("figure2_behavior_shift_position", "A"),
    )
    grouped_bar(
        axes[0, 1],
        d_block,
        "shift_type",
        "first_correct_latency",
        ["within_level", "cross_level"],
        "First-Correct Latency By Shift Type",
        "First-Correct Latency (Trials)",
        sig_pairs=[("within_level", "cross_level")],
        stat_meta=("figure2_behavior_shift_position", "B"),
    )
    grouped_bar(
        axes[1, 0],
        df,
        "position_bin",
        "correctness",
        ["t1", "t2_3", "t4plus"],
        "Accuracy By Position Bin",
        "Accuracy",
        sig_pairs=[("t1", "t2_3"), ("t1", "t4plus")],
        stat_meta=("figure2_behavior_shift_position", "C"),
    )
    grouped_bar(
        axes[1, 1],
        df,
        "position_bin",
        "log_rt",
        ["t1", "t2_3", "t4plus"],
        "Log-RT By Position Bin",
        "Log(RT)",
        sig_pairs=[("t1", "t2_3"), ("t1", "t4plus")],
        stat_meta=("figure2_behavior_shift_position", "D"),
    )
    _bin_labels = ["1st trial", "2nd–3rd trial", "4th+ trial"]
    axes[1, 0].set_xticklabels(_bin_labels, rotation=15)
    axes[1, 1].set_xticklabels(_bin_labels, rotation=15)
    _label_panels(axes, "ABCD")
    _save_figure_compact(fig, "figure2_behavior_shift_position.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    grouped_bar(
        axes[0, 0],
        df,
        "phase3",
        "correctness",
        ["pre_first_correct", "first_correct_to_pre_stable", "stable"],
        "Accuracy By Phase3",
        "Accuracy",
        sig_pairs=[("pre_first_correct", "first_correct_to_pre_stable"), ("pre_first_correct", "stable")],
        stat_meta=("figure3_behavior_phase_collapse", "A"),
    )
    grouped_bar(
        axes[0, 1],
        df,
        "phase3",
        "log_rt",
        ["pre_first_correct", "first_correct_to_pre_stable", "stable"],
        "Log-RT By Phase3",
        "Log(RT)",
        sig_pairs=[("pre_first_correct", "first_correct_to_pre_stable"), ("first_correct_to_pre_stable", "stable")],
        stat_meta=("figure3_behavior_phase_collapse", "B"),
    )
    grouped_bar(
        axes[1, 0],
        df,
        "post_collapse",
        "correctness",
        [0, 1],
        "Accuracy By Post-Collapse",
        "Accuracy",
        sig_pairs=[(0, 1)],
        stat_meta=("figure3_behavior_phase_collapse", "C"),
    )
    grouped_bar(
        axes[1, 1],
        df,
        "post_collapse",
        "log_rt",
        [0, 1],
        "Log-RT By Post-Collapse",
        "Log(RT)",
        sig_pairs=[(0, 1)],
        stat_meta=("figure3_behavior_phase_collapse", "D"),
    )
    _label_panels(axes, "ABCD")
    _save_figure_compact(fig, "figure3_behavior_phase_collapse.png")
    plt.close(fig)


def build_eeg_merged() -> pd.DataFrame:
    eeg = pd.read_csv(EEG_TRIAL)
    eeg = to_num(eeg, ["block_id", "trial_id", "feedback_locked_FRN", "feedback_locked_theta_power", "feedback_locked_P3b"])
    hmm = pd.read_csv(HMM_TRIAL)
    hmm = to_num(hmm, ["block_id", "trial_id", "first_correct_flag", "post_first_correct_flag", "stable_flag", "search_flag"])
    beh = pd.read_csv(ALL_TRIAL)
    beh = to_num(beh, ["block_id", "trial_id", "correctness"])
    trans = pd.read_csv(TRANS_TRIAL)
    trans = to_num(trans, ["block_id", "trial_id", "time_since_collapse", "stable_flag"])

    m = eeg.merge(
        hmm[["subj", "block_id", "trial_id", "first_correct_flag", "post_first_correct_flag", "stable_flag", "search_flag"]],
        on=["subj", "block_id", "trial_id"],
        how="left",
    )
    m = m.merge(beh[["subj", "block_id", "trial_id", "correctness"]], on=["subj", "block_id", "trial_id"], how="left")
    m = m.merge(trans[["subj", "block_id", "trial_id", "time_since_collapse"]], on=["subj", "block_id", "trial_id"], how="left")
    m["phase3"] = np.where(
        m["stable_flag"] == 1,
        "stable",
        np.where((m["post_first_correct_flag"] == 1) | (m["first_correct_flag"] == 1), "first_correct_to_pre_stable", "pre_first_correct"),
    )
    m["search_feedback_type"] = np.where(
        (m["search_flag"] == 1) & (m["correctness"] == 0),
        "search_wrong",
        np.where(
            (m["first_correct_flag"] == 1) & (m["correctness"] == 1),
            "first_correct",
            np.where(
                (m["search_flag"] == 1) & (m["correctness"] == 1) & (m["first_correct_flag"] != 1),
                "search_correct_nonterminal",
                "other",
            ),
        ),
    )
    return m


def make_eeg_feedback_figure(m: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    d1 = m[(m["phase"] == "search") & m["correctness"].isin([0, 1])]
    grouped_bar(
        axes[0, 0],
        d1,
        "correctness",
        "feedback_locked_FRN",
        [0, 1],
        "Search FRN: Wrong vs Correct",
        "FRN (uV)",
        sig_pairs=[(0, 1)],
        stat_meta=("figure4_feedback_locked_eeg", "A"),
    )
    grouped_bar(
        axes[0, 1],
        d1,
        "correctness",
        "feedback_locked_P3b",
        [0, 1],
        "Search P3b: Wrong vs Correct",
        "P3b (uV)",
        sig_pairs=[(0, 1)],
        stat_meta=("figure4_feedback_locked_eeg", "B"),
    )
    grouped_bar(
        axes[0, 2],
        d1,
        "correctness",
        "feedback_locked_theta_power",
        [0, 1],
        "Search Theta: Wrong vs Correct",
        "Theta Power",
        sig_pairs=[(0, 1)],
        stat_meta=("figure4_feedback_locked_eeg", "C"),
    )

    d2 = m[m["search_feedback_type"].isin(["first_correct", "search_correct_nonterminal", "search_wrong"])]
    grouped_bar(
        axes[1, 0],
        d2,
        "search_feedback_type",
        "feedback_locked_P3b",
        ["first_correct", "search_correct_nonterminal", "search_wrong"],
        "P3b By Search Feedback Type",
        "P3b (uV)",
        sig_pairs=[("first_correct", "search_wrong"), ("first_correct", "search_correct_nonterminal")],
        stat_meta=("figure4_feedback_locked_eeg", "D"),
    )

    d5 = m[m["phase3"].isin(["pre_first_correct", "first_correct_to_pre_stable", "stable"])]
    grouped_bar(
        axes[1, 1],
        d5,
        "phase3",
        "feedback_locked_theta_power",
        ["pre_first_correct", "first_correct_to_pre_stable", "stable"],
        "Theta By Phase3",
        "Theta Power",
        sig_pairs=[("pre_first_correct", "stable")],
        stat_meta=("figure4_feedback_locked_eeg", "E"),
    )
    grouped_bar(
        axes[1, 2],
        d5,
        "phase3",
        "feedback_locked_P3b",
        ["pre_first_correct", "first_correct_to_pre_stable", "stable"],
        "P3b By Phase3",
        "P3b (uV)",
        sig_pairs=[("pre_first_correct", "first_correct_to_pre_stable"), ("pre_first_correct", "stable")],
        stat_meta=("figure4_feedback_locked_eeg", "F"),
    )
    _label_panels(axes, "ABCDEF")
    _save_figure_compact(fig, "figure4_feedback_locked_eeg.png")
    plt.close(fig)


def make_priority_correlation_figure() -> None:
    data = pd.read_csv(PRIORITY_DATA)
    stats = pd.read_csv(PRIORITY_STATS)
    sig = stats[pd.to_numeric(stats["p_spearman_fdr_within_line"], errors="coerce") <= 0.1].copy()
    sig = sig.reset_index(drop=True)
    if sig.empty:
        return

    n = len(sig)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for i, row in sig.iterrows():
        xcol = row["eeg_var"]
        ycol = row["beh_var"]
        d = data[[xcol, ycol]].dropna()
        x = d[xcol].to_numpy(dtype=float)
        y = d[ycol].to_numpy(dtype=float)
        rho, p = spearmanr(x, y, nan_policy="omit")
        scatter_with_fit(axes[i], x, y, f"{xcol} vs {ycol}", xcol, ycol)
        panel = string.ascii_uppercase[i]
        pr = _p_report_with_threshold(float(p))
        axes[i].text(
            0.04,
            0.96,
            f"rho={rho:.3f}\n{pr}\nq(FDR)={float(row['p_spearman_fdr_within_line']):.4g}",
            transform=axes[i].transAxes,
            va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.8"),
            fontsize=11,
        )
        _append_inference_row(
            {
                "figure_file": "figure5_priority_correlations_q10.png",
                "panel": panel,
                "subplot_title": f"{xcol} vs {ycol}",
                "test": "spearman_correlation",
                "predictor": str(xcol),
                "outcome": str(ycol),
                "level_a": "",
                "level_b": "",
                "n_pairs": int(len(d)),
                "t_statistic": np.nan,
                "p_value": float(p),
                "mean_diff_a_minus_b": np.nan,
                "note": f"rho={rho:.6g}; q_fdr_within_line={float(row['p_spearman_fdr_within_line']):.6g}",
            }
        )
    _label_panels(axes, string.ascii_uppercase[:n])
    _save_figure_compact(fig, "figure5_priority_correlations_q10.png")
    plt.close(fig)


def make_cue_optional_figure() -> None:
    if not os.path.exists(EEG_CUE):
        return
    cue = pd.read_csv(EEG_CUE)
    cue = to_num(cue, ["frontal_theta_cue", "posterior_alpha_cue", "P3a_cue"])
    d3 = cue[cue["shift_type"].isin(["within_level", "cross_level"])].copy()
    if d3.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    grouped_bar(
        axes[0],
        d3,
        "shift_type",
        "frontal_theta_cue",
        ["cross_level", "within_level"],
        "Cue Theta By Shift Type",
        "Theta",
        sig_pairs=[("cross_level", "within_level")],
        stat_meta=("figure6_optional_cue_locked_shift", "A"),
    )
    grouped_bar(
        axes[1],
        d3,
        "shift_type",
        "posterior_alpha_cue",
        ["cross_level", "within_level"],
        "Cue Posterior Alpha By Shift Type",
        "Alpha",
        sig_pairs=[("cross_level", "within_level")],
        stat_meta=("figure6_optional_cue_locked_shift", "B"),
    )
    grouped_bar(
        axes[2],
        d3,
        "shift_type",
        "P3a_cue",
        ["cross_level", "within_level"],
        "Cue P3a By Shift Type",
        "P3a (uV)",
        sig_pairs=[("cross_level", "within_level")],
        stat_meta=("figure6_optional_cue_locked_shift", "C"),
    )
    _label_panels(axes, "ABC")
    _save_figure_compact(fig, "figure6_optional_cue_locked_shift.png")
    plt.close(fig)


def copy_isrsa_triptych() -> None:
    if os.path.exists(ISRSA_TRIPTYCH):
        shutil.copy2(ISRSA_TRIPTYCH, os.path.join(OUT_DIR, "figure7_isrsa_triptych_drop_mixture_1s.png"))


def _mean_sem(arr: np.ndarray):
    m = np.nanmean(arr, axis=0)
    se = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(max(arr.shape[0], 1))
    return m, se


def _extract_waveforms_for_contrasts(meta: pd.DataFrame):
    out = {
        "wrong_vs_correct": {"wrong": {}, "correct": {}},
        "first_correct_vs_search_wrong": {"first_correct": {}, "search_wrong": {}},
        "early_vs_late_found": {"early_found": {}, "late_found": {}},
        "early_vs_late_by_set": {
            "sets1_4": {"early_found": {}, "late_found": {}},
            "sets5_8": {"early_found": {}, "late_found": {}},
        },
        "phase3": {"pre_first_correct": {}, "first_correct_to_pre_stable": {}, "stable": {}},
    }
    maps = {
        "wrong_vs_correct": {"wrong": {}, "correct": {}},
        "first_correct_vs_search_wrong": {"first_correct": {}, "search_wrong": {}},
        "early_vs_late_found": {"early_found": {}, "late_found": {}},
        "early_vs_late_by_set": {
            "sets1_4": {"early_found": {}, "late_found": {}},
            "sets5_8": {"early_found": {}, "late_found": {}},
        },
        "phase3": {"pre_first_correct": {}, "first_correct_to_pre_stable": {}, "stable": {}},
    }
    info_eeg = None
    times_ref = None

    for subj, g in meta.groupby("subj"):
        set_path = os.path.join(EEG_DIR, f"{subj}_reknow_wcst.set")
        if not os.path.exists(set_path):
            continue
        try:
            raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
            ev, eid = mne.events_from_annotations(raw, event_id={FEEDBACK_CODE: 1}, verbose="ERROR")
            if len(ev) == 0:
                continue
            epochs = mne.Epochs(
                raw,
                ev,
                event_id=eid,
                tmin=-0.2,
                tmax=0.8,
                baseline=(-0.2, 0.0),
                preload=True,
                reject_by_annotation=False,
                verbose="ERROR",
            )
            ch = list(epochs.ch_names)
            idx_front = [ch.index(c) for c in ROI_FRONT if c in ch]
            if len(idx_front) == 0:
                continue
            picks_eeg = mne.pick_types(epochs.info, eeg=True, exclude=[])
            if len(picks_eeg) == 0:
                continue
            gg = g.copy()
            gg["feedback_event_num"] = pd.to_numeric(gg.get("feedback_event_num"), errors="coerce")
            gg = gg.dropna(subset=["feedback_event_num"])
            gg["ep_idx"] = gg["feedback_event_num"].astype(int) - 1
            gg = gg[(gg["ep_idx"] >= 0) & (gg["ep_idx"] < len(epochs))]
            if gg.empty:
                continue
            # subject-wise time-ordered 8-set assignment based on block order
            gg["block_num"] = pd.to_numeric(gg.get("block_id"), errors="coerce")
            ub = gg[["block_id", "block_num"]].dropna().drop_duplicates().sort_values("block_num")
            block_ids = ub["block_id"].astype(str).to_numpy()
            block_to_set = {}
            if len(block_ids):
                for si, ch in enumerate(np.array_split(block_ids, 8), start=1):
                    for bid in ch:
                        block_to_set[str(bid)] = int(si)
            gg["block_id_str"] = gg["block_id"].astype(str)
            gg["set_index"] = gg["block_id_str"].map(block_to_set)
            gg["set_group"] = np.where(pd.to_numeric(gg["set_index"], errors="coerce") <= 4, "sets1_4", "sets5_8")

            X_roi = epochs.get_data(copy=True)[:, idx_front, :] * 1e6
            X_roi = np.nanmean(X_roi, axis=1)  # n_ep x n_time
            X_eeg = epochs.get_data(copy=True)[:, picks_eeg, :] * 1e6  # n_ep x n_ch x n_time
            times = epochs.times.copy()
            times_ref = times
            if info_eeg is None:
                info_eeg = mne.pick_info(epochs.info, picks_eeg, copy=True)
            else:
                # ensure channel set stable across subjects
                if list(info_eeg["ch_names"]) != [epochs.ch_names[i] for i in picks_eeg]:
                    continue

            # wrong vs correct in search
            w = gg[(gg["phase"] == "search") & (gg["correctness"] == 0)]["ep_idx"].to_numpy(dtype=int)
            c = gg[(gg["phase"] == "search") & (gg["correctness"] == 1)]["ep_idx"].to_numpy(dtype=int)
            if len(w) >= 5 and len(c) >= 5:
                out["wrong_vs_correct"]["wrong"][subj] = np.nanmean(X_roi[w, :], axis=0)
                out["wrong_vs_correct"]["correct"][subj] = np.nanmean(X_roi[c, :], axis=0)
                maps["wrong_vs_correct"]["wrong"][subj] = np.nanmean(X_eeg[w, :, :], axis=0)
                maps["wrong_vs_correct"]["correct"][subj] = np.nanmean(X_eeg[c, :, :], axis=0)

            # first_correct vs search_wrong
            f = gg[gg["search_feedback_type"] == "first_correct"]["ep_idx"].to_numpy(dtype=int)
            sw = gg[gg["search_feedback_type"] == "search_wrong"]["ep_idx"].to_numpy(dtype=int)
            if len(f) >= 3 and len(sw) >= 5:
                out["first_correct_vs_search_wrong"]["first_correct"][subj] = np.nanmean(X_roi[f, :], axis=0)
                out["first_correct_vs_search_wrong"]["search_wrong"][subj] = np.nanmean(X_roi[sw, :], axis=0)
                maps["first_correct_vs_search_wrong"]["first_correct"][subj] = np.nanmean(X_eeg[f, :, :], axis=0)
                maps["first_correct_vs_search_wrong"]["search_wrong"][subj] = np.nanmean(X_eeg[sw, :, :], axis=0)

            # HMM Viterbi: search vs acquired (dict keys kept for downstream figure wiring)
            sidx = gg[gg["phase"] == "search"]["ep_idx"].to_numpy(dtype=int)
            aidx = gg[gg["phase"] == "acquired"]["ep_idx"].to_numpy(dtype=int)
            if len(sidx) >= 5 and len(aidx) >= 5:
                out["early_vs_late_found"]["early_found"][subj] = np.nanmean(X_roi[sidx, :], axis=0)
                out["early_vs_late_found"]["late_found"][subj] = np.nanmean(X_roi[aidx, :], axis=0)
                maps["early_vs_late_found"]["early_found"][subj] = np.nanmean(X_eeg[sidx, :, :], axis=0)
                maps["early_vs_late_found"]["late_found"][subj] = np.nanmean(X_eeg[aidx, :, :], axis=0)
            for sg in ["sets1_4", "sets5_8"]:
                gs = gg[(gg["set_group"] == sg) & (gg["phase"] == "search")]["ep_idx"].to_numpy(dtype=int)
                ga = gg[(gg["set_group"] == sg) & (gg["phase"] == "acquired")]["ep_idx"].to_numpy(dtype=int)
                if len(gs) >= 5 and len(ga) >= 5:
                    out["early_vs_late_by_set"][sg]["early_found"][subj] = np.nanmean(X_roi[gs, :], axis=0)
                    out["early_vs_late_by_set"][sg]["late_found"][subj] = np.nanmean(X_roi[ga, :], axis=0)
                    maps["early_vs_late_by_set"][sg]["early_found"][subj] = np.nanmean(X_eeg[gs, :, :], axis=0)
                    maps["early_vs_late_by_set"][sg]["late_found"][subj] = np.nanmean(X_eeg[ga, :, :], axis=0)

            # phase3
            for ph in ["pre_first_correct", "first_correct_to_pre_stable", "stable"]:
                idx = gg[gg["phase3"] == ph]["ep_idx"].to_numpy(dtype=int)
                if len(idx) >= 5:
                    out["phase3"][ph][subj] = np.nanmean(X_roi[idx, :], axis=0)
                    maps["phase3"][ph][subj] = np.nanmean(X_eeg[idx, :, :], axis=0)
        except Exception:
            continue

    return out, maps, info_eeg, times_ref


def _plot_wave(ax, times, arr_map, label, color):
    if len(arr_map) == 0:
        return
    arr = np.vstack(list(arr_map.values()))
    m, se = _mean_sem(arr)
    ax.plot(times, m, label=f"{label} (n={arr.shape[0]})", color=color, linewidth=2)
    ax.fill_between(times, m - se, m + se, color=color, alpha=0.2)


def _paired_window_ttest(arr_map_a: dict, arr_map_b: dict, times: np.ndarray, win: tuple) -> Optional[dict]:
    shared = sorted(set(arr_map_a.keys()).intersection(arr_map_b.keys()))
    if len(shared) < 6:
        return None
    m = (times >= win[0]) & (times <= win[1])
    if not np.any(m):
        return None
    a = np.array([np.nanmean(arr_map_a[s][m]) for s in shared], dtype=float)
    b = np.array([np.nanmean(arr_map_b[s][m]) for s in shared], dtype=float)
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 6:
        return None
    res = ttest_rel(a[good], b[good], nan_policy="omit")
    return {
        "n_pairs": int(good.sum()),
        "t_statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "mean_diff_a_minus_b": float(np.mean(a[good] - b[good])),
    }


def _paired_window_p(arr_map_a: dict, arr_map_b: dict, times: np.ndarray, win: tuple) -> float:
    det = _paired_window_ttest(arr_map_a, arr_map_b, times, win)
    if det is None:
        return np.nan
    return float(det["p_value"])


def _annotate_window_sig(
    ax,
    times: np.ndarray,
    arr_map_a: dict,
    arr_map_b: dict,
    win: tuple,
    y: float,
    tag: str,
    color: str,
    stat_base: Optional[dict] = None,
):
    det = _paired_window_ttest(arr_map_a, arr_map_b, times, win)
    p = float(det["p_value"]) if det else np.nan
    star = _p_to_star(p)
    x1, x2 = win
    ax.axvspan(x1, x2, color=color, alpha=0.18, zorder=0)
    if np.isfinite(p):
        txt = f"{tag}: {star}\n{_p_report_with_threshold(p)}"
    else:
        txt = f"{tag}: ns"
    ax.text(
        (x1 + x2) / 2,
        y,
        txt,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#7A1F1F",
        linespacing=1.05,
    )
    if stat_base is not None and det is not None:
        _append_inference_row(
            {
                "figure_file": stat_base["figure_file"],
                "panel": stat_base["panel"],
                "subplot_title": stat_base.get("subplot_title", tag),
                "test": stat_base.get("test", "paired_ttest_rel_subject_window_mean"),
                "predictor": stat_base.get("predictor", ""),
                "outcome": stat_base.get("outcome", "mean_uV_front_ROI"),
                "level_a": stat_base.get("level_a", ""),
                "level_b": stat_base.get("level_b", ""),
                "n_pairs": det["n_pairs"],
                "t_statistic": det["t_statistic"],
                "p_value": det["p_value"],
                "mean_diff_a_minus_b": det["mean_diff_a_minus_b"],
                "note": stat_base.get("note", f"time_window_s={win[0]}-{win[1]}"),
            }
        )


def _paired_topomap_diff(map_a: dict, map_b: dict, times: np.ndarray, win: tuple) -> np.ndarray:
    shared = sorted(set(map_a.keys()).intersection(map_b.keys()))
    if len(shared) == 0:
        return np.array([])
    m = (times >= win[0]) & (times <= win[1])
    if not np.any(m):
        return np.array([])
    diffs = []
    for s in shared:
        a = np.nanmean(map_a[s][:, m], axis=1)
        b = np.nanmean(map_b[s][:, m], axis=1)
        diffs.append(a - b)
    return np.nanmean(np.vstack(diffs), axis=0)


def _plot_topomap_with_roi(ax, vec: np.ndarray, info_eeg, title: str):
    if vec.size == 0 or info_eeg is None:
        ax.set_axis_off()
        return
    ch_names = list(info_eeg["ch_names"])
    pos = np.array([info_eeg["chs"][i]["loc"][:2] for i in range(len(ch_names))], dtype=float)
    good = np.isfinite(pos).all(axis=1) & np.isfinite(vec)
    if good.sum() < 4:
        ax.set_axis_off()
        return
    pos = pos[good]
    val = vec[good]
    names = [ch_names[i] for i in np.where(good)[0]]
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=val, cmap="RdBu_r", s=170, edgecolor="black", linewidth=0.5)
    # Highlight ROI sensors
    roi_idx = [i for i, n in enumerate(names) if n in ROI_FRONT]
    if roi_idx:
        ax.scatter(
            pos[roi_idx, 0],
            pos[roi_idx, 1],
            s=260,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
        )
    # Draw a head-like circle
    r = np.nanmax(np.sqrt(np.sum(pos**2, axis=1))) * 1.08
    circ = plt.Circle((0, 0), r, edgecolor="gray", facecolor="none", linewidth=1.0)
    ax.add_patch(circ)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-r * 1.05, r * 1.05)
    ax.set_ylim(-r * 1.05, r * 1.05)
    if SUBPLOT_TITLES:
        ax.set_title(title, fontsize=12)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)


def _make_wave_panel(ax, times, a_map, b_map, a_label, b_label, specs, title, stat_context: Optional[dict] = None):
    _plot_wave(ax, times, a_map, a_label, "tab:blue")
    _plot_wave(ax, times, b_map, b_label, "tab:red")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    y0, y1 = ax.get_ylim()
    yr = y1 - y0 if y1 > y0 else 1.0
    for i, (tag, win, color) in enumerate(specs):
        base = None
        if stat_context:
            base = {
                **stat_context,
                "subplot_title": title,
                "predictor": tag,
                "level_a": a_label,
                "level_b": b_label,
            }
        _annotate_window_sig(ax, times, a_map, b_map, win, y1 - (0.10 + i * 0.12) * yr, tag, color, stat_base=base)
    if SUBPLOT_TITLES:
        ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (uV)")
    ax.legend(frameon=False, fontsize=10)


def make_feedback_waveform_figures(meta: pd.DataFrame) -> None:
    waves, maps, info_eeg, times = _extract_waveforms_for_contrasts(meta)
    if times is None or info_eeg is None:
        return

    # Figure 8
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    _make_wave_panel(
        axes[0],
        times,
        waves["wrong_vs_correct"]["wrong"],
        waves["wrong_vs_correct"]["correct"],
        "search wrong",
        "search correct",
        [("FRN (0.20-0.35s)", (0.20, 0.35), "#BFD7EA"), ("P3b (0.30-0.50s)", (0.30, 0.50), "#F8D7DA")],
        "Search Wrong vs Correct (Front ROI)",
        stat_context={"figure_file": "figure8_waveform_search_wrong_vs_correct.png", "panel": "A"},
    )
    topo = _paired_topomap_diff(
        maps["wrong_vs_correct"]["wrong"], maps["wrong_vs_correct"]["correct"], times, (0.20, 0.35)
    )
    _plot_topomap_with_roi(axes[1], topo, info_eeg, "Topomap: Wrong - Correct\n(FRN window)")
    _label_panels(axes, "AB")
    _save_figure_compact(fig, "figure8_waveform_search_wrong_vs_correct.png")
    plt.close(fig)

    # Figure 9
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    _make_wave_panel(
        axes[0],
        times,
        waves["first_correct_vs_search_wrong"]["first_correct"],
        waves["first_correct_vs_search_wrong"]["search_wrong"],
        "first_correct",
        "search_wrong",
        [("P3b (0.30-0.50s)", (0.30, 0.50), "#F8D7DA")],
        "First-Correct vs Search-Wrong (Front ROI)",
        stat_context={"figure_file": "figure9_waveform_first_correct_vs_search_wrong.png", "panel": "A"},
    )
    topo = _paired_topomap_diff(
        maps["first_correct_vs_search_wrong"]["first_correct"],
        maps["first_correct_vs_search_wrong"]["search_wrong"],
        times,
        (0.30, 0.50),
    )
    _plot_topomap_with_roi(axes[1], topo, info_eeg, "Topomap: FirstCorrect - SearchWrong\n(P3b window)")
    _label_panels(axes, "AB")
    _save_figure_compact(fig, "figure9_waveform_first_correct_vs_search_wrong.png")
    plt.close(fig)

    # Merged Figure (old Figure 15 + Figure 16): one vertical waveform-only figure
    merged_file = "figure15_16_waveform_early_late_merged_vertical.png"
    fig, axes = plt.subplots(3, 1, figsize=(9, 12), constrained_layout=True)
    _make_wave_panel(
        axes[0],
        times,
        waves["early_vs_late_found"]["early_found"],
        waves["early_vs_late_found"]["late_found"],
        "search",
        "acquired",
        [("P3b (0.30-0.50s)", (0.30, 0.50), "#F8D7DA")],
        "Search vs Acquired (HMM, Front ROI)",
        stat_context={"figure_file": merged_file, "panel": "A"},
    )
    _make_wave_panel(
        axes[1],
        times,
        waves["early_vs_late_by_set"]["sets1_4"]["early_found"],
        waves["early_vs_late_by_set"]["sets1_4"]["late_found"],
        "search",
        "acquired",
        [("P3b (0.30-0.50s)", (0.30, 0.50), "#F8D7DA")],
        "Search vs Acquired (Sets 1-4, Front ROI)",
        stat_context={"figure_file": merged_file, "panel": "B"},
    )
    _make_wave_panel(
        axes[2],
        times,
        waves["early_vs_late_by_set"]["sets5_8"]["early_found"],
        waves["early_vs_late_by_set"]["sets5_8"]["late_found"],
        "search",
        "acquired",
        [("P3b (0.30-0.50s)", (0.30, 0.50), "#F8D7DA")],
        "Search vs Acquired (Sets 5-8, Front ROI)",
        stat_context={"figure_file": merged_file, "panel": "C"},
    )
    _label_panels(axes, "ABC")
    _save_figure_compact(fig, merged_file)
    plt.close(fig)

    # Figure 10
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    _plot_wave(axes[0], times, waves["phase3"]["pre_first_correct"], "pre_first_correct", "tab:orange")
    _plot_wave(
        axes[0], times, waves["phase3"]["first_correct_to_pre_stable"], "first_correct_to_pre_stable", "tab:purple"
    )
    _plot_wave(axes[0], times, waves["phase3"]["stable"], "stable", "tab:blue")
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    # color windows + pairwise p text
    axes[0].axvspan(0.30, 0.50, color="#F8D7DA", alpha=0.2, zorder=0)
    win_phase = (0.30, 0.50)
    det1 = _paired_window_ttest(
        waves["phase3"]["pre_first_correct"], waves["phase3"]["first_correct_to_pre_stable"], times, win_phase
    )
    det2 = _paired_window_ttest(waves["phase3"]["pre_first_correct"], waves["phase3"]["stable"], times, win_phase)
    p1 = float(det1["p_value"]) if det1 else np.nan
    p2 = float(det2["p_value"]) if det2 else np.nan
    axes[0].text(
        0.05,
        0.94,
        f"pre vs mid: {_p_to_star(p1)}\n{_p_report_with_threshold(p1)}",
        transform=axes[0].transAxes,
        va="top",
        fontsize=10,
        color="#7A1F1F",
        linespacing=1.05,
    )
    axes[0].text(
        0.05,
        0.78,
        f"pre vs stable: {_p_to_star(p2)}\n{_p_report_with_threshold(p2)}",
        transform=axes[0].transAxes,
        va="top",
        fontsize=10,
        color="#7A1F1F",
        linespacing=1.05,
    )
    if det1:
        _append_inference_row(
            {
                "figure_file": "figure10_waveform_phase3.png",
                "panel": "A",
                "subplot_title": "Phase3 waveforms P3b window",
                "test": "paired_ttest_rel_subject_window_mean",
                "predictor": "P3b 0.30–0.50s",
                "outcome": "mean_uV_front_ROI",
                "level_a": "pre_first_correct",
                "level_b": "first_correct_to_pre_stable",
                "n_pairs": det1["n_pairs"],
                "t_statistic": det1["t_statistic"],
                "p_value": det1["p_value"],
                "mean_diff_a_minus_b": det1["mean_diff_a_minus_b"],
                "note": "pre vs mid (phase3)",
            }
        )
    if det2:
        _append_inference_row(
            {
                "figure_file": "figure10_waveform_phase3.png",
                "panel": "A",
                "subplot_title": "Phase3 waveforms P3b window",
                "test": "paired_ttest_rel_subject_window_mean",
                "predictor": "P3b 0.30–0.50s",
                "outcome": "mean_uV_front_ROI",
                "level_a": "pre_first_correct",
                "level_b": "stable",
                "n_pairs": det2["n_pairs"],
                "t_statistic": det2["t_statistic"],
                "p_value": det2["p_value"],
                "mean_diff_a_minus_b": det2["mean_diff_a_minus_b"],
                "note": "pre vs stable (phase3)",
            }
        )
    if SUBPLOT_TITLES:
        axes[0].set_title("Phase3 waveforms (Front ROI)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (uV)")
    axes[0].legend(frameon=False, fontsize=10)
    topo = _paired_topomap_diff(maps["phase3"]["pre_first_correct"], maps["phase3"]["stable"], times, (0.30, 0.50))
    _plot_topomap_with_roi(axes[1], topo, info_eeg, "Topomap: Pre - Stable\n(P3b window)")
    _label_panels(axes, "AB")
    _save_figure_compact(fig, "figure10_waveform_phase3.png")
    plt.close(fig)


def make_behavior_supplement_figures(df: pd.DataFrame) -> None:
    """Extra panels for Results 3.1: shift accuracy/RT, four-way shift, turning point, feedback, collapse bins."""
    # --- Figure 11: shift type (trial) + four-way block metrics ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    d_tri = df[df["shift_type"].isin(["within_level", "cross_level"])].copy()
    grouped_bar(
        axes[0, 0],
        d_tri,
        "shift_type",
        "correctness",
        ["within_level", "cross_level"],
        "Accuracy By Shift Type (Trial Level)",
        "Accuracy",
        sig_pairs=[("within_level", "cross_level")],
        stat_meta=("figure11_behavior_shift_extended", "A"),
    )
    axes[0, 0].set_xticklabels(["Within level", "Cross level"], rotation=15)
    grouped_bar(
        axes[0, 1],
        d_tri,
        "shift_type",
        "log_rt",
        ["within_level", "cross_level"],
        "Log-RT By Shift Type (Trial Level)",
        "Log(RT)",
        sig_pairs=[("within_level", "cross_level")],
        stat_meta=("figure11_behavior_shift_extended", "B"),
    )
    axes[0, 1].set_xticklabels(["Within level", "Cross level"], rotation=15)

    d4 = df[["subj", "block_id", "shift_detail", "search_len", "first_correct_latency"]].drop_duplicates()
    d4 = d4[d4["shift_detail"].isin(["L-L", "L-G", "G-L", "G-G"])].copy()
    order4 = ["L-L", "G-G", "G-L", "L-G"]
    grouped_bar(
        axes[1, 0],
        d4,
        "shift_detail",
        "search_len",
        order4,
        "Search Length By Shift Direction (Block)",
        "Search Length (Trials)",
        sig_pairs=[("L-L", "L-G")],
        stat_meta=("figure11_behavior_shift_extended", "C"),
    )
    axes[1, 0].set_xticklabels(["L→L", "G→G", "G→L", "L→G"], rotation=15)
    grouped_bar(
        axes[1, 1],
        d4,
        "shift_detail",
        "first_correct_latency",
        order4,
        "First-Correct Latency By Shift Direction",
        "First-Correct Latency (Trials)",
        sig_pairs=[("L-L", "L-G")],
        stat_meta=("figure11_behavior_shift_extended", "D"),
    )
    axes[1, 1].set_xticklabels(["L→L", "G→G", "G→L", "L→G"], rotation=15)
    _label_panels(axes, "ABCD")
    _save_figure_compact(fig, "figure11_behavior_shift_extended.png")
    plt.close(fig)

    # --- Figure 12: search turning point (P correct next trial) ---
    ms = df[(df["search_flag"] == 1) & df["next_correct"].notna()].copy()
    if len(ms) > 0:

        def _sfcat(row) -> str:
            if pd.to_numeric(row["first_correct_flag"], errors="coerce") == 1:
                return "first_correct"
            if pd.to_numeric(row["correctness"], errors="coerce") == 1:
                return "search_correct_other"
            return "search_wrong"

        ms["search_outcome_cat"] = ms.apply(_sfcat, axis=1)
        sub = ms.groupby(["subj", "search_outcome_cat"], as_index=False)["next_correct"].mean()
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5), constrained_layout=True)
        grouped_bar(
            ax,
            sub,
            "search_outcome_cat",
            "next_correct",
            ["first_correct", "search_correct_other", "search_wrong"],
            "P(Correct On Next Trial | Search Phase)",
            "P(next correct)",
            sig_pairs=[("first_correct", "search_correct_other"), ("first_correct", "search_wrong")],
            stat_meta=("figure12_behavior_search_turning_point", "A"),
        )
        ax.set_xticklabels(["First correct", "Search correct\n(other)", "Search wrong"], rotation=12)
        _panel_label(ax, "A")
        _save_figure_compact(fig, "figure12_behavior_search_turning_point.png")
        plt.close(fig)

    # --- Figure 13: feedback history + post-collapse time ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fh = df[df["prev_correct_bin"].notna()].copy()
    fh["prev_ok"] = fh["prev_correct_bin"].astype(int)
    grouped_bar(
        axes[0, 0],
        fh,
        "prev_ok",
        "log_rt",
        [0, 1],
        "Log-RT By Previous-Trial Feedback",
        "Log(RT)",
        sig_pairs=[(0, 1)],
        stat_meta=("figure13_behavior_feedback_and_collapse_time", "A"),
    )
    axes[0, 0].set_xticklabels(["Prev incorrect", "Prev correct"], rotation=15)
    grouped_bar(
        axes[0, 1],
        fh,
        "prev_ok",
        "correctness",
        [0, 1],
        "Accuracy By Previous-Trial Feedback",
        "Accuracy",
        sig_pairs=[(0, 1)],
        stat_meta=("figure13_behavior_feedback_and_collapse_time", "B"),
    )
    axes[0, 1].set_xticklabels(["Prev incorrect", "Prev correct"], rotation=15)

    cbdf = df[pd.notna(df["collapse_bin"])].copy()
    cbdf["collapse_bin"] = cbdf["collapse_bin"].astype(str)
    order_c = ["0", "1", "2", "3+"]
    cbdf = cbdf[cbdf["collapse_bin"].isin(order_c)]
    if len(cbdf) > 0:
        grouped_bar(
            axes[1, 0],
            cbdf,
            "collapse_bin",
            "log_rt",
            order_c,
            "Log-RT By Time Since Logical Collapse",
            "Log(RT)",
            sig_pairs=None,
        )
        grouped_bar(
            axes[1, 1],
            cbdf,
            "collapse_bin",
            "correctness",
            order_c,
            "Accuracy By Time Since Logical Collapse",
            "Accuracy",
            sig_pairs=None,
        )
    else:
        for ax in (axes[1, 0], axes[1, 1]):
            ax.text(0.5, 0.5, "No post-collapse bins", ha="center", va="center", transform=ax.transAxes)

    _label_panels(axes, "ABCD")
    _save_figure_compact(fig, "figure13_behavior_feedback_and_collapse_time.png")
    plt.close(fig)


def _dot_bar_zero_line(ax, values: np.ndarray, title: str, ylabel: str) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 0.05, size=len(values))
    ax.scatter(x, values, s=28, alpha=0.55, marker="o", c="black", edgecolors="black", linewidths=0.35, zorder=3)
    mu = float(np.nanmean(values))
    se = float(np.nanstd(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
    ax.plot([-0.15, 0.15], [mu, mu], color="#4A6FA8", linewidth=2.5)
    if np.isfinite(se):
        ax.fill_between([-0.15, 0.15], mu - se, mu + se, color="#4A6FA8", alpha=0.25)
    ax.set_xticks([0])
    ax.set_xticklabels([""])
    ax.set_xlim(-0.25, 0.25)
    if SUBPLOT_TITLES:
        ax.set_title(title)
    ax.set_ylabel(ylabel)


def make_hmm_behavior_figure() -> None:
    """Results 3.2: CV summary + set-window contrast + block strategy mix."""
    if not os.path.isfile(HMM_CV_SUBJ) or not os.path.isfile(HMM_CV_FOLD):
        print("[WARN] HMM CSVs missing; skip figure14.")
        return
    if not os.path.isfile(SOFT_BLOCK):
        print("[WARN] soft block labels missing; skip panel D.")
        return
    if not os.path.isfile(HMM_SET_BY_SET):
        print("[WARN] hmm_set_by_set_fullonly.csv missing; skip set-window panel in figure14.")
        return

    subj_sum = pd.read_csv(HMM_CV_SUBJ)
    fold_df = pd.read_csv(HMM_CV_FOLD)
    soft = pd.read_csv(SOFT_BLOCK)
    set_df = pd.read_csv(HMM_SET_BY_SET)

    skip = {"reknow011", "reknow020", "reknow023"}
    subj_sum = subj_sum[~subj_sum["subj"].isin(skip)].copy()
    fold_df = fold_df[~fold_df["subj"].isin(skip)].copy()
    soft = soft[~soft["subj"].isin(skip)].copy()
    set_df = set_df[~set_df["subj"].isin(skip)].copy()

    col = "gradual_minus_one_shot_per_trial_mean"
    if col not in subj_sum.columns:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), constrained_layout=True)

    vals = pd.to_numeric(subj_sum[col], errors="coerce").to_numpy(dtype=float)
    _dot_bar_zero_line(
        axes[0, 0],
        vals,
        "Gradual Minus One-Shot (LL/Trial, CV)",
        "Δ log-likelihood / trial",
    )

    rows = []
    for subj, g in set_df.groupby("subj"):
        si = pd.to_numeric(g["set_index"], errors="coerce")
        val = pd.to_numeric(g["ll_diff_per_trial"], errors="coerce")
        early = val[si.isin([1.0, 2.0])].mean()
        late = val[si.isin([7.0, 8.0])].mean()
        if np.isfinite(early) and np.isfinite(late):
            rows.append({"subj": subj, "window": "sets1_2", "val": float(early)})
            rows.append({"subj": subj, "window": "sets7_8", "val": float(late)})
    win_df = pd.DataFrame(rows)
    if len(win_df) > 0:
        grouped_bar(
            axes[0, 1],
            win_df,
            "window",
            "val",
            ["sets1_2", "sets7_8"],
            "Gradual Advantage: Sets 1-2 Vs Sets 7-8",
            "Δ LL / trial (gradual − one-shot)",
            sig_pairs=[("sets1_2", "sets7_8")],
            stat_meta=("figure14_hmm_model_comparison", "B"),
        )
        axes[0, 1].set_xticklabels(["Sets 1-2", "Sets 7-8"], rotation=12)
    else:
        axes[0, 1].set_axis_off()

    frac_g = (
        soft.assign(is_g=(soft["strategy_hard"].astype(str) == "gradual_like").astype(float))
        .groupby("subj", as_index=False)["is_g"]
        .mean()
        .rename(columns={"is_g": "frac_gradual_blocks"})
    )
    _dot_bar_zero_line(
        axes[1, 0],
        pd.to_numeric(frac_g["frac_gradual_blocks"], errors="coerce").to_numpy(float),
        "Fraction Of Gradual-Like Blocks (Per Subject)",
        "Fraction",
    )
    axes[1, 0].set_ylim(-0.05, 1.05)

    mix = (
        soft.groupby(["subj", "strategy_hard"], as_index=False)
        .size()
        .pivot(index="subj", columns="strategy_hard", values="size")
        .fillna(0)
    )
    if len(mix) > 0 and "gradual_like" in mix.columns and "one_shot_like" in mix.columns:
        tot = mix[["gradual_like", "one_shot_like"]].sum(axis=1).replace(0, np.nan)
        ratio = (mix["gradual_like"] / tot).astype(float)
        axes[1, 1].scatter(
            np.arange(len(ratio)),
            ratio.to_numpy(),
            s=30,
            c="black",
            alpha=0.65,
            edgecolors="black",
            linewidths=0.35,
        )
        axes[1, 1].set_xticks(np.arange(len(ratio)))
        axes[1, 1].set_xticklabels(mix.index.astype(str), rotation=90, fontsize=8)
        axes[1, 1].set_ylim(-0.05, 1.05)
        axes[1, 1].set_ylabel("Frac. gradual-like blocks")
        if SUBPLOT_TITLES:
            axes[1, 1].set_title("Per-Subject Gradual-Like Block Proportion")
    else:
        axes[1, 1].set_axis_off()

    _label_panels(axes, "ABCD")
    _save_figure_compact(fig, "figure14_hmm_model_comparison.png")
    plt.close(fig)


def make_hmm_figure_abc_set_windows() -> None:
    """HMM Figure ABC: set-window contrasts (Sets 1-2 vs Sets 7-8)."""
    if not os.path.isfile(HMM_SET_BY_SET):
        print("[WARN] hmm_set_by_set_fullonly.csv missing; skip HMM Figure ABC.")
        return

    df = pd.read_csv(HMM_SET_BY_SET)
    skip = {"reknow011", "reknow020", "reknow023"}
    df = df[~df["subj"].isin(skip)].copy()
    if len(df) == 0:
        return

    df["set_index"] = pd.to_numeric(df["set_index"], errors="coerce")
    df["n_trials"] = pd.to_numeric(df["n_trials"], errors="coerce")
    df["ll_diff_per_trial"] = pd.to_numeric(df["ll_diff_per_trial"], errors="coerce")
    df["ll_gradual_pt"] = pd.to_numeric(df["ll_gradual"], errors="coerce") / df["n_trials"].replace(0, np.nan)
    df["ll_one_shot_pt"] = pd.to_numeric(df["ll_one_shot"], errors="coerce") / df["n_trials"].replace(0, np.nan)

    rows = []
    for subj, g in df.groupby("subj"):
        si = g["set_index"]
        g12 = si.isin([1.0, 2.0])
        g78 = si.isin([7.0, 8.0])
        for metric in ["ll_diff_per_trial", "ll_gradual_pt", "ll_one_shot_pt"]:
            e = pd.to_numeric(g.loc[g12, metric], errors="coerce").mean()
            l = pd.to_numeric(g.loc[g78, metric], errors="coerce").mean()
            if np.isfinite(e) and np.isfinite(l):
                rows.append({"subj": subj, "window": "sets1_2", "metric": metric, "value": float(e)})
                rows.append({"subj": subj, "window": "sets7_8", "metric": metric, "value": float(l)})
    long_df = pd.DataFrame(rows)
    if len(long_df) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), constrained_layout=False)

    d_a = long_df[long_df["metric"] == "ll_diff_per_trial"].copy()
    grouped_bar(
        axes[0],
        d_a,
        "window",
        "value",
        ["sets1_2", "sets7_8"],
        "Gradual Advantage",
        "Δ LL / trial (gradual − one-shot)",
        sig_pairs=[("sets1_2", "sets7_8")],
        stat_meta=("figureabc_hmm_set12_vs_set78", "A"),
    )
    axes[0].set_xticklabels(["Sets 1-2", "Sets 7-8"], rotation=12)

    d_b = long_df[long_df["metric"] == "ll_gradual_pt"].copy()
    grouped_bar(
        axes[1],
        d_b,
        "window",
        "value",
        ["sets1_2", "sets7_8"],
        "Gradual Model Fit",
        "LL / trial",
        sig_pairs=[("sets1_2", "sets7_8")],
        stat_meta=("figureabc_hmm_set12_vs_set78", "B"),
    )
    axes[1].set_xticklabels(["Sets 1-2", "Sets 7-8"], rotation=12)

    d_c = long_df[long_df["metric"] == "ll_one_shot_pt"].copy()
    grouped_bar(
        axes[2],
        d_c,
        "window",
        "value",
        ["sets1_2", "sets7_8"],
        "One-shot Model Fit",
        "LL / trial",
        sig_pairs=[("sets1_2", "sets7_8")],
        stat_meta=("figureabc_hmm_set12_vs_set78", "C"),
    )
    axes[2].set_xticklabels(["Sets 1-2", "Sets 7-8"], rotation=12)

    _label_panels(axes, "ABC")
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.16, top=0.94, wspace=0.10)
    _save_figure_compact(fig, "figureabc_hmm_set12_vs_set78.png")
    plt.close(fig)


def make_hmm_fit_params_by_set_lines() -> None:
    """New HMM line figure: parameter/fitting trends across sets."""
    if not os.path.isfile(HMM_SET_BY_SET):
        print("[WARN] hmm_set_by_set_fullonly.csv missing; skip HMM set-line figure.")
        return

    df = pd.read_csv(HMM_SET_BY_SET)
    skip = {"reknow011", "reknow020", "reknow023"}
    df = df[~df["subj"].isin(skip)].copy()
    if len(df) == 0:
        return

    df["set_index"] = pd.to_numeric(df["set_index"], errors="coerce")
    df["n_trials"] = pd.to_numeric(df["n_trials"], errors="coerce")
    df["ll_diff_per_trial"] = pd.to_numeric(df["ll_diff_per_trial"], errors="coerce")
    df["ll_gradual_pt"] = pd.to_numeric(df["ll_gradual"], errors="coerce") / df["n_trials"].replace(0, np.nan)
    df["ll_one_shot_pt"] = pd.to_numeric(df["ll_one_shot"], errors="coerce") / df["n_trials"].replace(0, np.nan)
    df = df.dropna(subset=["set_index"]).copy()

    fig, axes = plt.subplots(3, 1, figsize=(8.8, 10.6), constrained_layout=False, sharex=True)
    specs = [
        ("ll_diff_per_trial", "Δ LL / trial (gradual − one-shot)", "tab:purple"),
        ("ll_gradual_pt", "Gradual LL / trial", "tab:blue"),
        ("ll_one_shot_pt", "One-shot LL / trial", "tab:orange"),
    ]

    for ax, (metric, ylabel, color) in zip(axes, specs):
        s = (
            df.groupby(["subj", "set_index"], as_index=False)[metric]
            .mean()
            .dropna(subset=[metric])
        )
        agg = s.groupby("set_index")[metric].agg(["mean", "sem"]).reset_index()
        x = agg["set_index"].to_numpy(dtype=float)
        y = agg["mean"].to_numpy(dtype=float)
        se = np.nan_to_num(agg["sem"].to_numpy(dtype=float), nan=0.0)
        ax.plot(x, y, marker="o", linewidth=2.2, color=color)
        ax.fill_between(x, y - se, y + se, color=color, alpha=0.22)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlim(0.8, 8.2)
        ax.set_xticks(np.arange(1, 9, 1))

    axes[-1].set_xlabel("Set index")
    _label_panels(axes, "ABC")
    fig.subplots_adjust(left=0.11, right=0.995, bottom=0.07, top=0.96, hspace=0.14)
    _save_figure_compact(fig, "figure17_hmm_fit_params_by_set_lines.png")
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    clear_figure_inference_log()

    beh = build_behavior_master()
    make_behavior_figures(beh)
    make_behavior_supplement_figures(beh)
    make_hmm_figure_abc_set_windows()
    make_hmm_behavior_figure()
    make_hmm_fit_params_by_set_lines()

    eeg_m = build_eeg_merged()
    make_eeg_feedback_figure(eeg_m)
    make_feedback_waveform_figures(eeg_m)

    make_priority_correlation_figure()
    make_cue_optional_figure()
    copy_isrsa_triptych()

    csv_path = save_figure_inference_table(OUT_DIR)

    with open(os.path.join(OUT_DIR, "README_figures.txt"), "w", encoding="utf-8") as f:
        f.write("Generated figures for main-text candidate set (1s_comp):\n")
        for name in sorted(os.listdir(OUT_DIR)):
            f.write(f"- {name}\n")

    print("saved figures ->", OUT_DIR)
    print("inferential tests table ->", csv_path)
    print("\n".join(sorted(os.listdir(OUT_DIR))))


if __name__ == "__main__":
    main()

