import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from figure_style_and_metrics import p_to_sig, set_helvetica_font, update_metrics_file


BASE = "base_dir"
OUT_DIR = os.path.join(BASE, "trials_trialwise", "1s_comp", "eeg_paper_results", "solidity")
BEH_CSV = os.path.join(BASE, "trials_trialwise", "all_subjects_trialwise.csv")
HMM_LONG_CSV = os.path.join(BASE, "trials_trialwise", "hmm_long_tables", "hmm_trial_long.csv")
TRIAL_POST_CSV = os.path.join(BASE, "trials_trialwise", "1s_comp", "eeg_paper_results", "solidity", "hmm_unified_trial_posteriors.csv")
set_helvetica_font()


def _safe_num(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    beh = pd.read_csv(BEH_CSV)
    hmm = pd.read_csv(HMM_LONG_CSV)
    post = pd.read_csv(TRIAL_POST_CSV)

    beh["subj"] = beh["subj"].astype(str)
    beh["block_id"] = beh["block_id"].astype(str)
    hmm["subj"] = hmm["subj"].astype(str)
    hmm["block_id"] = hmm["block_id"].astype(str)
    post["subj"] = post["subj"].astype(str)
    post["block_id"] = post["block_id"].astype(str)
    _safe_num(beh, ["trial_id", "correctness", "rt"])
    _safe_num(hmm, ["trial_id", "candidates_before"])
    _safe_num(post, ["trial_id", "trial_index_1based", "acquisition_trial_core"])

    beh["phase"] = beh["phase"].astype(str)
    beh["rule_level"] = beh["rule_level"].astype(str)
    return beh, hmm, post


def _subject_paired_table(d: pd.DataFrame, value_col: str) -> pd.DataFrame:
    g = d.groupby(["subj", "rule_level"], as_index=False).agg(val=(value_col, "mean")).dropna()
    wide = g.pivot(index="subj", columns="rule_level", values="val")
    if ("global" not in wide.columns) or ("local" not in wide.columns):
        return pd.DataFrame(columns=["subj", "global", "local"])
    wide = wide.dropna(subset=["global", "local"]).reset_index()
    return wide[["subj", "global", "local"]]


def _plot_paired(
    ax: plt.Axes,
    wide: pd.DataFrame,
    ylabel: str,
    title: str,
    ylim=None,
    show_bracket: bool = True,
    star_offset: float = 0.01,
):
    xg, xl = 0.00, 0.56
    for _, r in wide.iterrows():
        ax.plot([xg, xl], [r["global"], r["local"]], color="#9aa0a6", alpha=0.40, lw=1)
    ax.scatter(np.repeat(xg, len(wide)), wide["global"], s=16, color="#2563eb", alpha=0.85, zorder=3)
    ax.scatter(np.repeat(xl, len(wide)), wide["local"], s=16, color="#ea580c", alpha=0.85, zorder=3)

    mg = float(wide["global"].mean()) if len(wide) else np.nan
    ml = float(wide["local"].mean()) if len(wide) else np.nan
    ax.plot([xg, xl], [mg, ml], color="#111827", lw=2.2, zorder=4)
    ax.scatter([xg, xl], [mg, ml], color="#111827", s=30, zorder=5)

    if len(wide) >= 2:
        _, p = ttest_rel(wide["global"], wide["local"], nan_policy="omit")
        sig = p_to_sig(float(p))
        y_ref = float(np.nanmax(np.r_[wide["global"].to_numpy(dtype=float), wide["local"].to_numpy(dtype=float)]))
        y0 = y_ref + (0.06 * max(1.0, abs(y_ref)))
        if show_bracket:
            ax.plot([xg, xg, xl, xl], [y0 - 0.04, y0, y0, y0 - 0.04], color="black", linewidth=1)
        ax.text((xg + xl) / 2, y0 + star_offset, sig, ha="center", va="bottom", fontsize=11, fontweight="bold")
        lo, hi = ax.get_ylim()
        # Keep star close to bracket but leave clear headroom from top border.
        ax.set_ylim(lo, max(hi, y0 + 0.22))
    ax.set_xticks([xg, xl])
    ax.set_xticklabels(["Global", "Local"])
    ax.set_xlim(-0.22, 0.78)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25, linestyle="--")


def _plot_uncertainty_modulation(
    ax: plt.Axes, beh: pd.DataFrame, post: pd.DataFrame, hmm: pd.DataFrame
) -> Optional[Dict[str, Any]]:
    """
    Pre-acquisition (HMM-core) trials only: trial_index_1based < acquisition_trial_core
    (same convention as figure28_global_precedence_search / expand plot).
    Uncertainty from logical candidates_before (hmm_trial_long).
    """
    x_low, x_high = 0.00, 0.56
    pcols = post[["subj", "block_id", "trial_id", "trial_index_1based", "acquisition_trial_core"]].dropna()
    ds = beh.merge(pcols, on=["subj", "block_id", "trial_id"], how="inner")
    ds = ds[ds["rule_level"].isin(["global", "local"])].copy()
    ti = pd.to_numeric(ds["trial_index_1based"], errors="coerce")
    core = pd.to_numeric(ds["acquisition_trial_core"], errors="coerce")
    ds = ds[np.isfinite(ti) & np.isfinite(core) & (ti < core)].copy()

    hm = hmm[["subj", "block_id", "trial_id", "candidates_before"]].dropna()
    ds = ds.merge(hm, on=["subj", "block_id", "trial_id"], how="inner")
    ds["uncertainty_high"] = (pd.to_numeric(ds["candidates_before"], errors="coerce") > 1).astype(int)
    ds = ds.dropna(subset=["uncertainty_high", "rt"])
    if ds.empty:
        ax.text(0.5, 0.5, "No pre-core data", ha="center", va="center")
        ax.set_title("C. Pre-core RT gap by uncertainty (trial < core)")
        return None

    subj_gap_rows = []
    for (subj, uh), g in ds.groupby(["subj", "uncertainty_high"]):
        gm = g.groupby("rule_level")["rt"].mean()
        if ("local" in gm.index) and ("global" in gm.index):
            subj_gap_rows.append(
                {
                    "subj": subj,
                    "uncertainty_high": int(uh),
                    "rt_gap_local_minus_global": float(gm["local"] - gm["global"]),
                }
            )
    gg = pd.DataFrame(subj_gap_rows)
    if gg.empty:
        ax.text(0.5, 0.5, "No paired uncertainty data", ha="center", va="center")
        ax.set_title("C. Pre-core RT gap by uncertainty (trial < core)")
        return None

    wide = gg.pivot(index="subj", columns="uncertainty_high", values="rt_gap_local_minus_global")
    cols = [c for c in [0, 1] if c in wide.columns]
    if len(cols) < 2:
        ax.text(0.5, 0.5, "Need both low/high uncertainty", ha="center", va="center")
        ax.set_title("C. Pre-core RT gap by uncertainty (trial < core)")
        return None
    wide = wide.dropna(subset=[0, 1]).reset_index()
    wide.columns = ["subj", "low_uncertainty", "high_uncertainty"]

    for _, r in wide.iterrows():
        ax.plot([x_low, x_high], [r["low_uncertainty"], r["high_uncertainty"]], color="#9aa0a6", alpha=0.40, lw=1)
    ax.scatter(np.repeat(x_low, len(wide)), wide["low_uncertainty"], s=16, color="#16a34a", alpha=0.85, zorder=3)
    ax.scatter(np.repeat(x_high, len(wide)), wide["high_uncertainty"], s=16, color="#7c3aed", alpha=0.85, zorder=3)

    m0 = float(wide["low_uncertainty"].mean())
    m1 = float(wide["high_uncertainty"].mean())
    ax.plot([x_low, x_high], [m0, m1], color="#111827", lw=2.2, zorder=4)
    ax.scatter([x_low, x_high], [m0, m1], color="#111827", s=30, zorder=5)

    tout, pout = np.nan, np.nan
    if len(wide) >= 2:
        tout, pout = ttest_rel(wide["high_uncertainty"], wide["low_uncertainty"], nan_policy="omit")
        sig = p_to_sig(float(pout))
        y_ref = float(np.nanmax(np.r_[wide["low_uncertainty"].to_numpy(dtype=float), wide["high_uncertainty"].to_numpy(dtype=float)]))
        y0 = y_ref + (0.06 * max(1.0, abs(y_ref)))
        ax.plot([x_low, x_low, x_high, x_high], [y0 - 0.04, y0, y0, y0 - 0.04], color="black", linewidth=1)
        ax.text((x_low + x_high) / 2, y0 + 0.001, sig, ha="center", va="bottom", fontsize=11, fontweight="bold")
        lo, hi = ax.get_ylim()
        # Match panel A style: star close to bracket with extra top margin.
        ax.set_ylim(lo, max(hi, y0 + 0.22))

    ax.axhline(0, color="#6b7280", lw=1, ls="--", alpha=0.6)
    ax.set_xticks([x_low, x_high])
    ax.set_xticklabels(["Low uncertainty", "High uncertainty"])
    ax.set_xlim(-0.22, 0.78)
    ax.set_ylabel("RT gap (local $-$ global), s")
    ax.set_title("C. Pre-core RT gap by uncertainty (trial $<$ core)")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    return {
        "metric": "precore_rtgap_high_minus_low_uncertainty",
        "n_subjects": int(len(wide)),
        "mean_low": float(wide["low_uncertainty"].mean()),
        "mean_high": float(wide["high_uncertainty"].mean()),
        "paired_t": float(tout),
        "p_value": float(pout),
        "significance": p_to_sig(float(pout)) if np.isfinite(pout) else "NA",
    }


def make_figure():
    fig_title = "Global precedence in acquired phase"
    fig_caption = (
        "Panel A compares HMM-core-defined acquired-phase RT between global and local rule levels. "
        "Panel B compares HMM-core-defined acquired-phase accuracy."
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    beh, hmm, post = _load_tables()

    # Acquired trials are defined by unified HMM core:
    # trial_index_1based >= acquisition_trial_core.
    pcols = post[["subj", "block_id", "trial_id", "trial_index_1based", "acquisition_trial_core"]].dropna()
    d_acquired = beh.merge(pcols, on=["subj", "block_id", "trial_id"], how="inner")
    d_acquired = d_acquired[
        d_acquired["rule_level"].isin(["global", "local"])
        & (pd.to_numeric(d_acquired["trial_index_1based"], errors="coerce") >= pd.to_numeric(d_acquired["acquisition_trial_core"], errors="coerce"))
    ].copy()

    rt_wide = _subject_paired_table(d_acquired, "rt") if not d_acquired.empty else pd.DataFrame(columns=["subj", "global", "local"])
    acc_wide = _subject_paired_table(d_acquired, "correctness") if not d_acquired.empty else pd.DataFrame(columns=["subj", "global", "local"])

    fig, axes = plt.subplots(1, 2, figsize=(9.7, 4.6), constrained_layout=False)
    _plot_paired(
        axes[0],
        rt_wide,
        ylabel="Reaction time (s)",
        title="A. Acquired phase: RT by rule level",
        star_offset=0.001,
    )
    _plot_paired(
        axes[1],
        acc_wide,
        ylabel="Accuracy",
        title="B. Acquired phase: Accuracy by rule level",
        ylim=(0.7, 1.1),
        show_bracket=True,
        star_offset=0.001,
    )

    fig.suptitle(fig_title, fontsize=14, y=1.02)
    # Keep title close to panels (no extra blank line below title).
    fig.subplots_adjust(top=0.84, wspace=0.24)
    out_png = os.path.join(OUT_DIR, "figure28_global_precedence_acquired_phase.png")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # small summary table for quick reporting
    out_rows = []
    if len(rt_wide) >= 2:
        t, p = ttest_rel(rt_wide["global"], rt_wide["local"], nan_policy="omit")
        out_rows.append(
            {
                "metric": "acquired_rt_global_vs_local",
                "n_subjects": int(len(rt_wide)),
                "global_mean": float(rt_wide["global"].mean()),
                "local_mean": float(rt_wide["local"].mean()),
                "paired_t": float(t),
                "p_value": float(p),
                "significance": p_to_sig(float(p)),
            }
        )
    if len(acc_wide) >= 2:
        t, p = ttest_rel(acc_wide["global"], acc_wide["local"], nan_policy="omit")
        out_rows.append(
            {
                "metric": "acquired_acc_global_vs_local",
                "n_subjects": int(len(acc_wide)),
                "global_mean": float(acc_wide["global"].mean()),
                "local_mean": float(acc_wide["local"].mean()),
                "paired_t": float(t),
                "p_value": float(p),
                "significance": p_to_sig(float(p)),
            }
        )
    out_csv = os.path.join(OUT_DIR, "figure28_global_precedence_acquired_phase_summary.csv")
    pd.DataFrame(out_rows).to_csv(out_csv, index=False)
    metric_rows = []
    for r in out_rows:
        metric_rows.append(
            {
                "panel": "A/B",
                "metric_name": r.get("metric", ""),
                "p_value": r.get("p_value", np.nan),
                "significance": r.get("significance", ""),
                "effect_or_stat": f"t={r.get('paired_t', np.nan):.3f}",
                "notes": f"n={int(r.get('n_subjects', 0))}",
            }
        )
    if not metric_rows:
        metric_rows.append(
            {
                "panel": "A/B",
                "metric_name": "descriptive_global_precedence_plot",
                "p_value": np.nan,
                "significance": "NA",
                "effect_or_stat": "",
                "notes": "No inferential statistics available.",
            }
        )
    update_metrics_file(
        figure_id="figure28",
        figure_title=fig_title,
        figure_caption=fig_caption,
        rows=metric_rows,
    )
    print(f"[OK] figure -> {out_png}")
    print(f"[OK] summary -> {out_csv}")


if __name__ == "__main__":
    make_figure()
