import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel


BASE = "base_dir"
OUT_DIR = os.path.join(BASE, "trials_trialwise", "1s_comp", "eeg_paper_results", "solidity")
BEH_CSV = os.path.join(BASE, "trials_trialwise", "all_subjects_trialwise.csv")
TRIAL_POST_CSV = os.path.join(BASE, "trials_trialwise", "1s_comp", "eeg_paper_results", "solidity", "hmm_unified_trial_posteriors.csv")


def p_to_sig(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


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
    star_offset: float = 0.001,
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
        ax.set_ylim(lo, max(hi, y0 + 0.22))

    ax.set_xticks([xg, xl])
    ax.set_xticklabels(["Global", "Local"])
    ax.set_xlim(-0.22, 0.78)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25, linestyle="--")


def _make_one_figure(beh: pd.DataFrame, subset_name: str, fig_name: str, fig_title: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    d = beh[beh["rule_level"].isin(["global", "local"])].copy()

    rt_wide = _subject_paired_table(d, "rt")
    acc_wide = _subject_paired_table(d, "correctness")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), constrained_layout=False)
    _plot_paired(
        axes[0],
        rt_wide,
        ylabel="Reaction time (ms)",
        title=f"A. {subset_name}: RT by rule level",
        star_offset=0.001,
    )
    _plot_paired(
        axes[1],
        acc_wide,
        ylabel="Accuracy",
        title=f"B. {subset_name}: Accuracy by rule level",
        star_offset=0.001,
    )
    fig.suptitle(fig_title, fontsize=13, y=0.98)
    fig.subplots_adjust(top=0.84, wspace=0.25)
    out_png = os.path.join(OUT_DIR, fig_name)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    rows = []
    if len(rt_wide) >= 2:
        t, p = ttest_rel(rt_wide["local"], rt_wide["global"], nan_policy="omit")
        rows.append(
            {
                "subset": subset_name,
                "metric": "rt_local_minus_global",
                "n_subjects": int(len(rt_wide)),
                "global_mean": float(rt_wide["global"].mean()),
                "local_mean": float(rt_wide["local"].mean()),
                "mean_diff": float((rt_wide["local"] - rt_wide["global"]).mean()),
                "paired_t": float(t),
                "p_value": float(p),
                "significance": p_to_sig(float(p)),
            }
        )
    if len(acc_wide) >= 2:
        t, p = ttest_rel(acc_wide["global"], acc_wide["local"], nan_policy="omit")
        rows.append(
            {
                "subset": subset_name,
                "metric": "acc_global_minus_local",
                "n_subjects": int(len(acc_wide)),
                "global_mean": float(acc_wide["global"].mean()),
                "local_mean": float(acc_wide["local"].mean()),
                "mean_diff": float((acc_wide["global"] - acc_wide["local"]).mean()),
                "paired_t": float(t),
                "p_value": float(p),
                "significance": p_to_sig(float(p)),
            }
        )
    out_csv = os.path.join(OUT_DIR, fig_name.replace(".png", "_summary.csv"))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_png, out_csv


def main():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    beh = pd.read_csv(BEH_CSV)
    post = pd.read_csv(TRIAL_POST_CSV)
    beh["subj"] = beh["subj"].astype(str)
    beh["block_id"] = beh["block_id"].astype(str)
    beh["phase"] = beh["phase"].astype(str)
    beh["rule_level"] = beh["rule_level"].astype(str)
    beh["trial_id"] = pd.to_numeric(beh["trial_id"], errors="coerce")
    beh["rt"] = pd.to_numeric(beh["rt"], errors="coerce")
    beh["correctness"] = pd.to_numeric(beh["correctness"], errors="coerce")
    post["subj"] = post["subj"].astype(str)
    post["block_id"] = post["block_id"].astype(str)
    post["trial_id"] = pd.to_numeric(post["trial_id"], errors="coerce")
    post["trial_index_1based"] = pd.to_numeric(post["trial_index_1based"], errors="coerce")
    post["acquisition_trial_core"] = pd.to_numeric(post["acquisition_trial_core"], errors="coerce")

    # Use unified-HMM core markers for phase logic consistency.
    pm = post[["subj", "block_id", "trial_id", "trial_index_1based", "acquisition_trial_core"]].dropna()
    dat = beh.merge(pm, on=["subj", "block_id", "trial_id"], how="inner")
    dat = dat[dat["rule_level"].isin(["global", "local"])].copy()

    all_d = dat.copy()
    search_d = dat[
        pd.to_numeric(dat["trial_index_1based"], errors="coerce") < pd.to_numeric(dat["acquisition_trial_core"], errors="coerce")
    ].copy()

    p1, c1 = _make_one_figure(
        all_d,
        subset_name="Overall (core-aligned all trials)",
        fig_name="figure28_global_precedence_overall.png",
        fig_title="Global precedence: overall (core-aligned)",
    )
    p2, c2 = _make_one_figure(
        search_d,
        subset_name="Search phase (trial < core)",
        fig_name="figure28_global_precedence_search.png",
        fig_title="Global precedence: search phase (core-aligned)",
    )
    print(f"saved: {p1}")
    print(f"saved: {c1}")
    print(f"saved: {p2}")
    print(f"saved: {c2}")


if __name__ == "__main__":
    main()
