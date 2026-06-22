import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import ttest_rel
from figure_style_and_metrics import p_to_sig, set_helvetica_font, update_metrics_file


BASE_SOL = "base_sol_dir"
TRIAL_POST = os.path.join(BASE_SOL, "hmm_unified_trial_posteriors.csv")
EEG_TRIAL = "eeg_trial_path"
BEH_TRIAL = "beh_trial_path"
HMM_LONG = "hmm_long_dir"
set_helvetica_font()


def _safe_num(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _set_index_map(df: pd.DataFrame, subj_col="subj", block_col="block_id") -> pd.DataFrame:
    d = df.copy()
    d[block_col] = d[block_col].astype(str)
    d["block_num"] = pd.to_numeric(d[block_col], errors="coerce")
    out_rows = []
    for subj, g in d.groupby(subj_col, sort=False):
        ub = g[[block_col, "block_num"]].dropna().drop_duplicates().sort_values("block_num")
        bids = ub[block_col].astype(str).to_numpy()
        b2s = {}
        for si, ch in enumerate(np.array_split(bids, 8), start=1):
            for b in ch:
                b2s[str(b)] = float(si)
        for b in bids:
            out_rows.append({subj_col: subj, block_col: str(b), "set_index": b2s[str(b)]})
    return pd.DataFrame(out_rows).drop_duplicates()


def run_erp_changes():
    fig_title = "P300/FRN Stage Changes: Early vs Late Sets"
    fig_caption = (
        "FRN and P3b stage means across search error, transition pre-core correct, transition first-correct, and acquired correct "
        "for early and late sets."
    )
    eeg = pd.read_csv(EEG_TRIAL)
    post = pd.read_csv(TRIAL_POST)
    eeg["subj"] = eeg["subj"].astype(str)
    eeg["block_id"] = eeg["block_id"].astype(str)
    post["subj"] = post["subj"].astype(str)
    post["block_id"] = post["block_id"].astype(str)
    _safe_num(eeg, ["trial_id", "feedback_locked_FRN", "feedback_locked_P3b"])
    _safe_num(post, ["trial_id", "correctness", "trial_index_1based", "first_correct_trial", "acquisition_trial_core", "set_index"])

    m = eeg.merge(
        post[
            [
                "subj",
                "block_id",
                "trial_id",
                "correctness",
                "trial_index_1based",
                "first_correct_trial",
                "acquisition_trial_core",
                "set_index",
            ]
        ],
        on=["subj", "block_id", "trial_id"],
        how="inner",
    )
    m["set_bin"] = np.where(m["set_index"] <= 2, "early_1_2", np.where(m["set_index"] >= 7, "late_7_8", "middle"))

    is_fc = m["trial_index_1based"] == m["first_correct_trial"]
    is_transition_precore_corr = (m["trial_index_1based"] > m["first_correct_trial"]) & (m["trial_index_1based"] < m["acquisition_trial_core"]) & (m["correctness"] == 1)
    is_sw = (m["trial_index_1based"] < m["first_correct_trial"]) & (m["correctness"] == 0)
    is_sc = (m["trial_index_1based"] < m["first_correct_trial"]) & (m["correctness"] == 1)
    is_acquired_corr = (m["trial_index_1based"] >= m["acquisition_trial_core"]) & (m["correctness"] == 1)

    m["category"] = ""
    m.loc[is_sw, "category"] = "search_error"
    m.loc[is_sc, "category"] = "search_correct"
    m.loc[is_fc, "category"] = "transition_first_correct"
    m.loc[is_transition_precore_corr, "category"] = "transition_pre_core_correct"
    m.loc[is_acquired_corr, "category"] = "acquired_correct"
    m = m[m["category"] != ""].copy()

    rows = []
    for feat in ["feedback_locked_FRN", "feedback_locked_P3b"]:
        for sb in ["early_1_2", "late_7_8", "all_sets"]:
            g = m.copy() if sb == "all_sets" else m[m["set_bin"] == sb].copy()
            blk = g.groupby(["subj", "block_id", "category"], as_index=False)[feat].mean()
            subj_cat = blk.groupby(["subj", "category"], as_index=False)[feat].mean()
            piv = subj_cat.pivot_table(index="subj", columns="category", values=feat)
            for c in ["transition_first_correct", "search_error", "transition_pre_core_correct", "acquired_correct"]:
                if c in piv.columns:
                    rows.append(
                        {
                            "feature": feat,
                            "set_bin": sb,
                            "category": c,
                            "mean": float(np.nanmean(piv[c])),
                            "sd": float(np.nanstd(piv[c], ddof=1)),
                            "n_subjects": int(np.isfinite(piv[c]).sum()),
                        }
                    )
            # paired contrasts
            for a, b in [("transition_first_correct", "search_error"), ("transition_first_correct", "transition_pre_core_correct"), ("transition_first_correct", "acquired_correct")]:
                if a in piv.columns and b in piv.columns:
                    d = piv[[a, b]].dropna()
                    if len(d) >= 8:
                        t, p = ttest_rel(d[a], d[b], nan_policy="omit")
                        rows.append(
                            {
                                "feature": feat,
                                "set_bin": sb,
                                "category": f"{a}_minus_{b}",
                                "mean": float((d[a] - d[b]).mean()),
                                "sd": float((d[a] - d[b]).std(ddof=1)),
                                "n_subjects": int(len(d)),
                                "t": float(t),
                                "p": float(p),
                            }
                        )
    out = pd.DataFrame(rows)
    p_csv = os.path.join(BASE_SOL, "p300_frn_changes_summary.csv")
    out.to_csv(p_csv, index=False)

    # quick plot
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8), constrained_layout=False)
    fig.subplots_adjust(right=0.80, wspace=0.32)
    for ax, feat in zip(axes, ["feedback_locked_FRN", "feedback_locked_P3b"]):
        dat = out[(out["feature"] == feat) & (out["category"].isin(["search_error", "transition_pre_core_correct", "transition_first_correct", "acquired_correct"]))].copy()
        order = ["search_error", "transition_pre_core_correct", "transition_first_correct", "acquired_correct"]
        x = np.arange(len(order))
        for sb, col in [("early_1_2", "#d95f0e"), ("late_7_8", "#1b9e77")]:
            d2 = dat[dat["set_bin"] == sb].set_index("category").reindex(order)
            y = d2["mean"].to_numpy(dtype=float)
            se = d2["sd"].to_numpy(dtype=float) / np.sqrt(np.maximum(1, d2["n_subjects"].to_numpy(dtype=float)))
            ax.errorbar(x, y, yerr=1.96 * se, marker="o", linewidth=2, capsize=3, color=col, label=sb)
        ax.set_xticks(x)
        ax.set_xticklabels(["Search\nerror", "Transition\npre-core", "Transition\nfirst correct", "Acquired\ncorrect"])
        ax.set_title(feat)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        # Add early/late/overall comparison marks at top (stacked, no overlap with border).
        pairs = [
            ("transition_first_correct_minus_search_error", 0, 2),
            ("transition_first_correct_minus_transition_pre_core_correct", 1, 2),
            ("transition_first_correct_minus_acquired_correct", 2, 3),
        ]
        y_max = float(np.nanmax(dat["mean"].to_numpy(dtype=float))) if not dat.empty else 0.0
        # Extend y-axis to reserve a clear top band for significance marks.
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, max(hi, y_max + 2.2))
        group_defs = [
            ("early_1_2", "#d95f0e", -0.08),
            ("late_7_8", "#1b9e77", 0.00),
            ("all_sets", "#4b5563", 0.08),
        ]

        sig_entries = []
        for cname, x1, x2 in pairs:
            for sb, col, xoff in group_defs:
                ann = out[
                    (out["feature"] == feat)
                    & (out["category"] == cname)
                    & (out["set_bin"] == sb)
                ]
                if ann.empty:
                    continue
                pv = float(ann.iloc[0]["p"])
                if (not np.isfinite(pv)) or (pv >= 0.05):
                    continue
                sig = p_to_sig(pv)
                sig_entries.append((x1, x2, xoff, col, sig))

        # Reserve enough headroom based on how many marks we will draw.
        n_sig = len(sig_entries)
        if n_sig > 0:
            needed_top = y_max + 0.9 + n_sig * 0.22
            lo2, hi2 = ax.get_ylim()
            ax.set_ylim(lo2, max(hi2, needed_top))
            y_top = ax.get_ylim()[1]
            for level, (x1, x2, xoff, col, sig) in enumerate(sig_entries):
                # Place below the top border with a fixed safety margin.
                y0 = y_top - 0.40 - level * 0.22
                ax.plot([x1 + xoff, x1 + xoff, x2 + xoff, x2 + xoff], [y0 - 0.07, y0, y0, y0 - 0.07], color=col, linewidth=1)
                ax.text(
                    (x1 + x2) / 2 + xoff,
                    y0 + 0.006,
                    sig,
                    ha="center",
                    va="bottom",
                    fontsize=10.5,
                    fontweight="bold",
                    color="black",
                )
    axes[0].set_ylabel("Amplitude (uV)")
    # Right-side note block (next to P3b panel): line-color and comparison-bracket explanations.
    line_handles = [
        Line2D([0], [0], color="#d95f0e", lw=2, marker="o", label="Line: early sets (1-2)"),
        Line2D([0], [0], color="#1b9e77", lw=2, marker="o", label="Line: late sets (7-8)"),
        Line2D([0], [0], color="#d95f0e", lw=1.2, label="Bracket: early-set contrast"),
        Line2D([0], [0], color="#1b9e77", lw=1.2, label="Bracket: late-set contrast"),
        Line2D([0], [0], color="#4b5563", lw=1.2, label="Bracket: overall contrast"),
    ]
    fig.legend(
        handles=line_handles,
        loc="center left",
        bbox_to_anchor=(0.81, 0.62),
        frameon=False,
        fontsize=9,
        handlelength=2.2,
    )
    fig.text(
        0.81,
        0.34,
        "Top brackets mark pairwise comparisons.\n"
        "Stars indicate significance level only\n"
        "(*, **, ***; non-significant not shown).",
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.suptitle(fig_title)
    p_fig = os.path.join(BASE_SOL, "figure27_p300_frn_stage_changes_early_vs_late.png")
    fig.savefig(p_fig, dpi=320, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    update_metrics_file(
        figure_id="figure27",
        figure_title=fig_title,
        figure_caption=fig_caption,
        rows=[
            {
                "panel": "FRN/P3b",
                "metric_name": "descriptive_stage_means",
                "p_value": np.nan,
                "significance": "NA",
                "effect_or_stat": "",
                "notes": "Inferential statistics are stored in p300_frn_changes_summary.csv",
            }
        ],
    )
    return p_csv, p_fig


def run_global_precedence():
    beh = pd.read_csv(BEH_TRIAL)
    hmm = pd.read_csv(HMM_LONG)

    beh["subj"] = beh["subj"].astype(str)
    beh["block_id"] = beh["block_id"].astype(str)
    _safe_num(beh, ["trial_id", "correctness", "rt"])
    set_map = _set_index_map(beh)
    beh = beh.merge(set_map, on=["subj", "block_id"], how="left")
    beh["phase"] = beh["phase"].astype(str)
    beh["rule_level"] = beh["rule_level"].astype(str)
    beh["acquired_phase"] = (beh["phase"] != "search").astype(int)

    # Base global precedence: global vs local in acquired/post-search phase.
    st = beh[beh["acquired_phase"] == 1].copy()
    subj_rule = st.groupby(["subj", "rule_level"], as_index=False).agg(rt_mean=("rt", "mean"), acc_mean=("correctness", "mean"))
    piv_rt = subj_rule.pivot_table(index="subj", columns="rule_level", values="rt_mean")
    piv_acc = subj_rule.pivot_table(index="subj", columns="rule_level", values="acc_mean")

    gp_rows = []
    if {"global", "local"}.issubset(piv_rt.columns):
        d = piv_rt[["global", "local"]].dropna()
        if len(d) >= 8:
            t, p = ttest_rel(d["local"], d["global"], nan_policy="omit")  # local-global >0 => global faster
            gp_rows.append({"metric": "rt_local_minus_global_acquired", "n_subjects": int(len(d)), "mean_diff": float((d["local"] - d["global"]).mean()), "t": float(t), "p": float(p)})
    if {"global", "local"}.issubset(piv_acc.columns):
        d = piv_acc[["global", "local"]].dropna()
        if len(d) >= 8:
            t, p = ttest_rel(d["global"], d["local"], nan_policy="omit")
            gp_rows.append({"metric": "acc_global_minus_local_acquired", "n_subjects": int(len(d)), "mean_diff": float((d["global"] - d["local"]).mean()), "t": float(t), "p": float(p)})

    # Uncertainty modulation (search phase only), using candidates_before from hmm long table.
    hmm["subj"] = hmm["subj"].astype(str)
    hmm["block_id"] = hmm["block_id"].astype(str)
    _safe_num(hmm, ["trial_id", "candidates_before"])
    hm = hmm[["subj", "block_id", "trial_id", "candidates_before"]].dropna()
    d = beh.merge(hm, on=["subj", "block_id", "trial_id"], how="inner")
    d = d[d["phase"] == "search"].copy()
    d["uncertainty_bin"] = np.where(pd.to_numeric(d["candidates_before"], errors="coerce") > 1, "high_uncertainty", "low_uncertainty")
    sr = d.groupby(["subj", "uncertainty_bin", "rule_level"], as_index=False)["rt"].mean()
    piv = sr.pivot_table(index=["subj", "uncertainty_bin"], columns="rule_level", values="rt")
    if {"global", "local"}.issubset(piv.columns):
        piv = piv.reset_index()
        piv["rt_local_minus_global"] = piv["local"] - piv["global"]
        uu = piv.pivot_table(index="subj", columns="uncertainty_bin", values="rt_local_minus_global")
        if {"high_uncertainty", "low_uncertainty"}.issubset(uu.columns):
            dd = uu[["high_uncertainty", "low_uncertainty"]].dropna()
            if len(dd) >= 8:
                t, p = ttest_rel(dd["high_uncertainty"], dd["low_uncertainty"], nan_policy="omit")
                gp_rows.append(
                    {
                        "metric": "global_precedence_gap_modulated_by_uncertainty",
                        "n_subjects": int(len(dd)),
                        "mean_diff": float((dd["high_uncertainty"] - dd["low_uncertainty"]).mean()),
                        "t": float(t),
                        "p": float(p),
                    }
                )

    out = pd.DataFrame(gp_rows)
    p_csv = os.path.join(BASE_SOL, "global_precedence_summary.csv")
    out.to_csv(p_csv, index=False)
    return p_csv


def main():
    os.makedirs(BASE_SOL, exist_ok=True)
    p_erp_csv, p_erp_fig = run_erp_changes()
    p_gp_csv = run_global_precedence()
    print(f"saved: {p_erp_csv}")
    print(f"saved: {p_erp_fig}")
    print(f"saved: {p_gp_csv}")
    print(pd.read_csv(p_gp_csv).to_string(index=False))


if __name__ == "__main__":
    main()
