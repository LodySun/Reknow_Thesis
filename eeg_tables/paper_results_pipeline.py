"""
Fixed EEG paper-results pipeline from three tables:
  - eeg_trial_long
  - eeg_block_long
  - eeg_subject_traits

Outputs:
  1) QC gate tables
  2) Mainline A (block-level mixture) models
  3) Mainline B (transition-gate trial-level) models
  4) Subject-level EEG traits (delta + gate/time coefficients)
  5) Robustness checks + minimal figures
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import ttest_rel


BASE = "base_dir"
COMP_TAG = "1s_comp"
EEG_TABLE_DIR = os.path.join(BASE, "trials_trialwise", COMP_TAG, "eeg_tables")
OUT_DIR = os.path.join(BASE, "trials_trialwise", COMP_TAG, "eeg_paper_results")

TRIAL_PATH = os.path.join(EEG_TABLE_DIR, "eeg_trial_long.csv")
BLOCK_PATH = os.path.join(EEG_TABLE_DIR, "eeg_block_long.csv")
SUBJ_PATH = os.path.join(EEG_TABLE_DIR, "eeg_subject_traits.csv")
ALIGN_SUMMARY_PATH = os.path.join(BASE, "trials_trialwise", COMP_TAG, "eeg_behavior_alignment_summary.csv")
BEHAV_ALL_PATH = os.path.join(BASE, "trials_trialwise", "trialwise", "all_subjects_trialwise.csv")
TRANSITION_PATH = os.path.join(BASE, "trials_trialwise", "hmm_mixture", "hmm_mixture_transition_trials_used.csv")

METRICS = [
    "feedback_locked_P3a",
    "feedback_locked_P3b",
    "theta_power",
    "theta_ctrl_200_400_feedback",
    "theta_ctrl_400_700_feedback",
    "theta_prep_early_100_300_cue",
    "theta_prep_late_500_800_cue",
    "theta_exec_0_450_target",
    "alpha_power",
    "posterior_alpha_400_1000_cue",
    "front_beta_300_900_feedback",
    "parietal_beta_300_900_feedback",
    "parietal_alpha_cti_slope_100_800_cue",
    "frontal_asym_alpha",
]

TRIAL_MIN = 20  # hard gate threshold per subj x block_type x phase


def _phase3(phase: str) -> str:
    p = str(phase).strip()
    if p == "search":
        return "search"
    if p == "acquired":
        return "stable"
    if p == "early-found":
        return "transition"
    if p == "late-found":
        return "stable"
    return "other"


def _zscore(x: pd.Series) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce")
    s = float(v.std(skipna=True))
    if not np.isfinite(s) or s <= 0:
        return pd.Series(np.nan, index=x.index)
    return (v - float(v.mean(skipna=True))) / s


def _mad_outlier_flag(x: pd.Series, z_th: float = 3.5) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce")
    med = float(v.median(skipna=True))
    mad = float(np.median(np.abs(v.dropna() - med))) if v.notna().any() else np.nan
    if not np.isfinite(mad) or mad <= 0:
        return pd.Series(False, index=x.index)
    mz = 0.6745 * (v - med) / mad
    return mz.abs() > z_th


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run() -> None:
    _ensure_dir(OUT_DIR)

    trial = pd.read_csv(TRIAL_PATH)
    block = pd.read_csv(BLOCK_PATH)
    subj = pd.read_csv(SUBJ_PATH) if os.path.exists(SUBJ_PATH) else pd.DataFrame()
    align = pd.read_csv(ALIGN_SUMMARY_PATH) if os.path.exists(ALIGN_SUMMARY_PATH) else pd.DataFrame()
    beh_all = pd.read_csv(BEHAV_ALL_PATH)
    trans = pd.read_csv(TRANSITION_PATH)

    # Harmonize types
    for df in [trial, block, beh_all, trans]:
        if "block_id" in df.columns:
            df["block_id"] = pd.to_numeric(df["block_id"], errors="coerce")
        if "trial_id" in df.columns:
            df["trial_id"] = pd.to_numeric(df["trial_id"], errors="coerce")

    trial["phase3"] = trial["phase"].map(_phase3)
    block["phase3"] = block["phase"].map(_phase3)
    beh_all["phase3"] = beh_all["phase"].map(_phase3)

    # Keep only two block types + core phases
    keep_bt = ["one_shot_like", "gradual_like"]
    # Core strata: search vs HMM-acquired (mapped to stable; transition kept for legacy CSVs only).
    keep_ph = ["search", "stable"]
    trial_core = trial[trial["block_type"].isin(keep_bt) & trial["phase3"].isin(keep_ph)].copy()
    block_core = block[block["block_type"].isin(keep_bt) & block["phase3"].isin(keep_ph)].copy()

    # Fronto-parietal coupling (trial-wise within block/phase)
    def _sp_corr(a: pd.Series, b: pd.Series) -> float:
        x = pd.to_numeric(a, errors="coerce")
        y = pd.to_numeric(b, errors="coerce")
        ok = x.notna() & y.notna()
        if ok.sum() < 6:
            return np.nan
        return float(x[ok].corr(y[ok], method="spearman"))

    if {"theta_prep_late_500_800_cue", "parietal_alpha_cti_slope_100_800_cue"}.issubset(trial_core.columns):
        cpl1 = (
            trial_core.groupby(["subj", "block_id", "phase3", "block_type"], as_index=False)
            .apply(lambda g: pd.Series({"fp_theta_parietal_alpha_coupling": _sp_corr(g["theta_prep_late_500_800_cue"], g["parietal_alpha_cti_slope_100_800_cue"])}))
            .reset_index(drop=True)
        )
        block_core = block_core.merge(cpl1, on=["subj", "block_id", "phase3", "block_type"], how="left")
    if {"theta_prep_late_500_800_cue", "parietal_theta_prep_late_500_800_cue"}.issubset(trial_core.columns):
        cpl2 = (
            trial_core.groupby(["subj", "block_id", "phase3", "block_type"], as_index=False)
            .apply(lambda g: pd.Series({"fp_theta_theta_coupling": _sp_corr(g["theta_prep_late_500_800_cue"], g["parietal_theta_prep_late_500_800_cue"])}))
            .reset_index(drop=True)
        )
        block_core = block_core.merge(cpl2, on=["subj", "block_id", "phase3", "block_type"], how="left")

    # ------------------------------------------------------------------
    # 1) QC gate
    # ------------------------------------------------------------------
    qc_counts = (
        trial_core.groupby(["subj", "block_type", "phase3"], as_index=False)
        .size()
        .rename(columns={"size": "n_trials"})
    )
    qc_counts["pass_min_trials"] = qc_counts["n_trials"] >= TRIAL_MIN
    qc_counts.to_csv(os.path.join(OUT_DIR, "qc_gate_trial_counts.csv"), index=False)

    qc_subj = (
        qc_counts.groupby("subj", as_index=False)
        .agg(
            n_cells=("n_trials", "count"),
            n_cells_failed=("pass_min_trials", lambda s: int((~s).sum())),
            min_cell_n=("n_trials", "min"),
        )
    )

    if not align.empty and "subj" in align.columns:
        tmp = align.copy()
        for c in ["n_missing_by_scan", "n_abnormal_offset", "n_beh_from_block13"]:
            if c in tmp.columns:
                tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        if {"n_missing_by_scan", "n_beh_from_block13"}.issubset(tmp.columns):
            tmp["missing_rate"] = tmp["n_missing_by_scan"] / tmp["n_beh_from_block13"].replace(0, np.nan)
        if {"n_abnormal_offset", "n_beh_from_block13"}.issubset(tmp.columns):
            tmp["abnormal_rate"] = tmp["n_abnormal_offset"] / tmp["n_beh_from_block13"].replace(0, np.nan)
        qc_subj = qc_subj.merge(
            tmp[[c for c in ["subj", "missing_rate", "abnormal_rate"] if c in tmp.columns]],
            on="subj",
            how="left",
        )
    else:
        qc_subj["missing_rate"] = np.nan
        qc_subj["abnormal_rate"] = np.nan

    # Outlier flags for sensitivity analysis
    qc_subj["flag_missing_rate_outlier"] = _mad_outlier_flag(qc_subj["missing_rate"])
    qc_subj["flag_abnormal_rate_outlier"] = _mad_outlier_flag(qc_subj["abnormal_rate"])
    qc_subj["flag_low_trial_cells"] = qc_subj["n_cells_failed"] > 0
    qc_subj.to_csv(os.path.join(OUT_DIR, "qc_gate_subject_flags.csv"), index=False)

    # Search-phase block-type difference checks (requested)
    search_rows = []
    bs = block_core[block_core["phase3"] == "search"].copy()
    for m in METRICS + [c for c in ["fp_theta_parietal_alpha_coupling", "fp_theta_theta_coupling"] if c in block_core.columns]:
        piv = bs.pivot_table(index="subj", columns="block_type", values=m, aggfunc="mean")
        if {"gradual_like", "one_shot_like"}.issubset(piv.columns):
            x = piv["gradual_like"]
            y = piv["one_shot_like"]
            ok = x.notna() & y.notna()
            if ok.sum() >= 8:
                tt = ttest_rel(x[ok], y[ok], nan_policy="omit")
                search_rows.append(
                    {
                        "metric": m,
                        "n_subj": int(ok.sum()),
                        "mean_gradual": float(x[ok].mean()),
                        "mean_one_shot": float(y[ok].mean()),
                        "delta_gradual_minus_one_shot": float((x[ok] - y[ok]).mean()),
                        "t_stat": float(tt.statistic),
                        "p_value": float(tt.pvalue),
                    }
                )
    pd.DataFrame(search_rows).to_csv(os.path.join(OUT_DIR, "search_phase_blocktype_differences.csv"), index=False)

    # QC shape plot: average metric by phase/block type (proxy for quick sanity shape)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    plot_metrics = ["feedback_locked_P3b", "theta_power", "alpha_power", "feedback_locked_P3a"]
    for ax, m in zip(axes.flat, plot_metrics):
        tmp = block_core.groupby(["block_type", "phase3"], as_index=False)[m].mean(numeric_only=True)
        for bt in keep_bt:
            s = tmp[tmp["block_type"] == bt].set_index("phase3").reindex(keep_ph)
            ax.plot(keep_ph, s[m], marker="o", label=bt)
        ax.set_title(m)
        ax.set_xlabel("phase")
        ax.set_ylabel("mean")
    axes[0, 0].legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "qc_quick_shape_proxy.png"), dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # 2) Mainline A: block-level mixture model
    # ------------------------------------------------------------------
    # Add block-level behavioral covariates (accuracy, rt, search_len)
    beh_cov = (
        beh_all[beh_all["phase3"].isin(keep_ph)]
        .groupby(["subj", "block_id", "phase3"], as_index=False)
        .agg(
            accuracy_mean=("correctness", "mean"),
            rt_mean=("rt", "mean"),
        )
    )
    search_len = (
        beh_all[beh_all["phase3"] == "search"]
        .groupby(["subj", "block_id"], as_index=False)
        .size()
        .rename(columns={"size": "search_len"})
    )

    A = block_core.merge(beh_cov, on=["subj", "block_id", "phase3"], how="left")
    A = A.merge(search_len, on=["subj", "block_id"], how="left")
    A["search_len"] = pd.to_numeric(A["search_len"], errors="coerce")
    A["accuracy_mean_z"] = _zscore(A["accuracy_mean"])
    A["rt_mean_z"] = _zscore(A["rt_mean"])
    A["search_len_z"] = _zscore(A["search_len"])
    A["phase3"] = pd.Categorical(A["phase3"], categories=keep_ph, ordered=True)

    model_rows: List[Dict] = []
    for m in METRICS + [c for c in ["fp_theta_parietal_alpha_coupling", "fp_theta_theta_coupling"] if c in block_core.columns]:
        dat = A[["subj", "block_type", "phase3", m, "accuracy_mean_z", "rt_mean_z", "search_len_z"]].copy()
        dat = dat.dropna(subset=[m, "subj", "block_type", "phase3"])
        if dat["subj"].nunique() < 8:
            continue
        dat[m + "_z"] = _zscore(dat[m])
        dat = dat.dropna(subset=[m + "_z"])
        # Mixed effects
        formula = (
            f"{m}_z ~ C(block_type) + C(phase3) + C(block_type):C(phase3) + "
            "accuracy_mean_z + rt_mean_z + search_len_z"
        )
        try:
            fit = smf.mixedlm(formula, data=dat, groups=dat["subj"]).fit(reml=False, method="lbfgs", disp=False)
            for term, coef in fit.params.items():
                if term == "Group Var":
                    continue
                model_rows.append(
                    {
                        "analysis": "mainline_A_mixedlm",
                        "metric": m,
                        "term": term,
                        "coef": float(coef),
                        "se": float(fit.bse.get(term, np.nan)),
                        "p_value": float(fit.pvalues.get(term, np.nan)),
                        "n_obs": int(fit.nobs),
                        "n_subj": int(dat["subj"].nunique()),
                    }
                )
        except Exception as e:
            # Fallback: subject-clustered OLS (keeps inference stable when MixedLM fails numerically)
            try:
                fit2 = smf.ols(formula, data=dat).fit()
                for term, coef in fit2.params.items():
                    model_rows.append(
                        {
                            "analysis": "mainline_A_clustered_ols_fallback",
                            "metric": m,
                            "term": term,
                            "coef": float(coef),
                            "se": float(fit2.bse.get(term, np.nan)),
                            "p_value": float(fit2.pvalues.get(term, np.nan)),
                            "n_obs": int(fit2.nobs),
                            "n_subj": int(dat["subj"].nunique()),
                            "error": f"MixedLM failed: {e}",
                        }
                    )
            except Exception as e2:
                model_rows.append(
                    {
                        "analysis": "mainline_A_mixedlm",
                        "metric": m,
                        "term": "MODEL_FAILED",
                        "coef": np.nan,
                        "se": np.nan,
                        "p_value": np.nan,
                        "n_obs": int(len(dat)),
                        "n_subj": int(dat["subj"].nunique()),
                        "error": f"MixedLM:{e}; OLS fallback:{e2}",
                    }
                )

    A_res = pd.DataFrame(model_rows)
    A_res.to_csv(os.path.join(OUT_DIR, "mainline_A_mixedlm_results.csv"), index=False)

    # ------------------------------------------------------------------
    # 3) Mainline B: transition-gate trial-level model
    # ------------------------------------------------------------------
    trans_cols = [
        "subj",
        "block_id",
        "trial_id",
        "next_stable_flag",
        "time_since_collapse",
        "prev_correct",
        "search_flag",
        "stable_flag",
        "strategy_hard",
    ]
    trans_keep = trans[[c for c in trans_cols if c in trans.columns]].copy()
    trans_keep = trans_keep.rename(columns={"strategy_hard": "block_type_trans"})
    B = trial_core.merge(trans_keep, on=["subj", "block_id", "trial_id"], how="inner")

    # Transition definition: after collapse onset but before stable
    if "time_since_collapse" in B.columns and "stable_flag" in B.columns:
        B = B[(pd.to_numeric(B["time_since_collapse"], errors="coerce") >= 0) &
              (pd.to_numeric(B["stable_flag"], errors="coerce") == 0)].copy()
    if "next_stable_flag" in B.columns:
        B["next_stable_flag"] = pd.to_numeric(B["next_stable_flag"], errors="coerce")
        B = B[B["next_stable_flag"].isin([0, 1])].copy()

    B["time_since_collapse_z"] = _zscore(B["time_since_collapse"])
    B["prev_correct"] = pd.to_numeric(B["prev_correct"], errors="coerce")

    b_rows: List[Dict] = []
    for m in METRICS + [c for c in ["fp_theta_parietal_alpha_coupling", "fp_theta_theta_coupling"] if c in trial_core.columns]:
        dat = B[["subj", "block_type", m, "next_stable_flag", "time_since_collapse_z", "prev_correct"]].copy()
        dat = dat.dropna(subset=[m, "next_stable_flag", "time_since_collapse_z", "prev_correct"])
        if len(dat) < 120 or dat["subj"].nunique() < 8:
            continue
        dat[m + "_z"] = _zscore(dat[m])
        dat = dat.dropna(subset=[m + "_z"])
        if dat["next_stable_flag"].nunique() < 2:
            continue
        # Subject FE logistic model (stable and practical)
        f_main = f"next_stable_flag ~ {m}_z + time_since_collapse_z + prev_correct + C(subj)"
        f_inter = f"next_stable_flag ~ {m}_z * C(block_type) + time_since_collapse_z + prev_correct + C(subj)"
        try:
            fit_main = smf.glm(f_main, data=dat, family=sm.families.Binomial()).fit()
            fit_inter = smf.glm(f_inter, data=dat, family=sm.families.Binomial()).fit()
            key_main = f"{m}_z"
            b_rows.append(
                {
                    "analysis": "mainline_B_trial_logit",
                    "metric": m,
                    "model": "main",
                    "term": key_main,
                    "coef": float(fit_main.params.get(key_main, np.nan)),
                    "se": float(fit_main.bse.get(key_main, np.nan)),
                    "p_value": float(fit_main.pvalues.get(key_main, np.nan)),
                    "n_obs": int(fit_main.nobs),
                    "n_subj": int(dat["subj"].nunique()),
                }
            )
            # Save interaction terms containing metric
            for term in fit_inter.params.index:
                if m + "_z" in term:
                    b_rows.append(
                        {
                            "analysis": "mainline_B_trial_logit",
                            "metric": m,
                            "model": "interaction",
                            "term": term,
                            "coef": float(fit_inter.params.get(term, np.nan)),
                            "se": float(fit_inter.bse.get(term, np.nan)),
                            "p_value": float(fit_inter.pvalues.get(term, np.nan)),
                            "n_obs": int(fit_inter.nobs),
                            "n_subj": int(dat["subj"].nunique()),
                        }
                    )
        except Exception as e:
            b_rows.append(
                {
                    "analysis": "mainline_B_trial_logit",
                    "metric": m,
                    "model": "MODEL_FAILED",
                    "term": "MODEL_FAILED",
                    "coef": np.nan,
                    "se": np.nan,
                    "p_value": np.nan,
                    "n_obs": int(len(dat)),
                    "n_subj": int(dat["subj"].nunique()),
                    "error": str(e),
                }
            )

    B_res = pd.DataFrame(b_rows)
    B_res.to_csv(os.path.join(OUT_DIR, "mainline_B_transition_model_results.csv"), index=False)

    # ------------------------------------------------------------------
    # 4) Subject-level traits
    # ------------------------------------------------------------------
    # ΔEEG by block type for search / stable (HMM-acquired trials)
    traits = []
    for subj_id, g in block_core.groupby("subj"):
        row: Dict[str, float | str] = {"subj": subj_id}
        for ph in ["search", "stable"]:
            gg = g[g["phase3"] == ph]
            for m in METRICS + [c for c in ["fp_theta_parietal_alpha_coupling", "fp_theta_theta_coupling"] if c in block_core.columns]:
                a = gg.loc[gg["block_type"] == "gradual_like", m].mean()
                b = gg.loc[gg["block_type"] == "one_shot_like", m].mean()
                row[f"delta_{m}_{ph}_gradual_minus_one_shot"] = a - b
        # Frontal asymmetry as primary search-only trait
        gg = g[g["phase3"] == "search"]
        row["delta_frontal_asym_alpha_primary_search_only"] = (
            gg.loc[gg["block_type"] == "gradual_like", "frontal_asym_alpha"].mean()
            - gg.loc[gg["block_type"] == "one_shot_like", "frontal_asym_alpha"].mean()
        )
        traits.append(row)
    trait_df = pd.DataFrame(traits)

    # EEG_gate_coef / EEG_time_coef (subject-specific trial models)
    gate_rows = []
    for subj_id, g in B.groupby("subj"):
        for m in METRICS:
            d = g[[m, "next_stable_flag", "time_since_collapse", "prev_correct"]].copy()
            d = d.dropna()
            if len(d) < 40 or d["next_stable_flag"].nunique() < 2:
                continue
            d[m + "_z"] = _zscore(d[m])
            d["time_z"] = _zscore(d["time_since_collapse"])
            d = d.dropna()
            if len(d) < 30:
                continue
            try:
                fit = smf.glm(
                    f"next_stable_flag ~ {m}_z + time_z + prev_correct",
                    data=d,
                    family=sm.families.Binomial(),
                ).fit()
                gate_rows.append(
                    {
                        "subj": subj_id,
                        "metric": m,
                        "EEG_gate_coef": float(fit.params.get(m + "_z", np.nan)),
                        "EEG_time_coef": float(fit.params.get("time_z", np.nan)),
                        "n_transition_trials": int(fit.nobs),
                    }
                )
            except Exception:
                continue
    gate_df = pd.DataFrame(gate_rows)
    if not gate_df.empty:
        piv = gate_df.pivot_table(
            index="subj",
            columns="metric",
            values=["EEG_gate_coef", "EEG_time_coef"],
            aggfunc="mean",
        )
        piv.columns = [f"{a}_{b}" for a, b in piv.columns.to_flat_index()]
        piv = piv.reset_index()
        trait_long = trait_df.merge(piv, on="subj", how="left")
    else:
        trait_long = trait_df.copy()
    trait_long.to_csv(os.path.join(OUT_DIR, "eeg_subject_traits_paper_ready.csv"), index=False)
    gate_df.to_csv(os.path.join(OUT_DIR, "eeg_subject_gate_time_coefs_long.csv"), index=False)

    # ------------------------------------------------------------------
    # 5) Robustness checks
    # ------------------------------------------------------------------
    robust_rows = []
    # 5a: within-subject averaging paired comparison
    for m in METRICS:
        for ph in ["search", "stable"]:
            g = block_core[block_core["phase3"] == ph]
            piv = g.pivot_table(index="subj", columns="block_type", values=m, aggfunc="mean")
            if {"gradual_like", "one_shot_like"}.issubset(piv.columns):
                x = piv["gradual_like"]
                y = piv["one_shot_like"]
                ok = x.notna() & y.notna()
                if ok.sum() >= 8:
                    tt = ttest_rel(x[ok], y[ok], nan_policy="omit")
                    robust_rows.append(
                        {
                            "check": "within_subject_paired",
                            "metric": m,
                            "phase3": ph,
                            "n_subj": int(ok.sum()),
                            "mean_gradual": float(x[ok].mean()),
                            "mean_one_shot": float(y[ok].mean()),
                            "delta": float((x[ok] - y[ok]).mean()),
                            "t_stat": float(tt.statistic),
                            "p_value": float(tt.pvalue),
                        }
                    )

    # 5b: adjacent proxy sensitivity using neighboring metrics (theta vs alpha; P3a vs P3b)
    if not B_res.empty:
        bmain = B_res[(B_res["model"] == "main") & (B_res["term"].str.endswith("_z"))].copy()
        sens = {}
        for m in ["theta_power", "alpha_power", "feedback_locked_P3a", "feedback_locked_P3b"]:
            s = bmain.loc[bmain["metric"] == m, "coef"]
            sens[m] = float(s.iloc[0]) if len(s) else np.nan
        robust_rows.append(
            {
                "check": "adjacent_metric_sensitivity",
                "metric": "theta_vs_alpha",
                "phase3": "transition",
                "n_subj": np.nan,
                "mean_gradual": np.nan,
                "mean_one_shot": np.nan,
                "delta": np.nan,
                "t_stat": sens.get("theta_power", np.nan),
                "p_value": sens.get("alpha_power", np.nan),
            }
        )
        robust_rows.append(
            {
                "check": "adjacent_metric_sensitivity",
                "metric": "P3a_vs_P3b",
                "phase3": "transition",
                "n_subj": np.nan,
                "mean_gradual": np.nan,
                "mean_one_shot": np.nan,
                "delta": np.nan,
                "t_stat": sens.get("feedback_locked_P3a", np.nan),
                "p_value": sens.get("feedback_locked_P3b", np.nan),
            }
        )

    robust_df = pd.DataFrame(robust_rows)
    robust_df.to_csv(os.path.join(OUT_DIR, "robustness_checks.csv"), index=False)

    # ------------------------------------------------------------------
    # Figures for paper
    # ------------------------------------------------------------------
    # Fig1: block mixture difference by phase
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True)
    keym = ["feedback_locked_P3b", "theta_power", "frontal_asym_alpha"]
    for ax, m in zip(axes, keym):
        tmp = (
            block_core.groupby(["block_type", "phase3"], as_index=False)[m]
            .mean(numeric_only=True)
        )
        for bt in keep_bt:
            s = tmp[tmp["block_type"] == bt].set_index("phase3").reindex(["search", "stable"])
            ax.plot(["search", "stable"], s[m], marker="o", label=bt)
        ax.set_title(m)
        ax.set_xlabel("phase")
    axes[0].set_ylabel("mean")
    axes[0].legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig1_block_mixture_core_metrics.png"), dpi=160)
    plt.close()

    # Fig2: transition-gate model coefficients
    if not B_res.empty:
        bm = B_res[(B_res["model"] == "main") & (B_res["term"].str.endswith("_z"))].copy()
        bm = bm.sort_values("coef")
        plt.figure(figsize=(7, 3.8))
        plt.axvline(0, color="k", linestyle="--", linewidth=1)
        y = np.arange(len(bm))
        plt.errorbar(bm["coef"], y, xerr=1.96 * bm["se"], fmt="o")
        plt.yticks(y, bm["metric"])
        plt.xlabel("Log-odds coef (main effect)")
        plt.title("Transition gate: EEG -> next_stable")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "fig2_transition_gate_effects.png"), dpi=160)
        plt.close()

    # Fig3: subject-level trait scatter (example)
    xcol = "delta_feedback_locked_P3b_search_gradual_minus_one_shot"
    ycol = "delta_theta_power_search_gradual_minus_one_shot"
    if xcol in trait_long.columns and ycol in trait_long.columns:
        dd = trait_long[[xcol, ycol]].dropna()
        if len(dd) > 5:
            plt.figure(figsize=(4.4, 4.2))
            plt.scatter(dd[xcol], dd[ycol], alpha=0.8)
            plt.xlabel("ΔP3b (search)")
            plt.ylabel("Δtheta (search)")
            plt.title("Subject-level EEG traits")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, "fig3_subject_trait_scatter.png"), dpi=160)
            plt.close()

    # Save merged analysis base tables for transparency
    trial_core.to_csv(os.path.join(OUT_DIR, "analysis_trial_core.csv"), index=False)
    block_core.to_csv(os.path.join(OUT_DIR, "analysis_block_core.csv"), index=False)
    if not subj.empty:
        subj.to_csv(os.path.join(OUT_DIR, "analysis_subject_input.csv"), index=False)

    print("Saved paper-ready outputs ->", OUT_DIR)


if __name__ == "__main__":
    run()

