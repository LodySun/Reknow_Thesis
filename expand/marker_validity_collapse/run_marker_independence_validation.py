import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


BASE = "base_dir"
SOL = os.path.join(BASE, "trials_trialwise", "1s_comp", "eeg_paper_results", "solidity")
EXP = os.path.join(BASE, "trials_trialwise", "1s_comp", "eeg_paper_results", "expand", "marker_validity_collapse")

POST_CSV = os.path.join(SOL, "hmm_unified_trial_posteriors.csv")
HMM_LONG_CSV = os.path.join(BASE, "trials_trialwise", "hmm_long_tables", "hmm_trial_long.csv")
STAGE_CSV = os.path.join(SOL, "eeg_acq_sequence_stage_means_long.csv")
BEH_CSV = os.path.join(BASE, "trials_trialwise", "all_subjects_trialwise.csv")

OUT_BLOCK = os.path.join(EXP, "marker_independence_block_table.csv")
OUT_MODELS = os.path.join(EXP, "marker_independence_models.csv")
OUT_META = os.path.join(EXP, "marker_independence_marker_sources.csv")


def _safe_num(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _clustered_ols(data: pd.DataFrame, formula: str, cluster_col: str, model_name: str) -> pd.DataFrame:
    d = data.copy().dropna()
    if len(d) < 30:
        return pd.DataFrame(
            [
                {
                    "model": model_name,
                    "term": "NA",
                    "n": int(len(d)),
                    "beta": np.nan,
                    "se_cluster": np.nan,
                    "t": np.nan,
                    "p": np.nan,
                    "note": "insufficient_n",
                }
            ]
        )
    try:
        fit = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d[cluster_col]})
        rows = []
        for term in fit.params.index:
            if term == "Intercept":
                continue
            rows.append(
                {
                    "model": model_name,
                    "term": term,
                    "n": int(len(d)),
                    "beta": float(fit.params[term]),
                    "se_cluster": float(fit.bse[term]),
                    "t": float(fit.tvalues[term]),
                    "p": float(fit.pvalues[term]),
                    "note": "",
                }
            )
        return pd.DataFrame(rows)
    except Exception as e:
        return pd.DataFrame(
            [
                {
                    "model": model_name,
                    "term": "NA",
                    "n": int(len(d)),
                    "beta": np.nan,
                    "se_cluster": np.nan,
                    "t": np.nan,
                    "p": np.nan,
                    "note": f"fit_failed: {type(e).__name__}",
                }
            ]
        )


def main():
    os.makedirs(EXP, exist_ok=True)
    post = pd.read_csv(POST_CSV)
    hlong = pd.read_csv(HMM_LONG_CSV)
    stage = pd.read_csv(STAGE_CSV)
    beh = pd.read_csv(BEH_CSV)

    # Harmonize keys
    for df in [post, hlong, stage, beh]:
        df["subj"] = df["subj"].astype(str)
        df["block_id"] = df["block_id"].astype(str)
    _safe_num(post, ["trial_id", "trial_index_1based", "first_correct_trial", "acquisition_trial_core", "set_index"])
    _safe_num(hlong, ["trial_id", "trial_in_block", "candidates_before", "candidates_after"])
    _safe_num(stage, ["value_mean", "set_index"])
    _safe_num(beh, ["trial_id", "trial_since_rule_switch", "correctness", "rt"])
    stage["category"] = stage["category"].astype(str)
    stage["feature"] = stage["feature"].astype(str)

    # 1) Behavioral marker from HMM (first-correct and core) - marker only
    bmark = (
        post.groupby(["subj", "block_id"], as_index=False)
        .agg(
            first_correct_trial=("first_correct_trial", "first"),
            acquisition_trial_core=("acquisition_trial_core", "first"),
            set_index=("set_index", "first"),
        )
        .dropna(subset=["first_correct_trial", "acquisition_trial_core"])
    )
    bmark["acq_lag_core"] = bmark["acquisition_trial_core"] - bmark["first_correct_trial"]

    # 2) Logical collapse marker from rule-space pruning (independent from unified HMM posterior)
    h = hlong[["subj", "block_id", "trial_in_block", "candidates_after"]].dropna().copy()
    h["collapsed_logic_flag"] = (pd.to_numeric(h["candidates_after"], errors="coerce") <= 1).astype(int)
    collapse = (
        h[h["collapsed_logic_flag"] == 1]
        .groupby(["subj", "block_id"], as_index=False)["trial_in_block"]
        .min()
        .rename(columns={"trial_in_block": "collapse_trial_logic"})
    )
    bmark = bmark.merge(collapse, on=["subj", "block_id"], how="left")
    bmark["collapse_to_fc"] = bmark["first_correct_trial"] - bmark["collapse_trial_logic"]
    bmark["collapse_to_core"] = bmark["acquisition_trial_core"] - bmark["collapse_trial_logic"]

    # 3) Neural marker from EEG stage table at transition first-correct (not used by HMM fitting)
    fcs = stage[stage["category"] == "transition_first_correct"].copy()
    piv_fc = (
        fcs.pivot_table(
            index=["subj", "block_id"],
            columns="feature",
            values="value_mean",
            aggfunc="first",
        )
        .reset_index()
    )
    piv_fc.columns.name = None
    keep_feats = [c for c in ["feedback_locked_FRN", "feedback_locked_P3b", "feedback_locked_P3a"] if c in piv_fc.columns]
    nmark = piv_fc[["subj", "block_id"] + keep_feats].copy()

    # 4) Subsequent acquired-state behavior from raw behavior table (independent target)
    # Define post-core window (core .. core+4), does not enter core criterion itself.
    pb = post[["subj", "block_id", "trial_id", "trial_index_1based", "acquisition_trial_core"]].dropna().copy()
    bb = beh[["subj", "block_id", "trial_id", "trial_since_rule_switch", "correctness", "rt"]].copy()
    m = pb.merge(bb, on=["subj", "block_id", "trial_id"], how="inner")
    m["trial_idx"] = pd.to_numeric(m["trial_index_1based"], errors="coerce")
    m["core"] = pd.to_numeric(m["acquisition_trial_core"], errors="coerce")
    m["is_post_core_5"] = ((m["trial_idx"] >= m["core"]) & (m["trial_idx"] <= (m["core"] + 4))).astype(int)
    m["is_search_pre_core"] = (m["trial_idx"] < m["core"]).astype(int)

    post5 = (
        m[m["is_post_core_5"] == 1]
        .groupby(["subj", "block_id"], as_index=False)
        .agg(post_core_acc5=("correctness", "mean"), post_core_rt5=("rt", "mean"))
    )
    pre = (
        m[m["is_search_pre_core"] == 1]
        .groupby(["subj", "block_id"], as_index=False)
        .agg(pre_core_rt=("rt", "mean"))
    )
    stab = post5.merge(pre, on=["subj", "block_id"], how="left")
    stab["rt_change_post_minus_pre"] = stab["post_core_rt5"] - stab["pre_core_rt"]

    # Merge all markers
    block = bmark.merge(nmark, on=["subj", "block_id"], how="left").merge(stab, on=["subj", "block_id"], how="left")
    block.to_csv(OUT_BLOCK, index=False)

    # Cross-prediction models across independent marker types
    model_rows = []
    # Logical -> behavioral marker relation
    model_rows.append(
        _clustered_ols(
            block[["subj", "acq_lag_core", "collapse_to_fc", "set_index"]].copy(),
            "acq_lag_core ~ collapse_to_fc + set_index",
            "subj",
            "M1_behavior_marker_on_logical_marker",
        )
    )

    # Neural + logical -> subsequent acquired-state behavior
    need_cols = ["subj", "post_core_acc5", "rt_change_post_minus_pre", "set_index", "collapse_to_fc"] + keep_feats
    d2 = block[need_cols].copy()
    if "feedback_locked_FRN" in d2.columns and "feedback_locked_P3b" in d2.columns:
        model_rows.append(
            _clustered_ols(
                d2[["subj", "post_core_acc5", "feedback_locked_FRN", "feedback_locked_P3b", "collapse_to_fc", "set_index"]].copy(),
                "post_core_acc5 ~ feedback_locked_FRN + feedback_locked_P3b + collapse_to_fc + set_index",
                "subj",
                "M2_acquired_acc_on_neural_and_logical",
            )
        )
        model_rows.append(
            _clustered_ols(
                d2[["subj", "rt_change_post_minus_pre", "feedback_locked_FRN", "feedback_locked_P3b", "collapse_to_fc", "set_index"]].copy(),
                "rt_change_post_minus_pre ~ feedback_locked_FRN + feedback_locked_P3b + collapse_to_fc + set_index",
                "subj",
                "M3_acquired_rtchange_on_neural_and_logical",
            )
        )

    models = pd.concat(model_rows, ignore_index=True) if model_rows else pd.DataFrame()
    models.to_csv(OUT_MODELS, index=False)

    # Marker source declaration table for manuscript transparency
    meta = pd.DataFrame(
        [
            {
                "marker": "collapse_trial_logic",
                "domain": "logical rule-space",
                "source_table": "hmm_trial_long.csv",
                "construction": "first trial with candidates_after <= 1",
                "independence_note": "does not use unified HMM posterior probabilities",
            },
            {
                "marker": "feedback_locked_FRN/P3b at transition first-correct",
                "domain": "neural",
                "source_table": "eeg_acq_sequence_stage_means_long.csv",
                "construction": "transition_first_correct stage mean amplitudes",
                "independence_note": "ERP features are not used in HMM fitting",
            },
            {
                "marker": "post_core_acc5 / rt_change_post_minus_pre",
                "domain": "subsequent behavior",
                "source_table": "all_subjects_trialwise.csv + hmm_unified_trial_posteriors.csv",
                "construction": "behavioral outcomes in window after core",
                "independence_note": "post-core acquired-state behavior is not part of the core criterion itself",
            },
        ]
    )
    meta.to_csv(OUT_META, index=False)

    print(f"saved: {OUT_BLOCK}")
    print(f"saved: {OUT_MODELS}")
    print(f"saved: {OUT_META}")
    if not models.empty:
        print(models.to_string(index=False))


if __name__ == "__main__":
    main()
