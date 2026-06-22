import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm


warnings.filterwarnings("ignore")

BASE = "base_dir"
ALL_TRIAL = os.path.join(BASE, "trials_trialwise", "trialwise", "all_subjects_trialwise.csv")
HMM_TRIAL = os.path.join(BASE, "trials_trialwise", "hmm_long_tables", "hmm_trial_long.csv")
TRANS_TRIAL = os.path.join(BASE, "trials_trialwise", "hmm_mixture", "hmm_mixture_transition_trials_used.csv")

OUT_DIR = os.path.join(BASE, "trials_trialwise", "behav_paper_results", "pure_behavior_suite")
LMM_OUT = os.path.join(OUT_DIR, "pure_behavior_lmm_results.csv")
GLMM_OUT = os.path.join(OUT_DIR, "pure_behavior_logistic_mixed_results.csv")
MODEL_LOG_OUT = os.path.join(OUT_DIR, "pure_behavior_model_run_log.csv")


def to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def fit_lmm(data: pd.DataFrame, formula: str, model_tag: str) -> tuple[pd.DataFrame, dict]:
    d = data.copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    info = {"model_tag": model_tag, "formula": formula, "n_obs": len(d), "n_subj": d["subj"].nunique()}
    if len(d) < 30 or d["subj"].nunique() < 8:
        info["status"] = "skipped_too_few_data"
        return pd.DataFrame(), info
    try:
        fit = smf.mixedlm(formula, d, groups=d["subj"]).fit(reml=False, method="lbfgs", maxiter=300)
        tab = pd.DataFrame(
            {
                "model_tag": model_tag,
                "formula": formula,
                "n_obs": len(d),
                "n_subj": d["subj"].nunique(),
                "term": fit.params.index,
                "estimate": fit.params.values,
                "se": fit.bse.values,
                "z_or_t": fit.tvalues.values,
                "p_value": fit.pvalues.values,
                "aic": fit.aic,
                "bic": fit.bic,
            }
        )
        info["status"] = "ok"
        return tab, info
    except Exception as e:
        info["status"] = f"failed:{type(e).__name__}"
        return pd.DataFrame(), info


def fit_logistic_mixed(data: pd.DataFrame, formula: str, model_tag: str) -> tuple[pd.DataFrame, dict]:
    # BinomialBayesMixedGLM: random intercept by subject
    d = data.copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    info = {"model_tag": model_tag, "formula": formula, "n_obs": len(d), "n_subj": d["subj"].nunique()}
    if len(d) < 30 or d["subj"].nunique() < 8 or d["correctness"].nunique() < 2:
        info["status"] = "skipped_too_few_data"
        return pd.DataFrame(), info
    try:
        vc = {"subj_re": "0 + C(subj)"}
        model = sm.BinomialBayesMixedGLM.from_formula(formula, vc, d)
        fit = model.fit_vb()
        fe_names = model.exog_names
        fe_mean = fit.fe_mean
        fe_sd = fit.fe_sd
        z_approx = fe_mean / fe_sd
        p_approx = 2 * (1 - norm.cdf(np.abs(z_approx)))
        tab = pd.DataFrame(
            {
                "model_tag": model_tag,
                "formula": formula,
                "n_obs": len(d),
                "n_subj": d["subj"].nunique(),
                "term": fe_names,
                "estimate_logit": fe_mean,
                "posterior_sd": fe_sd,
                "z_approx": z_approx,
                "p_approx": p_approx,
            }
        )
        info["status"] = "ok"
        return tab, info
    except Exception as e:
        info["status"] = f"failed:{type(e).__name__}"
        return pd.DataFrame(), info


def build_master_table() -> pd.DataFrame:
    a = pd.read_csv(ALL_TRIAL)
    h = pd.read_csv(HMM_TRIAL)
    t = pd.read_csv(TRANS_TRIAL)

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
    t = to_num(
        t,
        [
            "block_id",
            "trial_id",
            "correctness",
            "rt",
            "prev_correct",
            "time_since_collapse",
            "collapsed_flag",
            "search_flag",
            "stable_flag",
            "is_rule_switch_block",
        ],
    )

    # Merge core trialwise + HMM flags
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
                "current_streak",
                "collapsed_flag",
            ]
        ],
        on=["subj", "block_id", "trial_id"],
        how="left",
    )

    # Add transition/history variables
    m = m.merge(
        t[
            [
                "subj",
                "block_id",
                "trial_id",
                "prev_correct",
                "time_since_collapse",
                "ll_diff_per_trial_block",
                "strategy_hard",
            ]
        ],
        on=["subj", "block_id", "trial_id"],
        how="left",
    )

    # Harmonize trial index and rt/correctness
    if "trial_in_block" not in m.columns:
        m["trial_in_block"] = m["trial_since_rule_switch"]
    m["trial_in_block"] = pd.to_numeric(m["trial_in_block"], errors="coerce")
    m["rt"] = pd.to_numeric(m["rt"], errors="coerce")
    m["correctness"] = pd.to_numeric(m["correctness"], errors="coerce")
    m = m[(m["rt"] > 0) & m["correctness"].isin([0, 1])].copy()
    m["log_rt"] = np.log(m["rt"])

    # Per-block first-correct latency and search length
    fc = (
        m[m["first_correct_flag"] == 1]
        .groupby(["subj", "block_id"], as_index=False)["trial_in_block"]
        .min()
        .rename(columns={"trial_in_block": "first_correct_latency"})
    )
    block_meta = (
        m.groupby(["subj", "block_id"], as_index=False)
        .agg(rule_level=("rule_level", "first"), rule_type=("rule_type", "first"), is_rule_switch_block=("is_rule_switch_block", "first"))
    )
    block_meta = block_meta.merge(fc, on=["subj", "block_id"], how="left")
    block_meta["search_len"] = block_meta["first_correct_latency"] - 1

    # Shift type from previous block rule_level
    block_meta = block_meta.sort_values(["subj", "block_id"]).copy()
    block_meta["prev_rule_level"] = block_meta.groupby("subj")["rule_level"].shift(1)
    block_meta["shift_type"] = np.where(
        block_meta["prev_rule_level"].isna(),
        "missing",
        np.where(block_meta["prev_rule_level"] == block_meta["rule_level"], "within_level", "cross_level"),
    )
    block_meta["shift_4"] = (
        block_meta["prev_rule_level"].astype(str).str.replace("nan", "", regex=False)
        + "->"
        + block_meta["rule_level"].astype(str).str.replace("nan", "", regex=False)
    )

    m = m.merge(
        block_meta[["subj", "block_id", "first_correct_latency", "search_len", "shift_type", "shift_4"]],
        on=["subj", "block_id"],
        how="left",
    )

    # Position bins
    m["position_bin"] = np.where(
        m["trial_in_block"] == 1,
        "t1",
        np.where(m["trial_in_block"].isin([2, 3]), "t2_3", "t4plus"),
    )

    # Three-phase definition
    m["phase3"] = np.where(
        m["stable_flag"] == 1,
        "stable",
        np.where(
            (m["post_first_correct_flag"] == 1) | (m["first_correct_flag"] == 1),
            "first_correct_to_pre_stable",
            "pre_first_correct",
        ),
    )

    # Search feedback type
    m["search_feedback_type"] = np.where(
        (m["search_flag"] == 1) & (m["correctness"] == 0),
        "search_wrong",
        np.where(
            (m["first_correct_flag"] == 1) & (m["correctness"] == 1),
            "first_correct",
            np.where(
                (m["search_flag"] == 1) & (m["correctness"] == 1) & (m["first_correct_flag"] != 1),
                "search_correct_nonterminal",
                "missing",
            ),
        ),
    )

    # Next-trial correctness for lag effect
    m = m.sort_values(["subj", "block_id", "trial_in_block"]).copy()
    m["next_correctness"] = m.groupby(["subj", "block_id"])["correctness"].shift(-1)

    # Collapse bins
    m["post_collapse"] = np.where(m["time_since_collapse"].notna() & (m["time_since_collapse"] >= 0), 1, 0)
    return m


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = build_master_table()

    lmm_rows = []
    glmm_rows = []
    logs = []

    # Group 1: rule type (4 levels)
    d1 = df[df["rule_type"].notna()].copy()
    t, info = fit_lmm(d1[["subj", "log_rt", "rule_type"]], "log_rt ~ C(rule_type)", "G1_rule_type_logRT")
    lmm_rows.append(t); logs.append(info)
    t, info = fit_logistic_mixed(d1[["subj", "correctness", "rule_type"]], "correctness ~ C(rule_type)", "G1_rule_type_correct")
    glmm_rows.append(t); logs.append(info)

    # Group 2: shift type (within vs cross)
    d2_block = df[["subj", "block_id", "shift_type", "search_len", "first_correct_latency"]].drop_duplicates()
    d2_block = d2_block[d2_block["shift_type"].isin(["within_level", "cross_level"])].copy()
    t, info = fit_lmm(d2_block[["subj", "search_len", "shift_type"]], "search_len ~ C(shift_type)", "G2_shift_search_len")
    lmm_rows.append(t); logs.append(info)
    t, info = fit_lmm(
        d2_block[["subj", "first_correct_latency", "shift_type"]],
        "first_correct_latency ~ C(shift_type)",
        "G2_shift_first_correct_latency",
    )
    lmm_rows.append(t); logs.append(info)

    d2_trial = df[df["shift_type"].isin(["within_level", "cross_level"])].copy()
    t, info = fit_lmm(d2_trial[["subj", "log_rt", "shift_type"]], "log_rt ~ C(shift_type)", "G2_shift_logRT")
    lmm_rows.append(t); logs.append(info)
    t, info = fit_logistic_mixed(d2_trial[["subj", "correctness", "shift_type"]], "correctness ~ C(shift_type)", "G2_shift_correct")
    glmm_rows.append(t); logs.append(info)

    # Optional split into 4-level shift
    d2_4 = d2_trial[d2_trial["shift_4"].isin(["global->global", "global->local", "local->global", "local->local"])].copy()
    t, info = fit_lmm(d2_4[["subj", "log_rt", "shift_4"]], "log_rt ~ C(shift_4)", "G2_shift4_logRT")
    lmm_rows.append(t); logs.append(info)
    t, info = fit_logistic_mixed(d2_4[["subj", "correctness", "shift_4"]], "correctness ~ C(shift_4)", "G2_shift4_correct")
    glmm_rows.append(t); logs.append(info)

    # Group 3: position after switch
    d3 = df[df["position_bin"].isin(["t1", "t2_3", "t4plus"])].copy()
    t, info = fit_lmm(d3[["subj", "log_rt", "position_bin"]], "log_rt ~ C(position_bin)", "G3_position_logRT")
    lmm_rows.append(t); logs.append(info)
    t, info = fit_logistic_mixed(
        d3[["subj", "correctness", "position_bin"]], "correctness ~ C(position_bin)", "G3_position_correct"
    )
    glmm_rows.append(t); logs.append(info)

    # Group 4: phase3 comparison
    d4 = df[df["phase3"].isin(["pre_first_correct", "first_correct_to_pre_stable", "stable"])].copy()
    t, info = fit_lmm(d4[["subj", "log_rt", "phase3"]], "log_rt ~ C(phase3)", "G4_phase3_logRT")
    lmm_rows.append(t); logs.append(info)
    t, info = fit_logistic_mixed(d4[["subj", "correctness", "phase3"]], "correctness ~ C(phase3)", "G4_phase3_correct")
    glmm_rows.append(t); logs.append(info)

    # Group 5: search internal wrong/correct/first-correct
    d5 = df[df["search_feedback_type"].isin(["search_wrong", "search_correct_nonterminal", "first_correct"])].copy()
    t, info = fit_lmm(d5[["subj", "log_rt", "search_feedback_type"]], "log_rt ~ C(search_feedback_type)", "G5_search_feedback_logRT")
    lmm_rows.append(t); logs.append(info)
    d5n = d5[d5["next_correctness"].isin([0, 1])].copy()
    d5n = d5n[["subj", "next_correctness", "search_feedback_type"]].rename(columns={"next_correctness": "correctness"})
    t, info = fit_logistic_mixed(
        d5n[["subj", "correctness", "search_feedback_type"]],
        "correctness ~ C(search_feedback_type)",
        "G5_search_feedback_next_correct",
    )
    glmm_rows.append(t); logs.append(info)

    # Group 6: feedback history (prev_correct)
    d6 = df[df["prev_correct"].notna()].copy()
    # transition table stores a transformed prev_correct; recover binary direction.
    d6["prev_correct"] = (pd.to_numeric(d6["prev_correct"], errors="coerce") > 0).astype(int).astype(str)
    t, info = fit_lmm(d6[["subj", "log_rt", "prev_correct"]], "log_rt ~ C(prev_correct)", "G6_prev_correct_logRT")
    lmm_rows.append(t); logs.append(info)
    t, info = fit_logistic_mixed(d6[["subj", "correctness", "prev_correct"]], "correctness ~ C(prev_correct)", "G6_prev_correct_correct")
    glmm_rows.append(t); logs.append(info)

    # Group 7: collapse effects
    d7 = df.copy()
    d7["post_collapse"] = d7["post_collapse"].astype(int).astype(str)
    t, info = fit_lmm(d7[["subj", "log_rt", "post_collapse"]], "log_rt ~ C(post_collapse)", "G7_post_collapse_logRT")
    lmm_rows.append(t); logs.append(info)
    t, info = fit_logistic_mixed(d7[["subj", "correctness", "post_collapse"]], "correctness ~ C(post_collapse)", "G7_post_collapse_correct")
    glmm_rows.append(t); logs.append(info)

    d7t = df[df["time_since_collapse"].notna()].copy()
    t, info = fit_lmm(
        d7t[["subj", "log_rt", "time_since_collapse"]],
        "log_rt ~ time_since_collapse",
        "G7_time_since_collapse_logRT",
    )
    lmm_rows.append(t); logs.append(info)

    lmm_out = pd.concat([x for x in lmm_rows if not x.empty], ignore_index=True) if lmm_rows else pd.DataFrame()
    glmm_out = pd.concat([x for x in glmm_rows if not x.empty], ignore_index=True) if glmm_rows else pd.DataFrame()
    log_out = pd.DataFrame(logs)

    lmm_out.to_csv(LMM_OUT, index=False)
    glmm_out.to_csv(GLMM_OUT, index=False)
    log_out.to_csv(MODEL_LOG_OUT, index=False)

    print("saved ->", LMM_OUT)
    print("saved ->", GLMM_OUT)
    print("saved ->", MODEL_LOG_OUT)
    print("n_lmm_rows:", len(lmm_out))
    print("n_glmm_rows:", len(glmm_out))
    print(log_out.to_string(index=False))


if __name__ == "__main__":
    main()
