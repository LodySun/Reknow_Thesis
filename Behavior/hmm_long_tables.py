import os
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from HMM import _prepare_sequences, fit_model_with_restarts, run_subject_cv, score_loglik
    from trialwise_parser import build_trialwise_dataframe
    from trait_extractor import _search_inference_dynamics
except ModuleNotFoundError:
    from codes.behav.HMM import _prepare_sequences, fit_model_with_restarts, run_subject_cv, score_loglik
    from codes.behav.trialwise_parser import build_trialwise_dataframe
    from codes.behav.trait_extractor import _search_inference_dynamics

LOGS_DIR = "/Users/lodysun/Desktop/Thesis/logs"
OUT_DIR = "/Users/lodysun/Desktop/Thesis/trials_trialwise"
N_SUBJECTS = 34


def _add_trial_derived_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["trial_in_block"] = pd.to_numeric(out["trial_id"], errors="coerce")
    out["block_index_global"] = pd.to_numeric(out["block_id"], errors="coerce")

    # first-correct markers
    first_correct = (
        out[out["correctness"] == 1]
        .groupby("block_id", as_index=False)["trial_in_block"]
        .min()
        .rename(columns={"trial_in_block": "first_correct_idx"})
    )
    out = out.merge(first_correct, on="block_id", how="left")
    out["first_correct_flag"] = (out["trial_in_block"] == out["first_correct_idx"]).astype(float)
    out["post_first_correct_flag"] = (
        out["first_correct_idx"].notna() & (out["trial_in_block"] > out["first_correct_idx"])
    ).astype(float)

    # current streak (consecutive correct up to current trial)
    sorted_out = out.sort_values(["block_index_global", "trial_in_block"]).copy()
    sorted_out["current_streak"] = 0
    for _, idx in sorted_out.groupby("block_id").groups.items():
        c = pd.to_numeric(sorted_out.loc[idx, "correctness"], errors="coerce").fillna(0).astype(int).to_numpy()
        s = np.zeros(len(c), dtype=int)
        run = 0
        for i, v in enumerate(c):
            run = run + 1 if v == 1 else 0
            s[i] = run
        sorted_out.loc[idx, "current_streak"] = s
    out = sorted_out

    out["search_flag"] = (out["phase"] == "search").astype(float)
    out["stable_flag"] = (out["phase"] != "search").astype(float)
    out["missing_correctness_flag"] = out["correctness"].isna().astype(float)
    out["abnormal_response_flag"] = (
        pd.to_numeric(out["response_card"], errors="coerce").abs().isin([1, 2, 3, 4]).map(lambda x: 0.0 if x else 1.0)
    )
    return out


def _build_block_long(trial_long: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for (subj, block_id), bdf in trial_long.groupby(["subj", "block_id"]):
        bdf = bdf.sort_values("trial_in_block")
        n_trials = int(len(bdf))
        acc = float(pd.to_numeric(bdf["correctness"], errors="coerce").mean())
        rt_mean = float(pd.to_numeric(bdf["rt"], errors="coerce").mean())
        rt_median = float(pd.to_numeric(bdf["rt"], errors="coerce").median())

        fc_series = pd.to_numeric(
            bdf.loc[pd.to_numeric(bdf["first_correct_flag"], errors="coerce") == 1, "trial_in_block"],
            errors="coerce",
        )
        first_correct_idx = float(fc_series.iloc[0]) if len(fc_series) > 0 else np.nan
        post_length = float(n_trials - first_correct_idx + 1) if pd.notna(first_correct_idx) else np.nan

        search_len_k = float((bdf["search_flag"] == 1).sum())
        collapse_series = bdf[bdf["candidates_after"] == 1]
        collapse_onset_idx = (
            float(pd.to_numeric(collapse_series["trial_in_block"], errors="coerce").iloc[0])
            if not collapse_series.empty
            else np.nan
        )
        stable_series = bdf[bdf["stable_flag"] == 1]
        stable_onset_idx = (
            float(pd.to_numeric(stable_series["trial_in_block"], errors="coerce").iloc[0])
            if not stable_series.empty
            else np.nan
        )
        collapse_to_stable = (
            float(stable_onset_idx - collapse_onset_idx)
            if pd.notna(collapse_onset_idx) and pd.notna(stable_onset_idx) and (stable_onset_idx - collapse_onset_idx) >= 0
            else np.nan
        )

        post_collapse_fail_rate = (
            float((pd.to_numeric(collapse_series["correctness"], errors="coerce") == 0).mean())
            if not collapse_series.empty
            else np.nan
        )

        collapse_reversal_count = np.nan
        if pd.notna(collapse_onset_idx):
            after = bdf[pd.to_numeric(bdf["trial_in_block"], errors="coerce") >= collapse_onset_idx]
            nb = pd.to_numeric(after["candidates_after"], errors="coerce")
            collapse_reversal_count = float((nb.diff() > 0).sum()) if len(nb) >= 2 else 0.0

        conf_mask = bdf["eligible_confirmatory_trial"] == 1
        confirmatory_bias_block = (
            float(pd.to_numeric(bdf.loc[conf_mask, "confirmatory_choice"], errors="coerce").dropna().mean())
            if conf_mask.any()
            else np.nan
        )
        information_efficiency_block = (
            float(pd.to_numeric(bdf.loc[conf_mask, "information_efficiency"], errors="coerce").dropna().mean())
            if conf_mask.any()
            else np.nan
        )

        rows.append(
            {
                "subj": subj,
                "block_id": block_id,
                "block_index_global": float(pd.to_numeric(bdf["block_index_global"], errors="coerce").iloc[0]),
                "n_trials": float(n_trials),
                "accuracy_block": acc,
                "rt_median_block": rt_median,
                "rt_mean_block": rt_mean,
                "first_correct_idx": first_correct_idx,
                "post_length": post_length,
                "search_len_k": search_len_k,
                "collapse_onset_idx": collapse_onset_idx,
                "stable_onset_idx": stable_onset_idx,
                "collapse_to_stable": collapse_to_stable,
                "post_collapse_fail_rate": post_collapse_fail_rate,
                "collapse_reversal_count": collapse_reversal_count,
                "confirmatory_bias_block": confirmatory_bias_block,
                "information_efficiency_block": information_efficiency_block,
                "missing_correctness_rate": float(pd.to_numeric(bdf["missing_correctness_flag"], errors="coerce").mean()),
                "abnormal_response_rate": float(pd.to_numeric(bdf["abnormal_response_flag"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["subj", "block_index_global"]).reset_index(drop=True)


def _subject_block_model_scores(df_subj: pd.DataFrame, subj: str) -> pd.DataFrame:
    seqs = _prepare_sequences(df_subj)
    if len(seqs) == 0:
        return pd.DataFrame()
    seed = 30_000 + int(subj[-3:])
    one = fit_model_with_restarts(seqs, "one_shot", seed=seed + 1)
    grad = fit_model_with_restarts(seqs, "gradual", seed=seed + 2)
    rows = []
    for seq in seqs:
        ll_one = score_loglik([seq], one)
        ll_grad = score_loglik([seq], grad)
        n = float(len(seq.y))
        rows.append(
            {
                "subj": subj,
                "block_id": seq.block_id,
                "n_trials": n,
                "ll_one_shot_block": ll_one,
                "ll_gradual_block": ll_grad,
                "ll_diff_block": ll_grad - ll_one,
                "ll_diff_per_trial_block": (ll_grad - ll_one) / n if n > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    trial_rows: List[pd.DataFrame] = []
    subject_rows: List[Dict[str, float]] = []
    fold_rows: List[Dict[str, float]] = []
    block_score_rows: List[pd.DataFrame] = []

    for i in range(1, N_SUBJECTS + 1):
        subj = f"reknow{i:03d}"
        log_path = os.path.join(LOGS_DIR, f"{subj}.log")
        if not os.path.exists(log_path):
            print(f"[WARN] missing log: {log_path}")
            continue

        base = build_trialwise_dataframe(log_path, subj, include_aux=True)
        dyn = _search_inference_dynamics(base, subj)
        merged = base.merge(dyn, on=["subj", "block_id", "trial_id", "phase", "block_position", "correctness"], how="left")
        merged = _add_trial_derived_cols(merged)
        merged = merged.rename(
            columns={
                "candidates_before": "candidates_before_tmp"
            }
        )
        # keep requested names
        merged["candidates_before"] = pd.to_numeric(merged["n_possible_rules_before"], errors="coerce")
        merged["candidates_after"] = pd.to_numeric(merged["n_possible_rules_after"], errors="coerce")
        merged["collapsed_flag"] = (merged["candidates_after"] == 1).astype(float)
        trial_rows.append(merged)

        subj_row, folds = run_subject_cv(base, subj)
        subject_rows.append(subj_row)
        fold_rows.extend(folds)
        block_score_rows.append(_subject_block_model_scores(base, subj))
        print(f"prepared long tables: {subj}")

    if not trial_rows:
        print("No data prepared.")
        return

    trial_long = pd.concat(trial_rows, ignore_index=True)
    keep_cols = [
        "subj",
        "block_id",
        "trial_id",
        "trial_in_block",
        "block_index_global",
        "correctness",
        "rt",
        "first_correct_flag",
        "post_first_correct_flag",
        "current_streak",
        "candidates_before",
        "candidates_after",
        "collapsed_flag",
        "stable_flag",
        "search_flag",
        "chosen_p_correct",
        "max_p_correct",
        "chosen_info_gain",
        "max_info_gain",
        "information_efficiency",
        "confirmatory_choice",
        "eligible_confirmatory_trial",
        "eligible_reduction_trial",
        "missing_correctness_flag",
        "abnormal_response_flag",
    ]
    trial_long = trial_long[keep_cols]
    ncols = trial_long.select_dtypes(include=[np.number]).columns
    trial_long[ncols] = trial_long[ncols].round(3)
    trial_path = os.path.join(OUT_DIR, "hmm_trial_long.csv")
    trial_long.to_csv(trial_path, index=False)

    block_long = _build_block_long(trial_long)
    ncols = block_long.select_dtypes(include=[np.number]).columns
    block_long[ncols] = block_long[ncols].round(3)
    block_path = os.path.join(OUT_DIR, "hmm_block_long.csv")
    block_long.to_csv(block_path, index=False)

    subj_df = pd.DataFrame(subject_rows)
    fold_df = pd.DataFrame(fold_rows)
    block_score_df = pd.concat(block_score_rows, ignore_index=True) if block_score_rows else pd.DataFrame()

    if not fold_df.empty:
        # add per-trial normalized diagnostics
        fold_df["ll_diff_per_trial"] = fold_df["ll_diff_gradual_minus_one_shot"] / fold_df["n_test_trials"]

    # Long-table-style model summaries
    ncols = subj_df.select_dtypes(include=[np.number]).columns
    subj_df[ncols] = subj_df[ncols].round(3)
    subj_path = os.path.join(OUT_DIR, "hmm_model_subject_summary_long.csv")
    subj_df.to_csv(subj_path, index=False)

    if not block_score_df.empty:
        ncols = block_score_df.select_dtypes(include=[np.number]).columns
        block_score_df[ncols] = block_score_df[ncols].round(3)
    block_score_path = os.path.join(OUT_DIR, "hmm_model_block_scores_long.csv")
    block_score_df.to_csv(block_score_path, index=False)

    fold_path = os.path.join(OUT_DIR, "hmm_model_fold_diagnostics_long.csv")
    ncols = fold_df.select_dtypes(include=[np.number]).columns
    fold_df[ncols] = fold_df[ncols].round(3)
    fold_df.to_csv(fold_path, index=False)

    print(f"saved: {trial_path}")
    print(f"saved: {block_path}")
    print(f"saved: {subj_path}")
    print(f"saved: {block_score_path}")
    print(f"saved: {fold_path}")


if __name__ == "__main__":
    main()
