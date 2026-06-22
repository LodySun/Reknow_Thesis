import os

import numpy as np
import pandas as pd


BASE = "/Users/lodysun/Desktop/Thesis"
RESULT_BASE = os.path.join(BASE, "trials_trialwise", "1s_comp", "eeg_paper_results")
SOL = os.path.join(RESULT_BASE, "solidity")
EXP = os.path.join(RESULT_BASE, "expand")

EEG_TRIAL = os.path.join(BASE, "trials_trialwise", "1s_comp", "eeg_tables", "eeg_trial_long.csv")
TRIAL_POST = os.path.join(SOL, "hmm_unified_trial_posteriors.csv")
ACQ_STAGE_CSV = os.path.join(SOL, "eeg_acq_sequence_stage_means_long.csv")
SEARCH_STAGE_CSV = os.path.join(SOL, "search_transition_erp_stage_means_long.csv")

OUT_SUBJECTS = os.path.join(EXP, "bayesian_stage_twostep_trialcount_subject_contrasts.csv")
OUT_DRAWS = os.path.join(EXP, "bayesian_stage_twostep_trialcount_posterior_draws.csv")
OUT_SUMMARY = os.path.join(EXP, "bayesian_stage_twostep_trialcount_summary.csv")

FEATURES = ["feedback_locked_FRN", "feedback_locked_P3b"]
P300_STAGE_CONTRASTS = [
    ("transition_first_correct_minus_search_error", "transition_first_correct", "search_error"),
    ("transition_first_correct_minus_transition_pre_core_correct", "transition_first_correct", "transition_pre_core_correct"),
    ("transition_first_correct_minus_acquired_correct", "transition_first_correct", "acquired_correct"),
]
TWOSTEP_CONTRASTS = [
    ("step1_transition_fc_minus_search", "transition_first_correct", "search_error"),
    ("step2a_transition_precore_minus_fc", "transition_pre_core", "transition_first_correct"),
    ("step2b_acquired_core_minus_transition_precore", "acquired_core", "transition_pre_core"),
]
SET_BINS = {
    "early_1_2": [1, 2],
    "late_7_8": [7, 8],
    "all_sets": list(range(1, 9)),
}

N_DRAWS = 20000
SEED = 20260504


def _safe_num(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _posterior_mean_normal(vals: np.ndarray, rng: np.random.Generator, n_draws: int = N_DRAWS) -> tuple[np.ndarray, dict]:
    y = np.asarray(vals, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 3:
        draws = np.full(n_draws, np.nan)
        return draws, {
            "n_subjects": int(n),
            "posterior_mean": np.nan,
            "posterior_median": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "pr_gt_0": np.nan,
            "pr_lt_0": np.nan,
            "posterior_sd": np.nan,
        }

    # Weak Normal-Inverse-Gamma prior for subject-level contrast distribution.
    mu0 = 0.0
    k0 = 1e-6
    a0 = 1e-3
    b0 = 1e-3

    ybar = float(np.mean(y))
    ss = float(np.sum((y - ybar) ** 2))
    kn = k0 + n
    mun = (k0 * mu0 + n * ybar) / kn
    an = a0 + n / 2.0
    bn = b0 + 0.5 * ss + (k0 * n * (ybar - mu0) ** 2) / (2.0 * kn)

    sigma2 = 1.0 / rng.gamma(shape=an, scale=1.0 / bn, size=n_draws)
    mu = rng.normal(loc=mun, scale=np.sqrt(sigma2 / kn), size=n_draws)
    summary = {
        "n_subjects": int(n),
        "posterior_mean": float(np.mean(mu)),
        "posterior_median": float(np.median(mu)),
        "ci95_low": float(np.quantile(mu, 0.025)),
        "ci95_high": float(np.quantile(mu, 0.975)),
        "pr_gt_0": float(np.mean(mu > 0)),
        "pr_lt_0": float(np.mean(mu < 0)),
        "posterior_sd": float(np.std(mu, ddof=1)),
    }
    return mu, summary


def _harmonic_weight(na: np.ndarray, nb: np.ndarray) -> np.ndarray:
    na = np.asarray(na, dtype=float)
    nb = np.asarray(nb, dtype=float)
    out = np.full(len(na), np.nan, dtype=float)
    ok = np.isfinite(na) & np.isfinite(nb) & (na > 0) & (nb > 0)
    out[ok] = 1.0 / (1.0 / na[ok] + 1.0 / nb[ok])
    return out


def _build_p300_stage_cells() -> pd.DataFrame:
    eeg = pd.read_csv(EEG_TRIAL)
    post = pd.read_csv(TRIAL_POST)
    eeg["subj"] = eeg["subj"].astype(str)
    eeg["block_id"] = eeg["block_id"].astype(str)
    post["subj"] = post["subj"].astype(str)
    post["block_id"] = post["block_id"].astype(str)
    _safe_num(eeg, ["trial_id"] + FEATURES)
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
    m["category"] = ""
    is_fc = m["trial_index_1based"] == m["first_correct_trial"]
    is_transition_precore_corr = (
        (m["trial_index_1based"] > m["first_correct_trial"])
        & (m["trial_index_1based"] < m["acquisition_trial_core"])
        & (m["correctness"] == 1)
    )
    is_search_error = (m["trial_index_1based"] < m["first_correct_trial"]) & (m["correctness"] == 0)
    is_acquired_corr = (m["trial_index_1based"] >= m["acquisition_trial_core"]) & (m["correctness"] == 1)
    m.loc[is_search_error, "category"] = "search_error"
    m.loc[is_fc, "category"] = "transition_first_correct"
    m.loc[is_transition_precore_corr, "category"] = "transition_pre_core_correct"
    m.loc[is_acquired_corr, "category"] = "acquired_correct"
    m = m[m["category"] != ""].copy()

    rows = []
    for feat in FEATURES:
        g = (
            m.dropna(subset=[feat, "set_index"])
            .groupby(["subj", "block_id", "set_index", "category"], as_index=False)
            .agg(value_mean=(feat, "mean"), n_trials=(feat, "count"))
        )
        g["feature"] = feat
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def _build_twostep_stage_cells() -> pd.DataFrame:
    d1 = pd.read_csv(ACQ_STAGE_CSV)
    d1["subj"] = d1["subj"].astype(str)
    d1["block_id"] = d1["block_id"].astype(str)
    _safe_num(d1, ["set_index", "value_mean", "n_trials_in_cat"])
    d1 = d1[
        d1["feature"].isin(FEATURES)
        & d1["category"].isin(["transition_first_correct", "transition_pre_core", "acquired_core"])
    ].copy()
    d1 = d1.rename(columns={"n_trials_in_cat": "n_trials"})
    d1 = d1[["subj", "block_id", "set_index", "feature", "category", "value_mean", "n_trials"]]

    d2 = pd.read_csv(SEARCH_STAGE_CSV)
    d2["subj"] = d2["subj"].astype(str)
    d2["block_id"] = d2["block_id"].astype(str)
    _safe_num(d2, ["set_index", "value_mean", "n_trials"])
    d2 = d2[d2["feature"].isin(FEATURES) & (d2["category"] == "search_error")].copy()
    d2 = d2[["subj", "block_id", "set_index", "feature", "category", "value_mean", "n_trials"]]

    d = pd.concat([d1, d2], ignore_index=True)
    d = d.dropna(subset=["set_index", "value_mean", "n_trials"])
    d = d[d["n_trials"] > 0].copy()
    return d


def _subject_weighted_contrasts(
    cells: pd.DataFrame,
    analysis_family: str,
    set_name: str,
    feature: str,
    contrast_name: str,
    stage_a: str,
    stage_b: str,
) -> pd.DataFrame:
    da = cells[
        (cells["feature"] == feature)
        & (cells["category"] == stage_a)
        & (cells["set_index"].isin(SET_BINS[set_name]))
    ][["subj", "block_id", "value_mean", "n_trials"]].rename(columns={"value_mean": "va", "n_trials": "na"})
    db = cells[
        (cells["feature"] == feature)
        & (cells["category"] == stage_b)
        & (cells["set_index"].isin(SET_BINS[set_name]))
    ][["subj", "block_id", "value_mean", "n_trials"]].rename(columns={"value_mean": "vb", "n_trials": "nb"})
    m = da.merge(db, on=["subj", "block_id"], how="inner").dropna()
    if m.empty:
        return pd.DataFrame()
    m["diff"] = m["va"] - m["vb"]
    m["pair_weight"] = _harmonic_weight(m["na"].to_numpy(dtype=float), m["nb"].to_numpy(dtype=float))
    m = m[np.isfinite(m["diff"]) & np.isfinite(m["pair_weight"]) & (m["pair_weight"] > 0)].copy()
    if m.empty:
        return pd.DataFrame()
    rows = []
    for subj, g in m.groupby("subj", sort=True):
        w = g["pair_weight"].to_numpy(dtype=float)
        d = g["diff"].to_numpy(dtype=float)
        rows.append(
            {
                "analysis_family": analysis_family,
                "set_bin": set_name,
                "feature": feature,
                "contrast": contrast_name,
                "stage_a": stage_a,
                "stage_b": stage_b,
                "subj": subj,
                "weighted_mean_diff": float(np.average(d, weights=w)),
                "unweighted_mean_diff": float(np.mean(d)),
                "n_paired_blocks": int(len(g)),
                "sum_pair_weight": float(np.sum(w)),
                "median_n_stage_a": float(g["na"].median()),
                "median_n_stage_b": float(g["nb"].median()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(EXP, exist_ok=True)
    rng = np.random.default_rng(SEED)

    p300_cells = _build_p300_stage_cells()
    twostep_cells = _build_twostep_stage_cells()

    subject_tables = []
    for set_name in SET_BINS:
        for feat in FEATURES:
            for cname, a, b in P300_STAGE_CONTRASTS:
                subject_tables.append(
                    _subject_weighted_contrasts(p300_cells, "p300_frn_stage_changes", set_name, feat, cname, a, b)
                )
            for cname, a, b in TWOSTEP_CONTRASTS:
                subject_tables.append(
                    _subject_weighted_contrasts(twostep_cells, "two_step_transition_confirmation", set_name, feat, cname, a, b)
                )
    subj = pd.concat([x for x in subject_tables if x is not None and not x.empty], ignore_index=True)
    subj.to_csv(OUT_SUBJECTS, index=False)

    draw_rows = []
    summary_rows = []
    for key, g in subj.groupby(["analysis_family", "set_bin", "feature", "contrast", "stage_a", "stage_b"], sort=False):
        fam, set_name, feat, cname, a, b = key
        vals = g["weighted_mean_diff"].to_numpy(dtype=float)
        draws, summ = _posterior_mean_normal(vals, rng)
        subj_sd = float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else np.nan
        dz = float(np.nanmean(vals) / subj_sd) if np.isfinite(subj_sd) and subj_sd > 0 else np.nan
        for i, val in enumerate(draws):
            draw_rows.append(
                {
                    "analysis_family": fam,
                    "set_bin": set_name,
                    "feature": feat,
                    "contrast": cname,
                    "draw": i,
                    "posterior_mean_diff": val,
                }
            )
        direction_prob = summ["pr_gt_0"] if np.nanmean(vals) >= 0 else summ["pr_lt_0"]
        summary_rows.append(
            {
                "analysis_family": fam,
                "set_bin": set_name,
                "feature": feat,
                "contrast": cname,
                "stage_a": a,
                "stage_b": b,
                "n_subjects": summ["n_subjects"],
                "mean_subject_weighted_diff": float(np.nanmean(vals)),
                "subject_sd_weighted_diff": subj_sd,
                "dz_subject_weighted": dz,
                "posterior_mean": summ["posterior_mean"],
                "posterior_median": summ["posterior_median"],
                "ci95_low": summ["ci95_low"],
                "ci95_high": summ["ci95_high"],
                "pr_gt_0": summ["pr_gt_0"],
                "pr_lt_0": summ["pr_lt_0"],
                "pr_direction": direction_prob,
                "posterior_sd": summ["posterior_sd"],
                "median_paired_blocks_per_subject": float(g["n_paired_blocks"].median()),
                "min_paired_blocks_per_subject": int(g["n_paired_blocks"].min()),
                "median_n_stage_a": float(g["median_n_stage_a"].median()),
                "median_n_stage_b": float(g["median_n_stage_b"].median()),
            }
        )

    draws_df = pd.DataFrame(draw_rows)
    summary = pd.DataFrame(summary_rows)
    draws_df.to_csv(OUT_DRAWS, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    print(f"saved: {OUT_SUBJECTS}")
    print(f"saved: {OUT_DRAWS}")
    print(f"saved: {OUT_SUMMARY}")
    show = summary[
        [
            "analysis_family",
            "set_bin",
            "feature",
            "contrast",
            "n_subjects",
            "posterior_mean",
            "ci95_low",
            "ci95_high",
            "pr_gt_0",
            "pr_lt_0",
            "median_paired_blocks_per_subject",
            "median_n_stage_a",
            "median_n_stage_b",
        ]
    ]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
