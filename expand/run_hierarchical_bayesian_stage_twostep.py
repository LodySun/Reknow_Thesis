import argparse
import os
import sys

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from run_bayesian_stage_and_twostep_sensitivity import (  # noqa: E402
    EXP,
    FEATURES,
    P300_STAGE_CONTRASTS,
    SET_BINS,
    TWOSTEP_CONTRASTS,
    _build_p300_stage_cells,
    _build_twostep_stage_cells,
    _harmonic_weight,
)


OUT_PAIRS = os.path.join(EXP, "hierarchical_bayesian_stage_twostep_block_pairs.csv")
OUT_SUMMARY = os.path.join(EXP, "hierarchical_bayesian_stage_twostep_summary.csv")
OUT_DRAWS = os.path.join(EXP, "hierarchical_bayesian_stage_twostep_draws.csv")
OUT_IDATA = os.path.join(EXP, "hierarchical_bayesian_stage_twostep_idata.nc")

SEED = 20260504


def _block_pair_contrasts(
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
    ][["subj", "block_id", "set_index", "value_mean", "n_trials"]].rename(
        columns={"value_mean": "value_a", "n_trials": "n_a"}
    )
    db = cells[
        (cells["feature"] == feature)
        & (cells["category"] == stage_b)
        & (cells["set_index"].isin(SET_BINS[set_name]))
    ][["subj", "block_id", "set_index", "value_mean", "n_trials"]].rename(
        columns={"value_mean": "value_b", "n_trials": "n_b"}
    )
    m = da.merge(db, on=["subj", "block_id", "set_index"], how="inner").dropna()
    if m.empty:
        return pd.DataFrame()

    m["diff"] = m["value_a"] - m["value_b"]
    m["pair_weight"] = _harmonic_weight(m["n_a"].to_numpy(dtype=float), m["n_b"].to_numpy(dtype=float))
    m = m[np.isfinite(m["diff"]) & np.isfinite(m["pair_weight"]) & (m["pair_weight"] > 0)].copy()
    if m.empty:
        return pd.DataFrame()

    m.insert(0, "analysis_family", analysis_family)
    m.insert(1, "set_bin", set_name)
    m.insert(2, "feature", feature)
    m.insert(3, "contrast", contrast_name)
    m.insert(4, "stage_a", stage_a)
    m.insert(5, "stage_b", stage_b)
    return m


def build_block_pair_table() -> pd.DataFrame:
    p300_cells = _build_p300_stage_cells()
    twostep_cells = _build_twostep_stage_cells()

    tables = []
    for set_name in SET_BINS:
        for feature in FEATURES:
            for contrast_name, stage_a, stage_b in P300_STAGE_CONTRASTS:
                tables.append(
                    _block_pair_contrasts(
                        p300_cells,
                        "p300_frn_stage_changes",
                        set_name,
                        feature,
                        contrast_name,
                        stage_a,
                        stage_b,
                    )
                )
            for contrast_name, stage_a, stage_b in TWOSTEP_CONTRASTS:
                tables.append(
                    _block_pair_contrasts(
                        twostep_cells,
                        "two_step_transition_confirmation",
                        set_name,
                        feature,
                        contrast_name,
                        stage_a,
                        stage_b,
                    )
                )

    d = pd.concat([x for x in tables if x is not None and not x.empty], ignore_index=True)
    group_cols = ["analysis_family", "set_bin", "feature", "contrast", "stage_a", "stage_b"]
    d["group_label"] = d[group_cols].astype(str).agg(" | ".join, axis=1)
    return d


def fit_model(
    pairs: pd.DataFrame,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    target_accept: float,
) -> az.InferenceData:
    pairs = pairs.copy()
    group_labels = sorted(pairs["group_label"].unique())
    subject_labels = sorted(pairs["subj"].unique())
    group_index = {name: i for i, name in enumerate(group_labels)}
    subject_index = {name: i for i, name in enumerate(subject_labels)}

    pairs["group_idx"] = pairs["group_label"].map(group_index).astype(int)
    pairs["subject_idx"] = pairs["subj"].map(subject_index).astype(int)
    pairs["weight_rel"] = np.nan
    for _, g in pairs.groupby("group_idx"):
        med_weight = float(g["pair_weight"].median())
        pairs.loc[g.index, "weight_rel"] = g["pair_weight"] / med_weight
    pairs["weight_rel"] = pairs["weight_rel"].clip(lower=0.05, upper=20.0)

    y = pairs["diff"].to_numpy(dtype=float)
    group_idx = pairs["group_idx"].to_numpy(dtype=int)
    subject_idx = pairs["subject_idx"].to_numpy(dtype=int)
    weight_rel = pairs["weight_rel"].to_numpy(dtype=float)

    coords = {
        "group": group_labels,
        "subject": subject_labels,
        "obs": np.arange(len(pairs)),
    }

    with pm.Model(coords=coords) as model:
        mu = pm.Normal("mu", mu=0.0, sigma=5.0, dims="group")
        sigma_subject = pm.HalfNormal("sigma_subject", sigma=5.0, dims="group")
        sigma_obs = pm.HalfNormal("sigma_obs_median_weight", sigma=5.0, dims="group")
        z_subject = pm.Normal("z_subject", mu=0.0, sigma=1.0, dims=("group", "subject"))

        eta = mu[group_idx] + z_subject[group_idx, subject_idx] * sigma_subject[group_idx]
        obs_sigma = sigma_obs[group_idx] / np.sqrt(weight_rel)
        pm.StudentT("diff_obs", nu=4.0, mu=eta, sigma=obs_sigma, observed=y, dims="obs")

        pm.Deterministic(
            "d_marginal_median_weight",
            mu / pm.math.sqrt(sigma_subject**2 + sigma_obs**2),
            dims="group",
        )
        pm.Deterministic(
            "d_subject_heterogeneity",
            mu / sigma_subject,
            dims="group",
        )

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=SEED,
            progressbar=True,
        )

    return idata


def _stack_group_draws(idata: az.InferenceData, var_name: str) -> np.ndarray:
    return (
        idata.posterior[var_name]
        .stack(sample=("chain", "draw"))
        .transpose("group", "sample")
        .values
    )


def summarize(pairs: pd.DataFrame, idata: az.InferenceData) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_labels = list(idata.posterior.coords["group"].values)
    group_meta = (
        pairs.groupby("group_label", as_index=False)
        .agg(
            analysis_family=("analysis_family", "first"),
            set_bin=("set_bin", "first"),
            feature=("feature", "first"),
            contrast=("contrast", "first"),
            stage_a=("stage_a", "first"),
            stage_b=("stage_b", "first"),
            n_observations=("diff", "size"),
            n_subjects=("subj", "nunique"),
            median_n_a=("n_a", "median"),
            median_n_b=("n_b", "median"),
            median_pair_weight=("pair_weight", "median"),
            min_pair_weight=("pair_weight", "min"),
            max_pair_weight=("pair_weight", "max"),
            raw_block_mean_diff=("diff", "mean"),
        )
        .set_index("group_label")
    )

    mu = _stack_group_draws(idata, "mu")
    d_marginal = _stack_group_draws(idata, "d_marginal_median_weight")
    d_subject = _stack_group_draws(idata, "d_subject_heterogeneity")
    sigma_subject = _stack_group_draws(idata, "sigma_subject")
    sigma_obs = _stack_group_draws(idata, "sigma_obs_median_weight")

    rhat_mu = az.rhat(idata, var_names=["mu"])["mu"].values
    ess_mu = az.ess(idata, var_names=["mu"], method="bulk")["mu"].values
    divergences = int(idata.sample_stats["diverging"].sum().values)

    rows = []
    draw_rows = []
    for i, group_label in enumerate(group_labels):
        meta = group_meta.loc[group_label].to_dict()
        mu_draw = mu[i]
        dm_draw = d_marginal[i]
        ds_draw = d_subject[i]
        direction_prob = float(np.mean(mu_draw > 0)) if np.mean(mu_draw) >= 0 else float(np.mean(mu_draw < 0))
        mu_low, mu_high = np.quantile(mu_draw, [0.025, 0.975])
        dm_low, dm_high = np.quantile(dm_draw, [0.025, 0.975])
        ds_low, ds_high = np.quantile(ds_draw, [0.025, 0.975])
        supported_95 = bool((mu_low > 0 or mu_high < 0) and direction_prob >= 0.975)
        rows.append(
            {
                "group_label": group_label,
                **meta,
                "posterior_mean": float(np.mean(mu_draw)),
                "posterior_median": float(np.median(mu_draw)),
                "ci95_low": float(mu_low),
                "ci95_high": float(mu_high),
                "pr_gt_0": float(np.mean(mu_draw > 0)),
                "pr_lt_0": float(np.mean(mu_draw < 0)),
                "pr_direction": direction_prob,
                "bayes_supported_95": supported_95,
                "d_marginal_mean": float(np.mean(dm_draw)),
                "d_marginal_ci95_low": float(dm_low),
                "d_marginal_ci95_high": float(dm_high),
                "d_subject_mean": float(np.mean(ds_draw)),
                "d_subject_ci95_low": float(ds_low),
                "d_subject_ci95_high": float(ds_high),
                "sigma_subject_mean": float(np.mean(sigma_subject[i])),
                "sigma_obs_median_weight_mean": float(np.mean(sigma_obs[i])),
                "rhat_mu": float(rhat_mu[i]),
                "ess_bulk_mu": float(ess_mu[i]),
                "total_divergences": divergences,
            }
        )
        for j, val in enumerate(mu_draw):
            draw_rows.append(
                {
                    "group_label": group_label,
                    "draw": j,
                    "mu": float(val),
                    "d_marginal_median_weight": float(dm_draw[j]),
                    "d_subject_heterogeneity": float(ds_draw[j]),
                }
            )

    summary = pd.DataFrame(rows)
    draws = pd.DataFrame(draw_rows)
    return summary, draws


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=1200)
    parser.add_argument("--tune", type=int, default=1200)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--min-subjects", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.95)
    args = parser.parse_args()

    os.makedirs(EXP, exist_ok=True)
    pairs = build_block_pair_table()
    counts = pairs.groupby("group_label")["subj"].nunique()
    keep = counts[counts >= args.min_subjects].index
    pairs = pairs[pairs["group_label"].isin(keep)].copy()
    pairs.to_csv(OUT_PAIRS, index=False)

    idata = fit_model(pairs, args.draws, args.tune, args.chains, args.cores, args.target_accept)
    idata.to_netcdf(OUT_IDATA)
    summary, draws = summarize(pairs, idata)
    summary.to_csv(OUT_SUMMARY, index=False)
    draws.to_csv(OUT_DRAWS, index=False)

    print(f"saved: {OUT_PAIRS}")
    print(f"saved: {OUT_SUMMARY}")
    print(f"saved: {OUT_DRAWS}")
    print(f"saved: {OUT_IDATA}")
    show_cols = [
        "analysis_family",
        "set_bin",
        "feature",
        "contrast",
        "n_subjects",
        "n_observations",
        "posterior_mean",
        "ci95_low",
        "ci95_high",
        "pr_direction",
        "d_marginal_mean",
        "bayes_supported_95",
        "rhat_mu",
        "ess_bulk_mu",
        "total_divergences",
    ]
    print(summary[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
