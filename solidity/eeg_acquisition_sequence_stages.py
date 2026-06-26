import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr, ttest_rel

from transition_stages import stage_masks

try:
    import statsmodels.formula.api as smf
except Exception:
    smf = None


BASE = "/Users/lodysun/Desktop/Thesis"
EEG_TRIAL_CSV = os.path.join(BASE, "trials_trialwise", "1s_comp", "eeg_tables", "eeg_trial_long.csv")
HMM_BLOCK_CSV = os.path.join(BASE, "trials_trialwise", "1s_comp", "eeg_paper_results", "solidity", "hmm_unified_block_metrics.csv")
OUT_DIR = os.path.join(BASE, "trials_trialwise", "1s_comp", "eeg_paper_results", "solidity")

ERP_FEATURES = ["feedback_locked_FRN", "feedback_locked_P3a", "feedback_locked_P3b"]


def _paired_summary(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    d = a - b
    ok = np.isfinite(d)
    n = int(ok.sum())
    if n < 8:
        return {"n": n, "mean_diff": np.nan, "t": np.nan, "p": np.nan, "dz": np.nan}
    aa = a[ok]
    bb = b[ok]
    dd = aa - bb
    t, p = ttest_rel(aa, bb, nan_policy="omit")
    sd = float(np.nanstd(dd, ddof=1))
    dz = float(np.nanmean(dd) / sd) if sd > 0 else np.nan
    return {"n": n, "mean_diff": float(np.nanmean(dd)), "t": float(t), "p": float(p), "dz": dz}


def _build_stage_rows(eeg: pd.DataFrame, block: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    g = eeg.groupby(["subj", "block_id"], sort=False)
    for (subj, bid), bdf in g:
        hk = block[(block["subj"] == subj) & (block["block_id"] == bid)]
        if hk.empty:
            continue
        h = hk.iloc[0]
        fc = h["first_correct_trial"]
        core = h["acquisition_trial_core"]
        if pd.isna(fc) or pd.isna(core):
            continue
        fc = int(fc)
        core = int(core)
        if core < 1 or fc < 1:
            continue

        b = bdf.copy()
        b["correctness"] = pd.to_numeric(b.get("correctness"), errors="coerce")

        # Trial-stage classification: single source of truth (transition_stages).
        # This file's historical variant: index = trial_id, acquired '==' core,
        # NO correctness filter on pre_core/acquired, and OVERLAPPING views (a
        # fc==core trial belongs to both transition_first_correct and acquired_core).
        masks = stage_masks(
            b, trial_col="trial_id", fc=fc, core=core,
            acquired_boundary="==", pre_core_require_correct=False,
            acquired_require_correct=False,
        )
        cat_map = {
            "search_error": b[masks["search_error"]],
            "transition_first_correct": b[masks["transition_first_correct"]],
            "transition_pre_core": b[masks["transition_pre_core"]],
            "acquired_core": b[masks["acquired"]],
        }

        for feat in ERP_FEATURES:
            if feat not in b.columns:
                continue
            for cat, cdf in cat_map.items():
                vals = pd.to_numeric(cdf[feat], errors="coerce").to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                rows.append(
                    {
                        "subj": subj,
                        "block_id": bid,
                        "feature": feat,
                        "category": cat,
                        "value_mean": float(np.nanmean(vals)) if len(vals) else np.nan,
                        "n_trials_in_cat": int(len(vals)),
                        "first_correct_trial": float(fc),
                        "core_trial": float(core),
                        "lag_core": h["acquisition_lag_core"],
                        "width_20_80": h["transition_width_20_to_80"],
                        "set_index": h["set_index"],
                        "set_bin": h["set_bin"],
                        "rule_level": h["rule_level"],
                        "shift_type": h["shift_type"],
                        "is_early_set": float(h["set_index"] <= 4) if pd.notna(h["set_index"]) else np.nan,
                        "is_lag_pos": float(h["acquisition_lag_core"] > 0) if pd.notna(h["acquisition_lag_core"]) else np.nan,
                        "is_width_pos": float(h["transition_width_20_to_80"] > 0) if pd.notna(h["transition_width_20_to_80"]) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _paired_stage_tests(stage_df: pd.DataFrame) -> pd.DataFrame:
    subset_defs = {
        "all_blocks": np.ones(len(stage_df), dtype=bool),
        "early_sets": stage_df["is_early_set"] == 1,
        "lag_pos": stage_df["is_lag_pos"] == 1,
        "width_pos": stage_df["is_width_pos"] == 1,
        "early_and_lag_pos": (stage_df["is_early_set"] == 1) & (stage_df["is_lag_pos"] == 1),
        "early_and_width_pos": (stage_df["is_early_set"] == 1) & (stage_df["is_width_pos"] == 1),
    }

    pair_defs: List[Tuple[str, str]] = [
        ("transition_first_correct", "acquired_core"),
        ("transition_first_correct", "transition_pre_core"),
        ("transition_pre_core", "acquired_core"),
        ("search_error", "transition_first_correct"),
    ]

    rows: List[Dict[str, object]] = []
    for subset_name, mask in subset_defs.items():
        sd = stage_df[mask].copy()
        if sd.empty:
            continue
        for feat in ERP_FEATURES:
            sf = sd[sd["feature"] == feat]
            if sf.empty:
                continue
            for c1, c2 in pair_defs:
                a = sf[sf["category"] == c1][["subj", "block_id", "value_mean"]].rename(columns={"value_mean": "v1"})
                b = sf[sf["category"] == c2][["subj", "block_id", "value_mean"]].rename(columns={"value_mean": "v2"})
                m = a.merge(b, on=["subj", "block_id"], how="inner").dropna()
                if m.empty:
                    continue
                st = _paired_summary(m["v1"].to_numpy(dtype=float), m["v2"].to_numpy(dtype=float))
                rows.append(
                    {
                        "subset": subset_name,
                        "feature": feat,
                        "comparison": f"{c1}_minus_{c2}",
                        "n_blocks": int(len(m)),
                        "mean_c1": float(np.nanmean(m["v1"])),
                        "mean_c2": float(np.nanmean(m["v2"])),
                        "mean_diff": st["mean_diff"],
                        "t": st["t"],
                        "p": st["p"],
                        "dz": st["dz"],
                    }
                )
    return pd.DataFrame(rows)


def _predictive_tables(stage_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Predictor: first-correct trial ERP amplitude within the transition phase (per block)
    fctr = (
        stage_df[stage_df["category"] == "transition_first_correct"]
        .pivot_table(
            index=["subj", "block_id", "set_index", "set_bin", "rule_level", "shift_type", "lag_core", "width_20_80", "is_early_set", "is_lag_pos", "is_width_pos"],
            columns="feature",
            values="value_mean",
            aggfunc="first",
        )
        .reset_index()
    )
    fctr.columns.name = None

    corr_rows: List[Dict[str, object]] = []
    subset_defs = {
        "all_blocks": np.ones(len(fctr), dtype=bool),
        "early_sets": fctr["is_early_set"] == 1,
        "lag_pos": fctr["is_lag_pos"] == 1,
        "width_pos": fctr["is_width_pos"] == 1,
        "early_and_lag_pos": (fctr["is_early_set"] == 1) & (fctr["is_lag_pos"] == 1),
        "early_and_width_pos": (fctr["is_early_set"] == 1) & (fctr["is_width_pos"] == 1),
    }
    for sub, mask in subset_defs.items():
        d = fctr[mask].copy()
        if d.empty:
            continue
        for feat in ERP_FEATURES:
            if feat not in d.columns:
                continue
            for y in ["lag_core", "width_20_80"]:
                x = pd.to_numeric(d[feat], errors="coerce").to_numpy(dtype=float)
                yy = pd.to_numeric(d[y], errors="coerce").to_numpy(dtype=float)
                ok = np.isfinite(x) & np.isfinite(yy)
                if ok.sum() < 12:
                    continue
                rho, p = spearmanr(x[ok], yy[ok], nan_policy="omit")
                n_ok = int(ok.sum())
                if n_ok > 3 and np.isfinite(rho) and abs(float(rho)) < 1:
                    z = np.arctanh(float(rho))
                    se_z = 1.0 / np.sqrt(n_ok - 3)
                    ci_low = float(np.tanh(z - 1.96 * se_z))
                    ci_high = float(np.tanh(z + 1.96 * se_z))
                else:
                    ci_low = np.nan
                    ci_high = np.nan
                corr_rows.append(
                    {
                        "subset": sub,
                        "x_feature": feat,
                        "y_metric": y,
                        "n": n_ok,
                        "spearman_rho": float(rho),
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "p": float(p),
                    }
                )

    corr_df = pd.DataFrame(corr_rows)

    # Optional mixedlm slope estimates
    reg_rows: List[Dict[str, object]] = []
    if smf is not None and not fctr.empty:
        dd = fctr.copy()
        dd["set_index"] = pd.to_numeric(dd["set_index"], errors="coerce")
        dd = dd.dropna(subset=["set_index", "subj"])
        dd = dd[dd["shift_type"] != "first_block"].copy()
        for y in ["lag_core", "width_20_80"]:
            for x in ERP_FEATURES:
                if x not in dd.columns:
                    continue
                d = dd[[y, x, "set_index", "rule_level", "shift_type", "subj"]].dropna().copy()
                if len(d) < 80:
                    continue
                try:
                    m = smf.mixedlm(f"{y} ~ {x} + set_index + C(shift_type) + C(rule_level)", d, groups=d["subj"], re_formula="1")
                    r = m.fit(method="lbfgs", reml=False, maxiter=200, disp=False)
                    ci = r.conf_int()
                    beta = float(r.params.get(x, np.nan))
                    se = float(r.bse.get(x, np.nan))
                    z_value = float(beta / se) if np.isfinite(beta) and np.isfinite(se) and se > 0 else np.nan
                    reg_rows.append(
                        {
                            "y_metric": y,
                            "x_feature": x,
                            "n": int(len(d)),
                            "coef_x": beta,
                            "se_x": se,
                            "ci95_low": float(ci.loc[x, 0]) if x in ci.index else np.nan,
                            "ci95_high": float(ci.loc[x, 1]) if x in ci.index else np.nan,
                            "z_x": z_value,
                            "p_x": float(r.pvalues.get(x, np.nan)),
                            "converged": bool(getattr(r, "converged", False)),
                        }
                    )
                except Exception:
                    continue
    reg_df = pd.DataFrame(reg_rows)
    return corr_df, reg_df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    eeg = pd.read_csv(EEG_TRIAL_CSV)
    block = pd.read_csv(HMM_BLOCK_CSV)

    eeg["block_id"] = eeg["block_id"].astype(str)
    block["block_id"] = block["block_id"].astype(str)
    for c in [
        "first_correct_trial",
        "acquisition_trial_core",
        "acquisition_lag_core",
        "transition_width_20_to_80",
        "set_index",
    ]:
        block[c] = pd.to_numeric(block[c], errors="coerce")

    stage_df = _build_stage_rows(eeg, block)
    stage_path = os.path.join(OUT_DIR, "eeg_acq_sequence_stage_means_long.csv")
    stage_df.to_csv(stage_path, index=False)

    pair_df = _paired_stage_tests(stage_df)
    pair_path = os.path.join(OUT_DIR, "eeg_acq_sequence_stage_pair_tests.csv")
    pair_df.to_csv(pair_path, index=False)

    corr_df, reg_df = _predictive_tables(stage_df)
    corr_path = os.path.join(OUT_DIR, "eeg_transition_firstcorrect_predicts_hmm_metrics_spearman.csv")
    reg_path = os.path.join(OUT_DIR, "eeg_transition_firstcorrect_predicts_hmm_metrics_mixedlm.csv")
    corr_df.to_csv(corr_path, index=False)
    reg_df.to_csv(reg_path, index=False)

    print(f"saved: {stage_path}")
    print(f"saved: {pair_path}")
    print(f"saved: {corr_path}")
    print(f"saved: {reg_path}")


if __name__ == "__main__":
    main()
