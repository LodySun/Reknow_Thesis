import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


BASE = "base_dir"
POST_CSV = os.path.join(
    BASE,
    "trials_trialwise",
    "1s_comp",
    "eeg_paper_results",
    "solidity",
    "hmm_unified_trial_posteriors.csv",
)
HMM_LONG_CSV = os.path.join(BASE, "trials_trialwise", "hmm_long_tables", "hmm_trial_long.csv")
OUT_DIR = os.path.join(
    BASE,
    "trials_trialwise",
    "1s_comp",
    "eeg_paper_results",
    "expand",
    "marker_validity_collapse",
)

OUT_BLOCK = os.path.join(OUT_DIR, "hmm_core_sensitivity_block_level.csv")
OUT_SUM = os.path.join(OUT_DIR, "hmm_core_sensitivity_summary.csv")
OUT_CHECKS = os.path.join(OUT_DIR, "hmm_core_sensitivity_key_checks.csv")


def _safe_num(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _first_run_start(flag: np.ndarray, run_len: int) -> float:
    if len(flag) == 0:
        return np.nan
    streak = 0
    for i, v in enumerate(flag):
        if bool(v):
            streak += 1
            if streak >= run_len:
                return float(i - run_len + 2)  # 1-based trial index
        else:
            streak = 0
    return np.nan


def _first_crossing(x: np.ndarray, threshold: float) -> float:
    idx = np.where(x >= threshold)[0]
    if len(idx) == 0:
        return np.nan
    return float(idx[0] + 1)  # 1-based


def _build_collapse_after(hlong: pd.DataFrame) -> pd.DataFrame:
    h = hlong[["subj", "block_id", "trial_in_block", "candidates_after"]].dropna().copy()
    h["collapsed_flag_after"] = (pd.to_numeric(h["candidates_after"], errors="coerce") <= 1).astype(int)
    out = (
        h[h["collapsed_flag_after"] == 1]
        .groupby(["subj", "block_id"], as_index=False)["trial_in_block"]
        .min()
        .rename(columns={"trial_in_block": "collapse_after_trial"})
    )
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    post = pd.read_csv(POST_CSV)
    hlong = pd.read_csv(HMM_LONG_CSV)
    for df in [post, hlong]:
        df["subj"] = df["subj"].astype(str)
        df["block_id"] = df["block_id"].astype(str)
    _safe_num(
        post,
        [
            "trial_index_1based",
            "p_acq_filtered",
            "first_correct_trial",
            "acquisition_trial_core",
            "set_index",
        ],
    )
    _safe_num(hlong, ["trial_in_block", "candidates_after"])

    collapse = _build_collapse_after(hlong)

    # Reference (current default in manuscript pipeline)
    ref = (
        post.groupby(["subj", "block_id"], as_index=False)
        .agg(
            first_correct_trial=("first_correct_trial", "first"),
            acquisition_trial_core_ref=("acquisition_trial_core", "first"),
            set_index=("set_index", "first"),
        )
        .dropna(subset=["first_correct_trial"])
    )

    grids = [(0.7, 1), (0.7, 2), (0.7, 3), (0.8, 1), (0.8, 2), (0.8, 3), (0.9, 1), (0.9, 2), (0.9, 3)]

    rows = []
    for tau, run_len in grids:
        for (subj, bid), g in post.groupby(["subj", "block_id"], sort=False):
            g = g.sort_values("trial_index_1based")
            p = pd.to_numeric(g["p_acq_filtered"], errors="coerce").to_numpy(dtype=float)
            if len(p) == 0:
                continue
            core = _first_run_start(p >= tau, run_len=run_len)
            t20 = _first_crossing(p, 0.2)
            t80 = _first_crossing(p, 0.8)
            width_20_80 = (t80 - t20) if (np.isfinite(t20) and np.isfinite(t80)) else np.nan

            rows.append(
                {
                    "tau": float(tau),
                    "run_len": int(run_len),
                    "subj": subj,
                    "block_id": bid,
                    "core_trial_grid": core,
                    "t20": t20,
                    "t80": t80,
                    "width_20_80": width_20_80,
                }
            )

    block = pd.DataFrame(rows)
    block = (
        block.merge(ref, on=["subj", "block_id"], how="left")
        .merge(collapse, on=["subj", "block_id"], how="left")
        .copy()
    )
    block["lag_fc_to_core"] = block["core_trial_grid"] - block["first_correct_trial"]
    block["lag_collapse_to_core"] = block["core_trial_grid"] - block["collapse_after_trial"]
    block["delta_core_vs_ref"] = block["core_trial_grid"] - block["acquisition_trial_core_ref"]
    block.to_csv(OUT_BLOCK, index=False)

    sum_rows = []
    for (tau, run_len), d in block.groupby(["tau", "run_len"], sort=True):
        x = pd.to_numeric(d["core_trial_grid"], errors="coerce")
        fc = pd.to_numeric(d["first_correct_trial"], errors="coerce")
        co = pd.to_numeric(d["collapse_after_trial"], errors="coerce")
        ref_core = pd.to_numeric(d["acquisition_trial_core_ref"], errors="coerce")
        w = pd.to_numeric(d["width_20_80"], errors="coerce")

        ok = np.isfinite(x) & np.isfinite(fc) & np.isfinite(ref_core)
        if ok.sum() < 10:
            continue

        lag_fc = x[ok] - fc[ok]
        delta_ref = x[ok] - ref_core[ok]
        rho, p = spearmanr(x[ok], ref_core[ok], nan_policy="omit")

        okc = ok & np.isfinite(co)
        lag_col = (x[okc] - co[okc]) if okc.sum() else np.array([])

        sum_rows.append(
            {
                "tau": float(tau),
                "run_len": int(run_len),
                "n_blocks": int(ok.sum()),
                "core_mean_trial": float(np.nanmean(x[ok])),
                "lag_fc_to_core_mean": float(np.nanmean(lag_fc)),
                "prop_fc_before_or_eq_core": float(np.mean(fc[ok] <= x[ok])),
                "lag_collapse_to_core_mean": float(np.nanmean(lag_col)) if len(lag_col) else np.nan,
                "prop_collapse_before_or_eq_core": float(np.mean(co[okc] <= x[okc])) if okc.sum() else np.nan,
                "width_20_80_mean": float(np.nanmean(w)),
                "delta_core_vs_ref_mean": float(np.nanmean(delta_ref)),
                "rho_core_vs_ref": float(rho) if np.isfinite(rho) else np.nan,
                "p_core_vs_ref": float(p) if np.isfinite(p) else np.nan,
            }
        )

    summary = pd.DataFrame(sum_rows).sort_values(["tau", "run_len"]).reset_index(drop=True)
    summary.to_csv(OUT_SUM, index=False)

    # Key checks for manuscript: whether qualitative direction flips
    checks = []
    if not summary.empty:
        for _, r in summary.iterrows():
            checks.append(
                {
                    "tau": r["tau"],
                    "run_len": int(r["run_len"]),
                    "check_fc_before_core_majority": int(r["prop_fc_before_or_eq_core"] >= 0.5),
                    "check_collapse_before_core_majority": int(
                        np.isfinite(r["prop_collapse_before_or_eq_core"]) and (r["prop_collapse_before_or_eq_core"] >= 0.5)
                    ),
                    "check_core_ref_rank_agreement": int(np.isfinite(r["rho_core_vs_ref"]) and (r["rho_core_vs_ref"] >= 0.6)),
                    "note": "",
                }
            )
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(OUT_CHECKS, index=False)

    print(f"saved: {OUT_BLOCK}")
    print(f"saved: {OUT_SUM}")
    print(f"saved: {OUT_CHECKS}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
