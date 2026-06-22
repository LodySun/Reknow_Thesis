import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, spearmanr


BASE = "base_dir"
HMM_LONG = os.path.join(BASE, "trials_trialwise", "hmm_long_tables", "hmm_trial_long.csv")
POST = os.path.join(BASE, "trials_trialwise", "1s_comp", "eeg_paper_results", "solidity", "hmm_unified_trial_posteriors.csv")
OUT_DIR = os.path.join(BASE, "trials_trialwise", "1s_comp", "eeg_paper_results", "expand", "marker_validity_collapse")

OUT_BLOCK = os.path.join(OUT_DIR, "collapse_methods_block_level.csv")
OUT_SUM = os.path.join(OUT_DIR, "collapse_methods_summary.csv")


def _safe_num(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _first_run_start(flag: np.ndarray, run_len: int = 2) -> float:
    # flag is boolean-like array in trial order
    if len(flag) == 0:
        return np.nan
    streak = 0
    for i, v in enumerate(flag):
        if bool(v):
            streak += 1
            if streak >= run_len:
                return float(i - run_len + 2)  # 1-based index
        else:
            streak = 0
    return np.nan


def _prob_collapse_onset(g: pd.DataFrame, tau: float = 0.8, min_run: int = 2) -> tuple[float, np.ndarray]:
    # Hidden state: collapsed (C) vs not-collapsed (N)
    # Observation: z_t = 1 if candidates_after<=1 else 0
    # Noise-tolerant emissions with lapse rate.
    # Use candidates_after as observation so the collapse timing reflects
    # hypothesis-space state AFTER integrating current-trial feedback.
    z = (pd.to_numeric(g["candidates_after"], errors="coerce").to_numpy(dtype=float) <= 1).astype(int)
    shrink = (
        pd.to_numeric(g["candidates_after"], errors="coerce").to_numpy(dtype=float)
        < pd.to_numeric(g["candidates_before"], errors="coerce").to_numpy(dtype=float)
    ).astype(int)
    miss = pd.to_numeric(g.get("missing_correctness_flag", 0), errors="coerce").fillna(0).to_numpy(dtype=float) > 0
    abn = pd.to_numeric(g.get("abnormal_response_flag", 0), errors="coerce").fillna(0).to_numpy(dtype=float) > 0
    noisy = miss | abn

    pC = 0.01
    post = np.full(len(z), np.nan, dtype=float)
    for t in range(len(z)):
        # Transition
        alpha = min(0.30, 0.02 + 0.22 * float(shrink[t]))  # N->C
        beta = 0.01  # C->N (rare reversals due to noise)
        pC_pred = pC * (1 - beta) + (1 - pC) * alpha

        # Emission with lapse
        eps = 0.10 if not noisy[t] else 0.25
        if z[t] == 1:
            likC = 1 - eps
            likN = eps
        else:
            likC = eps
            likN = 1 - eps
        num = likC * pC_pred
        den = num + likN * (1 - pC_pred)
        pC = num / den if den > 0 else pC_pred
        post[t] = pC

    onset = _first_run_start(post >= tau, run_len=min_run)
    return onset, post


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    h = pd.read_csv(HMM_LONG)
    p = pd.read_csv(POST)

    for df in [h, p]:
        df["subj"] = df["subj"].astype(str)
        df["block_id"] = df["block_id"].astype(str)
    _safe_num(
        h,
        [
            "trial_in_block",
            "candidates_before",
            "candidates_after",
            "missing_correctness_flag",
            "abnormal_response_flag",
            "correctness",
            "first_correct_flag",
        ],
    )
    _safe_num(p, ["first_correct_trial", "acquisition_trial_core"])

    # unified reference markers
    ref = (
        p.groupby(["subj", "block_id"], as_index=False)
        .agg(first_correct_trial=("first_correct_trial", "first"), acquisition_trial_core=("acquisition_trial_core", "first"))
        .dropna(subset=["first_correct_trial", "acquisition_trial_core"])
    )

    rows = []
    for (subj, bid), g in h.groupby(["subj", "block_id"], sort=False):
        g = g.sort_values("trial_in_block").copy()
        if g.empty:
            continue

        ref_row = ref[(ref["subj"] == subj) & (ref["block_id"] == bid)]
        if ref_row.empty:
            continue
        fc = float(ref_row.iloc[0]["first_correct_trial"])
        core = float(ref_row.iloc[0]["acquisition_trial_core"])

        z = (pd.to_numeric(g["candidates_after"], errors="coerce").to_numpy(dtype=float) <= 1).astype(int)
        valid = ~(
            (pd.to_numeric(g.get("missing_correctness_flag", 0), errors="coerce").fillna(0).to_numpy(dtype=float) > 0)
            | (pd.to_numeric(g.get("abnormal_response_flag", 0), errors="coerce").fillna(0).to_numpy(dtype=float) > 0)
        )

        # Hard original
        hard_idx = np.where(z == 1)[0]
        hard_collapse = float(hard_idx[0] + 1) if len(hard_idx) else np.nan

        # Hard robust: require valid run>=2
        z_valid = (z == 1) & valid
        hard_robust = _first_run_start(z_valid, run_len=2)

        # Probabilistic collapse
        prob_collapse_08, post08 = _prob_collapse_onset(g, tau=0.8, min_run=2)
        prob_collapse_09, _ = _prob_collapse_onset(g, tau=0.9, min_run=2)

        rows.append(
            {
                "subj": subj,
                "block_id": bid,
                "first_correct_trial": fc,
                "acquisition_trial_core": core,
                "collapse_hard": hard_collapse,
                "collapse_hard_robust": hard_robust,
                "collapse_prob_tau08": prob_collapse_08,
                "collapse_prob_tau09": prob_collapse_09,
                "lag_core_minus_hard": core - hard_collapse if np.isfinite(hard_collapse) else np.nan,
                "lag_core_minus_hard_robust": core - hard_robust if np.isfinite(hard_robust) else np.nan,
                "lag_core_minus_prob08": core - prob_collapse_08 if np.isfinite(prob_collapse_08) else np.nan,
                "lag_core_minus_prob09": core - prob_collapse_09 if np.isfinite(prob_collapse_09) else np.nan,
                "lag_fc_minus_hard": fc - hard_collapse if np.isfinite(hard_collapse) else np.nan,
                "lag_fc_minus_prob08": fc - prob_collapse_08 if np.isfinite(prob_collapse_08) else np.nan,
                "mean_post_prob08": float(np.nanmean(post08)) if len(post08) else np.nan,
            }
        )

    block = pd.DataFrame(rows)
    block.to_csv(OUT_BLOCK, index=False)

    # Summary and sensitivity
    sum_rows = []
    methods = {
        "hard": "collapse_hard",
        "hard_robust": "collapse_hard_robust",
        "prob_tau08": "collapse_prob_tau08",
        "prob_tau09": "collapse_prob_tau09",
    }
    for mname, col in methods.items():
        x = pd.to_numeric(block[col], errors="coerce")
        core = pd.to_numeric(block["acquisition_trial_core"], errors="coerce")
        fc = pd.to_numeric(block["first_correct_trial"], errors="coerce")
        ok = np.isfinite(x) & np.isfinite(core) & np.isfinite(fc)
        if ok.sum() < 10:
            continue
        lag_core = core[ok] - x[ok]
        lag_fc = fc[ok] - x[ok]
        rho_core, p_core = spearmanr(x[ok], core[ok], nan_policy="omit")
        sum_rows.append(
            {
                "method": mname,
                "n_blocks": int(ok.sum()),
                "collapse_mean_trial": float(np.nanmean(x[ok])),
                "lag_core_mean": float(np.nanmean(lag_core)),
                "lag_fc_mean": float(np.nanmean(lag_fc)),
                "prop_collapse_before_fc": float(np.mean((x[ok] <= fc[ok]))),
                "prop_collapse_before_core": float(np.mean((x[ok] <= core[ok]))),
                "rho_collapse_vs_core": float(rho_core) if np.isfinite(rho_core) else np.nan,
                "p_rho_collapse_vs_core": float(p_core) if np.isfinite(p_core) else np.nan,
            }
        )

    # Pairwise tests between methods for lag-to-core sensitivity.
    pair_defs = [
        ("lag_core_minus_hard", "lag_core_minus_hard_robust", "hard_vs_hardrobust"),
        ("lag_core_minus_hard", "lag_core_minus_prob08", "hard_vs_prob08"),
        ("lag_core_minus_hard_robust", "lag_core_minus_prob08", "hardrobust_vs_prob08"),
    ]
    for a, b, tag in pair_defs:
        m = block[[a, b]].dropna()
        if len(m) < 10:
            continue
        t, p = ttest_rel(m[a], m[b], nan_policy="omit")
        sum_rows.append(
            {
                "method": f"sensitivity_{tag}",
                "n_blocks": int(len(m)),
                "collapse_mean_trial": np.nan,
                "lag_core_mean": float((m[a] - m[b]).mean()),
                "lag_fc_mean": np.nan,
                "prop_collapse_before_fc": np.nan,
                "prop_collapse_before_core": np.nan,
                "rho_collapse_vs_core": float(t),
                "p_rho_collapse_vs_core": float(p),
            }
        )

    summary = pd.DataFrame(sum_rows)
    summary.to_csv(OUT_SUM, index=False)

    print(f"saved: {OUT_BLOCK}")
    print(f"saved: {OUT_SUM}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
