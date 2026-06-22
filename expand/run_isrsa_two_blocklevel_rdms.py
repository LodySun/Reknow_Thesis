import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


BASE = "base_dir"
SOL = os.path.join(BASE, "solidity")
EXP = os.path.join(BASE, "expand")

BLOCK_CSV = os.path.join(SOL, "hmm_unified_block_metrics.csv")
STAGE_CSV = os.path.join(SOL, "eeg_acq_sequence_stage_means_long.csv")

OUT_BEH = os.path.join(EXP, "isrsa_two_blocklevel_behavior_subject_features.csv")
OUT_NEU = os.path.join(EXP, "isrsa_two_blocklevel_neural_subject_features.csv")
OUT_RESULTS = os.path.join(EXP, "isrsa_two_blocklevel_rdm_results.csv")

N_PERM = 10000
BLOCK_NUMS = list(range(13, 109))
NEURAL_FEATURES = [
    "neu_p3b_transition_fc_minus_acquired_core",
    "neu_frn_p3b_balance_fc",
]


@dataclass
class Spec:
    name: str
    cols: list[str]
    note: str


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    idx = np.where(ok)[0]
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adj = np.empty(m, dtype=float)
    adj[-1] = ranked[-1]
    for i in range(m - 2, -1, -1):
        adj[i] = min(ranked[i] * m / (i + 1), adj[i + 1])
    restored = np.empty(m, dtype=float)
    restored[order] = np.clip(adj, 0.0, 1.0)
    out[idx] = restored
    return out


def _zscore_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        x = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x)
        if finite.sum() == 0:
            out[col] = 0.0
            continue
        mu = np.nanmean(x)
        sd = np.nanstd(x, ddof=1)
        out[col] = (x - mu) / sd if np.isfinite(sd) and sd > 0 else 0.0
    return out


def _upper(mat: np.ndarray) -> np.ndarray:
    return mat[np.triu_indices_from(mat, k=1)]


def _nan_euclidean_rdm(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    n, p = x.shape
    out = np.full((n, n), np.nan, dtype=float)
    n_valid = np.zeros((n, n), dtype=int)
    for i in range(n):
        out[i, i] = 0.0
        n_valid[i, i] = int(np.isfinite(x[i]).sum())
        for j in range(i + 1, n):
            ok = np.isfinite(x[i]) & np.isfinite(x[j])
            n_ok = int(ok.sum())
            n_valid[i, j] = n_ok
            n_valid[j, i] = n_ok
            if n_ok == 0:
                continue
            diff = x[i, ok] - x[j, ok]
            dist = np.sqrt(np.sum(diff * diff) * p / n_ok)
            out[i, j] = dist
            out[j, i] = dist
    return out, n_valid


def _perm_test(beh_rdm: np.ndarray, neu_rdm: np.ndarray, n_perm: int, seed: int) -> tuple[float, float]:
    vb = _upper(beh_rdm)
    vn = _upper(neu_rdm)
    ok = np.isfinite(vb) & np.isfinite(vn)
    rho, _ = spearmanr(vb[ok], vn[ok])
    rng = np.random.default_rng(seed)
    n = beh_rdm.shape[0]
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(n)
        vp = _upper(neu_rdm[np.ix_(perm, perm)])
        okp = np.isfinite(vb) & np.isfinite(vp)
        null[i], _ = spearmanr(vb[okp], vp[okp])
    p_perm = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n_perm + 1))
    return float(rho), p_perm


def build_behavior_block_trajectory(block: pd.DataFrame) -> pd.DataFrame:
    d = block.copy()
    d["subj"] = d["subj"].astype(str)
    for col in ["block_num", "acquisition_lag_core", "transition_width_20_to_80"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d[d["block_num"].isin(BLOCK_NUMS)].copy()
    d["transition_nonzero"] = ((d["acquisition_lag_core"] > 0) | (d["transition_width_20_to_80"] > 0)).astype(float)
    d["lag_log1p"] = np.log1p(np.clip(d["acquisition_lag_core"].to_numpy(dtype=float), 0.0, None))

    rows = []
    for subj, g in d.groupby("subj", sort=True):
        row = {"subj": subj}
        for block_num in BLOCK_NUMS:
            s = g[g["block_num"] == block_num]
            row[f"block{block_num}_transition_nonzero"] = float(s["transition_nonzero"].iloc[0]) if len(s) else np.nan
            row[f"block{block_num}_lag_log1p"] = float(s["lag_log1p"].iloc[0]) if len(s) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_neural_block_trajectory(stage: pd.DataFrame) -> pd.DataFrame:
    s = stage.copy()
    s["subj"] = s["subj"].astype(str)
    s["block_id"] = s["block_id"].astype(str)
    for col in ["block_id", "value_mean"]:
        s[col] = pd.to_numeric(s[col], errors="coerce")
    s = s[s["block_id"].isin(BLOCK_NUMS)].copy()
    s["block_num"] = s["block_id"].astype(int)

    piv = s.pivot_table(index=["subj", "block_num"], columns=["feature", "category"], values="value_mean", aggfunc="first")
    idx = piv.index

    def col(feat: str, cat: str) -> pd.Series:
        if (feat, cat) in piv.columns:
            return piv[(feat, cat)]
        return pd.Series(np.nan, index=idx)

    block_level = pd.DataFrame(
        {
            "subj": [i[0] for i in idx],
            "block_num": [i[1] for i in idx],
            "neu_p3b_transition_fc_minus_acquired_core": (
                col("feedback_locked_P3b", "transition_first_correct") - col("feedback_locked_P3b", "acquired_core")
            ).to_numpy(dtype=float),
            "neu_frn_p3b_balance_fc": (
                col("feedback_locked_FRN", "transition_first_correct") - col("feedback_locked_P3b", "transition_first_correct")
            ).to_numpy(dtype=float),
        }
    )

    rows = []
    for subj, g in block_level.groupby("subj", sort=True):
        row = {"subj": subj}
        for block_num in BLOCK_NUMS:
            sblock = g[g["block_num"] == block_num]
            for feat in NEURAL_FEATURES:
                row[f"block{block_num}_{feat}"] = float(sblock[feat].iloc[0]) if len(sblock) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _coverage(n_valid: np.ndarray) -> tuple[int, float, int]:
    vals = _upper(n_valid.astype(float))
    vals = vals[np.isfinite(vals)]
    return int(np.nanmin(vals)), float(np.nanmedian(vals)), int(np.nanmax(vals))


def main() -> None:
    os.makedirs(EXP, exist_ok=True)
    block = pd.read_csv(BLOCK_CSV)
    stage = pd.read_csv(STAGE_CSV)

    beh = build_behavior_block_trajectory(block)
    neu = build_neural_block_trajectory(stage)
    beh.to_csv(OUT_BEH, index=False)
    neu.to_csv(OUT_NEU, index=False)

    nonzero_cols = [f"block{i}_transition_nonzero" for i in BLOCK_NUMS]
    lag_cols = [f"block{i}_lag_log1p" for i in BLOCK_NUMS]
    neu_cols = [f"block{i}_{feat}" for i in BLOCK_NUMS for feat in NEURAL_FEATURES]
    beh_specs = [
        Spec("block_nonzero_96block_nan_euclidean", nonzero_cols, "Primary: 96-block delayed-transition presence trajectory"),
        Spec("block_loglag_96block_nan_euclidean", lag_cols, "Sensitivity: 96-block log1p acquisition-lag trajectory"),
    ]
    neural_note = "Block-level neural trajectory: P3b transition first-correct minus acquired core plus FRN/P3b balance at first correct"

    rows = []
    for i, bspec in enumerate(beh_specs):
        d = beh[["subj"] + bspec.cols].merge(neu[["subj"] + neu_cols], on="subj", how="inner")
        d = d.dropna(subset=bspec.cols + neu_cols, how="all").reset_index(drop=True)
        bz = _zscore_cols(d[["subj"] + bspec.cols], bspec.cols)
        nz = _zscore_cols(d[["subj"] + neu_cols], neu_cols)
        brdm, bvalid = _nan_euclidean_rdm(bz[bspec.cols].to_numpy(dtype=float))
        nrdm, nvalid = _nan_euclidean_rdm(nz[neu_cols].to_numpy(dtype=float))
        rho, p_perm = _perm_test(brdm, nrdm, n_perm=N_PERM, seed=20260428 + i * 101)
        bmin, bmed, bmax = _coverage(bvalid)
        nmin, nmed, nmax = _coverage(nvalid)
        rows.append(
            {
                "behavior_rdm": bspec.name,
                "neural_rdm": "neu_block_acquired_core_2feature_trajectory",
                "n_subjects": int(len(d)),
                "rho_spearman_uppertri": rho,
                "p_perm": p_perm,
                "behavior_features": "|".join(bspec.cols),
                "neural_features": "|".join(neu_cols),
                "behavior_note": bspec.note,
                "neural_note": neural_note,
                "behavior_pairwise_valid_min": bmin,
                "behavior_pairwise_valid_median": bmed,
                "behavior_pairwise_valid_max": bmax,
                "neural_pairwise_valid_min": nmin,
                "neural_pairwise_valid_median": nmed,
                "neural_pairwise_valid_max": nmax,
            }
        )

    out = pd.DataFrame(rows)
    out["p_fdr_two"] = _bh_fdr(out["p_perm"].to_numpy(dtype=float))
    out.to_csv(OUT_RESULTS, index=False)

    print(f"saved: {OUT_BEH}")
    print(f"saved: {OUT_NEU}")
    print(f"saved: {OUT_RESULTS}")
    cols = [
        "behavior_rdm",
        "neural_rdm",
        "n_subjects",
        "rho_spearman_uppertri",
        "p_perm",
        "p_fdr_two",
        "behavior_pairwise_valid_median",
        "neural_pairwise_valid_median",
    ]
    print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
