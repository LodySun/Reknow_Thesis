import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


BASE = "base_dir"
EXP = os.path.join(BASE, "expand")

BEH_CSV = os.path.join(EXP, "isrsa_two_blocklevel_behavior_subject_features.csv")
NEU_CSV = os.path.join(EXP, "isrsa_two_blocklevel_neural_subject_features.csv")

OUT_RESULTS = os.path.join(EXP, "isrsa_two_blocklevel_sensitivity_results.csv")
OUT_LOO = os.path.join(EXP, "isrsa_two_blocklevel_leaveoneout.csv")
OUT_SUMMARY = os.path.join(EXP, "isrsa_two_blocklevel_sensitivity_summary.csv")

N_PERM = 10000
BLOCK_NUMS = list(range(13, 109))
NEURAL_FEATURES = [
    "neu_p3b_transition_fc_minus_acquired_core",
    "neu_frn_p3b_balance_fc",
]


@dataclass
class Analysis:
    check: str
    behavior_rdm: str
    neural_rdm: str
    behavior_cols: list[str]
    neural_cols: list[str]
    behavior_metric: str
    neural_metric: str
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


def _nan_hamming_rdm(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    n = x.shape[0]
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
            dist = float(np.mean(x[i, ok] != x[j, ok]))
            out[i, j] = dist
            out[j, i] = dist
    return out, n_valid


def _nan_jaccard_rdm(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    n = x.shape[0]
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
            a = x[i, ok] > 0
            b = x[j, ok] > 0
            union = np.logical_or(a, b).sum()
            dist = 0.0 if union == 0 else 1.0 - float(np.logical_and(a, b).sum() / union)
            out[i, j] = dist
            out[j, i] = dist
    return out, n_valid


def _nan_spearman_distance_rdm(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    n = x.shape[0]
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
            if n_ok < 3:
                continue
            xi = x[i, ok]
            xj = x[j, ok]
            if np.nanstd(xi) == 0 and np.nanstd(xj) == 0:
                rho = 1.0 if np.allclose(xi, xj) else 0.0
            elif np.nanstd(xi) == 0 or np.nanstd(xj) == 0:
                rho = 0.0
            else:
                rho, _ = spearmanr(xi, xj)
            out[i, j] = 1.0 - float(rho)
            out[j, i] = out[i, j]
    return out, n_valid


def _rdm(values: np.ndarray, metric: str) -> tuple[np.ndarray, np.ndarray]:
    if metric == "nan_euclidean":
        return _nan_euclidean_rdm(values)
    if metric == "nan_hamming":
        return _nan_hamming_rdm(values)
    if metric == "nan_jaccard":
        return _nan_jaccard_rdm(values)
    if metric == "nan_spearman_distance":
        return _nan_spearman_distance_rdm(values)
    raise ValueError(f"unknown metric: {metric}")


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


def _coverage(n_valid: np.ndarray) -> tuple[int, float, int]:
    vals = _upper(n_valid.astype(float))
    vals = vals[np.isfinite(vals)]
    return int(np.nanmin(vals)), float(np.nanmedian(vals)), int(np.nanmax(vals))


def _block_cols(prefix: str, suffix: str) -> list[str]:
    return [f"block{i}_{suffix}" if prefix == "" else f"block{i}_{prefix}" for i in BLOCK_NUMS]


def _neural_cols(features: list[str], blocks: list[int] = None) -> list[str]:
    use_blocks = BLOCK_NUMS if blocks is None else blocks
    return [f"block{i}_{feat}" for i in use_blocks for feat in features]


def _complete_blocks(d: pd.DataFrame, beh_suffix: str, neural_features: list[str]) -> list[int]:
    blocks = []
    for block_num in BLOCK_NUMS:
        cols = [f"block{block_num}_{beh_suffix}"] + [f"block{block_num}_{feat}" for feat in neural_features]
        if all(col in d.columns for col in cols) and d[cols].notna().all().all():
            blocks.append(block_num)
    return blocks


def _make_analyses(d: pd.DataFrame) -> list[Analysis]:
    nonzero_cols = [f"block{i}_transition_nonzero" for i in BLOCK_NUMS]
    lag_cols = [f"block{i}_lag_log1p" for i in BLOCK_NUMS]
    full_neu_cols = _neural_cols(NEURAL_FEATURES)
    p3b_cols = _neural_cols(["neu_p3b_transition_fc_minus_acquired_core"])
    balance_cols = _neural_cols(["neu_frn_p3b_balance_fc"])

    nonzero_complete_blocks = _complete_blocks(d, "transition_nonzero", NEURAL_FEATURES)
    lag_complete_blocks = _complete_blocks(d, "lag_log1p", NEURAL_FEATURES)

    analyses = [
        Analysis(
            "primary_recomputed",
            "block_nonzero_96block_nan_euclidean",
            "neu_block_2feature_nan_euclidean",
            nonzero_cols,
            full_neu_cols,
            "nan_euclidean",
            "nan_euclidean",
            "Primary block-level nonzero trajectory, recomputed.",
        ),
        Analysis(
            "primary_recomputed",
            "block_loglag_96block_nan_euclidean",
            "neu_block_2feature_nan_euclidean",
            lag_cols,
            full_neu_cols,
            "nan_euclidean",
            "nan_euclidean",
            "Primary block-level log1p lag trajectory, recomputed.",
        ),
        Analysis(
            "missingness_complete_blocks",
            f"block_nonzero_complete{len(nonzero_complete_blocks)}block_euclidean",
            f"neu_block_2feature_complete{len(nonzero_complete_blocks)}block_euclidean",
            [f"block{i}_transition_nonzero" for i in nonzero_complete_blocks],
            _neural_cols(NEURAL_FEATURES, nonzero_complete_blocks),
            "nan_euclidean",
            "nan_euclidean",
            "Complete-case blocks only: no behavior or neural missingness in retained block dimensions.",
        ),
        Analysis(
            "missingness_complete_blocks",
            f"block_loglag_complete{len(lag_complete_blocks)}block_euclidean",
            f"neu_block_2feature_complete{len(lag_complete_blocks)}block_euclidean",
            [f"block{i}_lag_log1p" for i in lag_complete_blocks],
            _neural_cols(NEURAL_FEATURES, lag_complete_blocks),
            "nan_euclidean",
            "nan_euclidean",
            "Complete-case blocks only: no behavior or neural missingness in retained block dimensions.",
        ),
        Analysis(
            "metric_binary",
            "block_nonzero_96block_hamming",
            "neu_block_2feature_nan_euclidean",
            nonzero_cols,
            full_neu_cols,
            "nan_hamming",
            "nan_euclidean",
            "Binary transition trajectory using Hamming distance.",
        ),
        Analysis(
            "metric_binary",
            "block_nonzero_96block_jaccard",
            "neu_block_2feature_nan_euclidean",
            nonzero_cols,
            full_neu_cols,
            "nan_jaccard",
            "nan_euclidean",
            "Binary transition trajectory using Jaccard distance.",
        ),
        Analysis(
            "metric_lag",
            "block_loglag_96block_spearman_distance",
            "neu_block_2feature_nan_euclidean",
            lag_cols,
            full_neu_cols,
            "nan_spearman_distance",
            "nan_euclidean",
            "Log1p lag trajectory using pairwise Spearman distance.",
        ),
        Analysis(
            "neural_feature_dominance",
            "block_nonzero_96block_nan_euclidean",
            "neu_block_p3b_only_nan_euclidean",
            nonzero_cols,
            p3b_cols,
            "nan_euclidean",
            "nan_euclidean",
            "P3b transition first-correct minus acquired core only.",
        ),
        Analysis(
            "neural_feature_dominance",
            "block_nonzero_96block_nan_euclidean",
            "neu_block_balance_only_nan_euclidean",
            nonzero_cols,
            balance_cols,
            "nan_euclidean",
            "nan_euclidean",
            "FRN/P3b balance at first correct only.",
        ),
        Analysis(
            "neural_feature_dominance",
            "block_loglag_96block_nan_euclidean",
            "neu_block_p3b_only_nan_euclidean",
            lag_cols,
            p3b_cols,
            "nan_euclidean",
            "nan_euclidean",
            "P3b transition first-correct minus acquired core only.",
        ),
        Analysis(
            "neural_feature_dominance",
            "block_loglag_96block_nan_euclidean",
            "neu_block_balance_only_nan_euclidean",
            lag_cols,
            balance_cols,
            "nan_euclidean",
            "nan_euclidean",
            "FRN/P3b balance at first correct only.",
        ),
    ]
    return analyses


def _run_one(d: pd.DataFrame, analysis: Analysis, seed: int) -> dict:
    use_cols = ["subj"] + analysis.behavior_cols + analysis.neural_cols
    dd = d[use_cols].dropna(subset=analysis.behavior_cols + analysis.neural_cols, how="all").reset_index(drop=True)
    bz = _zscore_cols(dd[["subj"] + analysis.behavior_cols], analysis.behavior_cols)
    nz = _zscore_cols(dd[["subj"] + analysis.neural_cols], analysis.neural_cols)
    brdm, bvalid = _rdm(bz[analysis.behavior_cols].to_numpy(dtype=float), analysis.behavior_metric)
    nrdm, nvalid = _rdm(nz[analysis.neural_cols].to_numpy(dtype=float), analysis.neural_metric)
    rho, p_perm = _perm_test(brdm, nrdm, N_PERM, seed)
    bmin, bmed, bmax = _coverage(bvalid)
    nmin, nmed, nmax = _coverage(nvalid)
    return {
        "check": analysis.check,
        "behavior_rdm": analysis.behavior_rdm,
        "neural_rdm": analysis.neural_rdm,
        "n_subjects": int(len(dd)),
        "rho_spearman_uppertri": rho,
        "p_perm": p_perm,
        "behavior_metric": analysis.behavior_metric,
        "neural_metric": analysis.neural_metric,
        "behavior_n_features": len(analysis.behavior_cols),
        "neural_n_features": len(analysis.neural_cols),
        "behavior_pairwise_valid_min": bmin,
        "behavior_pairwise_valid_median": bmed,
        "behavior_pairwise_valid_max": bmax,
        "neural_pairwise_valid_min": nmin,
        "neural_pairwise_valid_median": nmed,
        "neural_pairwise_valid_max": nmax,
        "note": analysis.note,
    }


def _leave_one_out(d: pd.DataFrame, analyses: list[Analysis]) -> pd.DataFrame:
    primary = [a for a in analyses if a.check == "primary_recomputed"]
    rows = []
    for analysis in primary:
        for subj in sorted(d["subj"].unique()):
            dd = d[d["subj"] != subj].reset_index(drop=True)
            use_cols = ["subj"] + analysis.behavior_cols + analysis.neural_cols
            dd = dd[use_cols].dropna(subset=analysis.behavior_cols + analysis.neural_cols, how="all").reset_index(drop=True)
            bz = _zscore_cols(dd[["subj"] + analysis.behavior_cols], analysis.behavior_cols)
            nz = _zscore_cols(dd[["subj"] + analysis.neural_cols], analysis.neural_cols)
            brdm, _ = _rdm(bz[analysis.behavior_cols].to_numpy(dtype=float), analysis.behavior_metric)
            nrdm, _ = _rdm(nz[analysis.neural_cols].to_numpy(dtype=float), analysis.neural_metric)
            vb = _upper(brdm)
            vn = _upper(nrdm)
            ok = np.isfinite(vb) & np.isfinite(vn)
            rho, _ = spearmanr(vb[ok], vn[ok])
            rows.append(
                {
                    "behavior_rdm": analysis.behavior_rdm,
                    "neural_rdm": analysis.neural_rdm,
                    "left_out_subj": subj,
                    "n_subjects": int(len(dd)),
                    "rho_spearman_uppertri": float(rho),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    beh = pd.read_csv(BEH_CSV)
    neu = pd.read_csv(NEU_CSV)
    beh["subj"] = beh["subj"].astype(str)
    neu["subj"] = neu["subj"].astype(str)
    d = beh.merge(neu, on="subj", how="inner")

    analyses = _make_analyses(d)
    rows = []
    for i, analysis in enumerate(analyses):
        rows.append(_run_one(d, analysis, seed=20260428 + i * 113))

    out = pd.DataFrame(rows)
    out["p_fdr_all_sensitivity"] = _bh_fdr(out["p_perm"].to_numpy(dtype=float))
    out["p_fdr_within_check"] = np.nan
    for check, idx in out.groupby("check").groups.items():
        out.loc[idx, "p_fdr_within_check"] = _bh_fdr(out.loc[idx, "p_perm"].to_numpy(dtype=float))
    out.to_csv(OUT_RESULTS, index=False)

    loo = _leave_one_out(d, analyses)
    loo.to_csv(OUT_LOO, index=False)
    summary = (
        loo.groupby(["behavior_rdm", "neural_rdm"], as_index=False)
        .agg(
            loo_rho_min=("rho_spearman_uppertri", "min"),
            loo_rho_median=("rho_spearman_uppertri", "median"),
            loo_rho_max=("rho_spearman_uppertri", "max"),
        )
    )
    summary.to_csv(OUT_SUMMARY, index=False)

    print(f"saved: {OUT_RESULTS}")
    print(f"saved: {OUT_LOO}")
    print(f"saved: {OUT_SUMMARY}")
    cols = [
        "check",
        "behavior_rdm",
        "neural_rdm",
        "n_subjects",
        "rho_spearman_uppertri",
        "p_perm",
        "p_fdr_within_check",
    ]
    print(out[cols].to_string(index=False))
    print("\nLeave-one-out summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
