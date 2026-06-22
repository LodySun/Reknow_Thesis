import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


BASE = "base_dir"
HMM_BLOCK_CSV = os.path.join(BASE, "hmm_unified_block_metrics.csv")
OUT_BLOCK = os.path.join(BASE, "hmm_based_search_metrics_block_level.csv")
OUT_SUBJ = os.path.join(BASE, "hmm_based_search_metrics_subject_level.csv")
OUT_TEST = os.path.join(BASE, "hmm_based_search_metrics_within_vs_cross_tests.csv")


def _paired_stats(df: pd.DataFrame, col_within: str, col_cross: str, metric: str) -> dict:
    m = df[["subj", col_within, col_cross]].dropna().copy()
    if len(m) < 8:
        return {
            "metric": metric,
            "n_subjects": int(len(m)),
            "mean_within": np.nan,
            "mean_cross": np.nan,
            "mean_cross_minus_within": np.nan,
            "t": np.nan,
            "p": np.nan,
            "dz": np.nan,
        }
    x = pd.to_numeric(m[col_within], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(m[col_cross], errors="coerce").to_numpy(dtype=float)
    t, p = ttest_rel(y, x, nan_policy="omit")
    d = y - x
    sd = np.nanstd(d, ddof=1)
    dz = float(np.nanmean(d) / sd) if np.isfinite(sd) and sd > 0 else np.nan
    return {
        "metric": metric,
        "n_subjects": int(len(m)),
        "mean_within": float(np.nanmean(x)),
        "mean_cross": float(np.nanmean(y)),
        "mean_cross_minus_within": float(np.nanmean(d)),
        "t": float(t),
        "p": float(p),
        "dz": dz,
    }


def main() -> None:
    d = pd.read_csv(HMM_BLOCK_CSV)
    d["subj"] = d["subj"].astype(str)
    d["block_id"] = d["block_id"].astype(str)
    d["shift_type"] = d["shift_type"].astype(str)
    d["set_index"] = pd.to_numeric(d["set_index"], errors="coerce")
    d["first_correct_trial"] = pd.to_numeric(d["first_correct_trial"], errors="coerce")
    d["acquisition_trial_core"] = pd.to_numeric(d["acquisition_trial_core"], errors="coerce")

    # first_correct_latency keeps the original first-correct timing definition.
    d["first_correct_latency"] = d["first_correct_trial"]
    # Updated definition requested by user:
    # search length is the number of trials before entering core acquired state.
    d["search_len"] = d["acquisition_trial_core"] - 1.0

    # Keep blocks with valid first-correct latency and valid core-derived search length.
    d = d[
        np.isfinite(d["first_correct_latency"])
        & (d["first_correct_latency"] >= 1)
        & np.isfinite(d["search_len"])
        & (d["search_len"] >= 0)
    ].copy()

    # Save detailed block table (including first_block).
    block_cols = [
        "subj",
        "block_id",
        "set_index",
        "shift_type",
        "first_correct_trial",
        "acquisition_trial_core",
        "first_correct_latency",
        "search_len",
    ]
    d[block_cols].to_csv(OUT_BLOCK, index=False)

    # Subject summaries: all blocks + within/cross split.
    all_subj = (
        d.groupby("subj", as_index=False)
        .agg(
            first_correct_latency_mean_all=("first_correct_latency", "mean"),
            search_len_mean_all=("search_len", "mean"),
            n_blocks_all=("block_id", "count"),
        )
    )

    dx = d[d["shift_type"].isin(["within_level", "cross_level"])].copy()
    split = (
        dx.groupby(["subj", "shift_type"], as_index=False)
        .agg(
            first_correct_latency_mean=("first_correct_latency", "mean"),
            search_len_mean=("search_len", "mean"),
            n_blocks=("block_id", "count"),
        )
        .pivot(index="subj", columns="shift_type")
    )
    split.columns = [f"{c1}_{c2}" for c1, c2 in split.columns]
    split = split.reset_index()

    subj = all_subj.merge(split, on="subj", how="left")
    for m in ["first_correct_latency_mean", "search_len_mean"]:
        w = f"{m}_within_level"
        c = f"{m}_cross_level"
        subj[f"{m}_cross_minus_within"] = pd.to_numeric(subj[c], errors="coerce") - pd.to_numeric(subj[w], errors="coerce")
    subj.to_csv(OUT_SUBJ, index=False)

    # Group paired tests (within vs cross), subject as repeated unit.
    rows = []
    rows.append(
        _paired_stats(
            subj,
            "first_correct_latency_mean_within_level",
            "first_correct_latency_mean_cross_level",
            "first_correct_latency",
        )
    )
    rows.append(
        _paired_stats(
            subj,
            "search_len_mean_within_level",
            "search_len_mean_cross_level",
            "search_len",
        )
    )
    test_df = pd.DataFrame(rows)
    test_df.to_csv(OUT_TEST, index=False)

    print(f"saved: {OUT_BLOCK}")
    print(f"saved: {OUT_SUBJ}")
    print(f"saved: {OUT_TEST}")
    print(test_df.to_string(index=False))


if __name__ == "__main__":
    main()
