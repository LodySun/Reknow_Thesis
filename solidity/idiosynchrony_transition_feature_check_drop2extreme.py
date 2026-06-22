import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


BASE = "base_dir"
BLOCK_CSV = os.path.join(BASE, "hmm_unified_block_metrics.csv")
STAGE_CSV = os.path.join(BASE, "eeg_acq_sequence_stage_means_long.csv")
OUT_DIR = BASE

# Remove two extreme blocks identified in sensitivity checks.
EXCLUDED_BLOCKS = {("reknow018", "21"), ("reknow029", "39")}

PRIMARY_NEURAL_FEATURES = [
    "neu_p3b_transition_fc_minus_precore",
    "neu_frn_p3b_balance_fc",
]
SECONDARY_NEURAL_FEATURES = [
    "neu_p3b_transition_fc_minus_acquired_core",
]
EXPLORATORY_NEURAL_FEATURES = [
    "neu_frn_transition_fc_minus_precore",
]


def _drop_extreme_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["subj"] = out["subj"].astype(str)
    out["block_id"] = out["block_id"].astype(str)
    out["sb_key"] = out["subj"] + "|" + out["block_id"]
    drop_keys = {f"{s}|{b}" for s, b in EXCLUDED_BLOCKS}
    return out[~out["sb_key"].isin(drop_keys)].copy()


def _split_half_r(block_df: pd.DataFrame, value_col: str) -> float:
    d = block_df[["subj", "block_num", value_col]].dropna().copy()
    if d.empty:
        return np.nan
    d["half"] = np.where((pd.to_numeric(d["block_num"], errors="coerce").astype("Int64") % 2) == 0, "even", "odd")
    piv = (
        d.groupby(["subj", "half"], as_index=False)[value_col]
        .mean()
        .pivot_table(index="subj", columns="half", values=value_col)
    )
    if ("odd" not in piv.columns) or ("even" not in piv.columns):
        return np.nan
    x = piv["odd"].to_numpy(dtype=float)
    y = piv["even"].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def _build_behavior_features(block: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = block.copy()
    for c in ["set_index", "acquisition_lag_core", "transition_width_20_to_80", "collapse_to_acquisition_lag_proxy", "block_num"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["set_index"].isin([1, 2])].copy()

    subj = (
        d.groupby("subj", as_index=False)
        .agg(
            beh_lag_mean=("acquisition_lag_core", "mean"),
            beh_lag_nonzero_prop=("acquisition_lag_core", lambda x: float((pd.to_numeric(x, errors="coerce") > 0).mean())),
            beh_width_mean=("transition_width_20_to_80", "mean"),
            beh_width_nonzero_prop=("transition_width_20_to_80", lambda x: float((pd.to_numeric(x, errors="coerce") > 0).mean())),
            beh_proxy_collapse_lag_mean=("collapse_to_acquisition_lag_proxy", "mean"),
            beh_proxy_collapse_lag_sd=("collapse_to_acquisition_lag_proxy", "std"),
            n_blocks_early12=("block_id", "count"),
        )
    )

    rel_rows = []
    for col in ["acquisition_lag_core", "transition_width_20_to_80", "collapse_to_acquisition_lag_proxy"]:
        rel_rows.append(
            {
                "domain": "behavior",
                "feature": col,
                "split_half_r": _split_half_r(d, col),
                "subject_sd": float(
                    pd.to_numeric(
                        subj[
                            "beh_lag_mean"
                            if col == "acquisition_lag_core"
                            else "beh_width_mean"
                            if col == "transition_width_20_to_80"
                            else "beh_proxy_collapse_lag_mean"
                        ],
                        errors="coerce",
                    ).std()
                ),
            }
        )
    return subj, pd.DataFrame(rel_rows)


def _build_neural_features(stage: pd.DataFrame, block_lookup: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    s = stage.copy()
    for c in ["set_index", "value_mean"]:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s[s["set_index"].isin([1, 2])].copy()

    piv = s.pivot_table(index=["subj", "block_id"], columns=["feature", "category"], values="value_mean", aggfunc="first")
    idx = piv.index

    def col(feat: str, cat: str):
        return piv[(feat, cat)] if (feat, cat) in piv.columns else pd.Series(np.nan, index=idx)

    frn_fc = col("feedback_locked_FRN", "transition_first_correct")
    frn_precore = col("feedback_locked_FRN", "transition_pre_core")
    p3b_fc = col("feedback_locked_P3b", "transition_first_correct")
    p3b_precore = col("feedback_locked_P3b", "transition_pre_core")
    p3b_core = col("feedback_locked_P3b", "acquired_core")

    block_level = pd.DataFrame(
        {
            "neu_frn_transition_fc_minus_precore": frn_fc - frn_precore,
            "neu_p3b_transition_fc_minus_precore": p3b_fc - p3b_precore,
            "neu_p3b_transition_fc_minus_acquired_core": p3b_fc - p3b_core,
            "neu_frn_p3b_balance_fc": frn_fc - p3b_fc,
        },
        index=idx,
    ).reset_index()
    block_level["block_id"] = block_level["block_id"].astype(str)

    subj = block_level.groupby("subj", as_index=False).mean(numeric_only=True)

    sb = block_lookup[["subj", "block_id", "block_num"]].dropna().drop_duplicates()
    bl = block_level.merge(sb, on=["subj", "block_id"], how="left")

    rel_rows = []
    for coln in ["neu_frn_transition_fc_minus_precore", "neu_p3b_transition_fc_minus_precore", "neu_p3b_transition_fc_minus_acquired_core", "neu_frn_p3b_balance_fc"]:
        rel_rows.append(
            {
                "domain": "neural",
                "feature": coln,
                "split_half_r": _split_half_r(bl, coln),
                "subject_sd": float(pd.to_numeric(subj[coln], errors="coerce").std()),
            }
        )
    return subj, pd.DataFrame(rel_rows)


def _subject_corr_table(
    beh: pd.DataFrame,
    neu: pd.DataFrame,
    primary_behavior_only: bool = False,
    neural_subset: List[str] = None,
) -> pd.DataFrame:
    d = beh.merge(neu, on="subj", how="inner")
    if primary_behavior_only:
        bcols = [c for c in ["beh_lag_mean", "beh_lag_nonzero_prop", "beh_width_mean", "beh_width_nonzero_prop"] if c in d.columns]
    else:
        bcols = [c for c in d.columns if c.startswith("beh_")]
    ncols = [c for c in d.columns if c.startswith("neu_")]
    if neural_subset is not None:
        ncols = [c for c in ncols if c in set(neural_subset)]

    rows = []
    for bc in bcols:
        for nc in ncols:
            x = pd.to_numeric(d[bc], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(d[nc], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 10:
                continue
            rho, p = spearmanr(x[ok], y[ok], nan_policy="omit")
            rows.append({"behavior_feature": bc, "neural_feature": nc, "n_subjects": int(ok.sum()), "spearman_rho": float(rho), "p": float(p)})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    block = _drop_extreme_keys(pd.read_csv(BLOCK_CSV))
    stage = _drop_extreme_keys(pd.read_csv(STAGE_CSV))

    beh_subj, beh_rel = _build_behavior_features(block)
    block_lookup = block[["subj", "block_id", "block_num"]].copy()
    block_lookup["block_id"] = block_lookup["block_id"].astype(str)
    block_lookup["block_num"] = pd.to_numeric(block_lookup["block_num"], errors="coerce")
    neu_subj, neu_rel = _build_neural_features(stage, block_lookup)

    rel = pd.concat([beh_rel, neu_rel], ignore_index=True)
    corr_all = _subject_corr_table(beh_subj, neu_subj, primary_behavior_only=False, neural_subset=None)
    corr_primary = _subject_corr_table(beh_subj, neu_subj, primary_behavior_only=True, neural_subset=PRIMARY_NEURAL_FEATURES)
    corr_secondary = _subject_corr_table(beh_subj, neu_subj, primary_behavior_only=True, neural_subset=SECONDARY_NEURAL_FEATURES)
    corr_exploratory = _subject_corr_table(beh_subj, neu_subj, primary_behavior_only=True, neural_subset=EXPLORATORY_NEURAL_FEATURES)

    p1 = os.path.join(OUT_DIR, "idiosynchrony_transition_behavior_subject_features_drop2extreme.csv")
    p2 = os.path.join(OUT_DIR, "idiosynchrony_transition_neural_subject_features_drop2extreme.csv")
    p3 = os.path.join(OUT_DIR, "idiosynchrony_transition_reliability_summary_drop2extreme.csv")
    p4 = os.path.join(OUT_DIR, "idiosynchrony_transition_behavior_neural_correlations_drop2extreme.csv")
    p5 = os.path.join(OUT_DIR, "idiosynchrony_transition_primary_behavior_neural_correlations_drop2extreme.csv")
    p6 = os.path.join(OUT_DIR, "idiosynchrony_transition_secondary_behavior_neural_correlations_drop2extreme.csv")
    p7 = os.path.join(OUT_DIR, "idiosynchrony_transition_exploratory_behavior_neural_correlations_drop2extreme.csv")

    beh_subj.to_csv(p1, index=False)
    neu_subj.to_csv(p2, index=False)
    rel.to_csv(p3, index=False)
    corr_all.to_csv(p4, index=False)
    corr_primary.to_csv(p5, index=False)
    corr_secondary.to_csv(p6, index=False)
    corr_exploratory.to_csv(p7, index=False)

    print(f"saved: {p1}")
    print(f"saved: {p2}")
    print(f"saved: {p3}")
    print(f"saved: {p4}")
    print(f"saved: {p5}")
    print(f"saved: {p6}")
    print(f"saved: {p7}")
    print(rel.to_string(index=False))
    print(corr_primary.sort_values("p").to_string(index=False) if not corr_primary.empty else "No valid primary behavior-neural correlations.")


if __name__ == "__main__":
    main()
