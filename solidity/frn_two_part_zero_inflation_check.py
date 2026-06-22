import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import kendalltau, spearmanr


BASE = "base_dir"
BLOCK_CSV = os.path.join(BASE, "hmm_unified_block_metrics.csv")
STAGE_CSV = os.path.join(BASE, "eeg_acq_sequence_stage_means_long.csv")
OUT_DIR = BASE


def _load_block_level_table() -> pd.DataFrame:
    block = pd.read_csv(BLOCK_CSV)
    stage = pd.read_csv(STAGE_CSV)

    # FRN at first-correct, block-level
    frn_fc = stage[
        (stage["feature"] == "feedback_locked_FRN")
        & (stage["category"] == "transition_first_correct")
    ][["subj", "block_id", "value_mean"]].copy()
    frn_fc = frn_fc.rename(columns={"value_mean": "frn_fc"})

    block["block_id"] = block["block_id"].astype(str)
    frn_fc["block_id"] = frn_fc["block_id"].astype(str)
    d = block.merge(frn_fc, on=["subj", "block_id"], how="inner")

    for c in ["lag_core", "width_20_80"]:
        if c in d.columns:
            continue
    d["lag_core"] = pd.to_numeric(d["acquisition_lag_core"], errors="coerce")
    d["width_20_80"] = pd.to_numeric(d["transition_width_20_to_80"], errors="coerce")
    d["frn_fc"] = pd.to_numeric(d["frn_fc"], errors="coerce")
    d["set_index"] = pd.to_numeric(d["set_index"], errors="coerce")
    d["lag_nonzero"] = (d["lag_core"] > 0).astype(float)
    d["width_nonzero"] = (d["width_20_80"] > 0).astype(float)
    d = d.dropna(subset=["frn_fc", "lag_core", "width_20_80", "set_index", "shift_type", "rule_level", "subj"]).copy()
    d = d[d["shift_type"] != "first_block"].copy()
    return d


def _safe_get(res, key: str):
    return float(res.params.get(key, np.nan)), float(res.bse.get(key, np.nan)), float(res.pvalues.get(key, np.nan))


def run_two_part(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    outcomes = [
        ("lag_nonzero", "lag_core"),
        ("width_nonzero", "width_20_80"),
    ]
    for y_bin, y_pos in outcomes:
        # Part 1: whether non-zero interval exists
        gee_bin = smf.gee(
            f"{y_bin} ~ frn_fc + set_index + C(shift_type) + C(rule_level)",
            groups="subj",
            data=d,
            family=sm.families.Binomial(),
        ).fit()
        b, se, p = _safe_get(gee_bin, "frn_fc")
        rows.append(
            {
                "analysis": "part1_presence",
                "outcome": y_bin,
                "model": "GEE_Binomial",
                "n_blocks": int(gee_bin.nobs),
                "beta_frn": b,
                "se_frn": se,
                "p_frn": p,
            }
        )

        # Part 2: conditional magnitude among positive-only blocks
        d_pos = d[d[y_pos] > 0].copy()
        if len(d_pos) >= 50:
            nb = smf.glm(
                f"{y_pos} ~ frn_fc + set_index + C(shift_type) + C(rule_level)",
                data=d_pos,
                family=sm.families.NegativeBinomial(),
            ).fit(cov_type="cluster", cov_kwds={"groups": d_pos["subj"]})
            b, se, p = _safe_get(nb, "frn_fc")
            rows.append(
                {
                    "analysis": "part2_positive_only",
                    "outcome": y_pos,
                    "model": "GLM_NB_cluster",
                    "n_blocks": int(nb.nobs),
                    "beta_frn": b,
                    "se_frn": se,
                    "p_frn": p,
                }
            )

            # rank-based sensitivity for positive-only
            rho, p_s = spearmanr(d_pos["frn_fc"], d_pos[y_pos], nan_policy="omit")
            tau, p_k = kendalltau(d_pos["frn_fc"], d_pos[y_pos], nan_policy="omit")
            rows.append(
                {
                    "analysis": "part2_positive_only_rank",
                    "outcome": y_pos,
                    "model": "Spearman_Kendall",
                    "n_blocks": int(len(d_pos)),
                    "beta_frn": float(rho),
                    "se_frn": float(tau),
                    "p_frn": float(p_s),
                    "p_aux": float(p_k),
                }
            )
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    d = _load_block_level_table()
    out_main = run_two_part(d)

    # basic distribution check
    dist = pd.DataFrame(
        [
            {
                "n_blocks": int(len(d)),
                "lag_zero_prop": float((d["lag_core"] == 0).mean()),
                "width_zero_prop": float((d["width_20_80"] == 0).mean()),
                "lag_positive_n": int((d["lag_core"] > 0).sum()),
                "width_positive_n": int((d["width_20_80"] > 0).sum()),
            }
        ]
    )

    p1 = os.path.join(OUT_DIR, "frn_two_part_zero_check_results.csv")
    p2 = os.path.join(OUT_DIR, "frn_two_part_zero_check_distribution.csv")
    out_main.to_csv(p1, index=False)
    dist.to_csv(p2, index=False)

    print(f"saved: {p1}")
    print(f"saved: {p2}")
    print(out_main.to_string(index=False))
    print(dist.to_string(index=False))


if __name__ == "__main__":
    main()
