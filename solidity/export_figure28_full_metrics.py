"""
Full inferential metrics for Figure 28 (current definitions):
  A) Acquired RT: HMM-core trials (trial_index >= acquisition_trial_core), contrast local - global (s).
  B) Acquired accuracy: same trials, contrast global - local (proportion correct).
  C) Pre-core uncertainty: trial_index < core, RT gap local-global; contrast high - low uncertainty.

Outputs:
  trials_trialwise/1s_comp/eeg_paper_results/solidity/figure28_global_precedence_full_metrics.csv
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_rel

BASE = "base_dir"
OUT_DIR = os.path.join(BASE, "trials_trialwise", "1s_comp", "eeg_paper_results", "solidity")
BEH_CSV = os.path.join(BASE, "trials_trialwise", "all_subjects_trialwise.csv")
HMM_LONG_CSV = os.path.join(BASE, "trials_trialwise", "hmm_long_tables", "hmm_trial_long.csv")
TRIAL_POST_CSV = os.path.join(OUT_DIR, "hmm_unified_trial_posteriors.csv")
OUT_CSV = os.path.join(OUT_DIR, "figure28_global_precedence_full_metrics.csv")


def _safe_num(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values for the family of tests (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    p_sorted = p[order]
    adj = np.empty(m, dtype=float)
    adj[m - 1] = float(p_sorted[m - 1])
    for i in range(m - 2, -1, -1):
        adj[i] = min(float(p_sorted[i]) * m / (i + 1), adj[i + 1])
    out = np.empty(m, dtype=float)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out


def _paired_diff_stats(
    a: pd.Series,
    b: pd.Series,
    contrast_label: str,
    *,
    panel: str,
    outcome: str,
    delta_definition: str,
) -> Optional[Dict[str, Any]]:
    m = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    n = int(len(m))
    if n < 3:
        return None
    d = (m["a"] - m["b"]).to_numpy(dtype=float)
    mean_diff = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    se = sd / np.sqrt(n)
    df = n - 1
    tcrit = float(student_t.ppf(0.975, df))
    ci_lo = mean_diff - tcrit * se
    ci_hi = mean_diff + tcrit * se
    t_stat, p_val = ttest_rel(m["a"], m["b"], nan_policy="omit")
    dz = mean_diff / sd if sd > 0 else float("nan")
    return {
        "panel": panel,
        "outcome": outcome,
        "contrast": contrast_label,
        "delta_definition": delta_definition,
        "n": n,
        "df": df,
        "delta": mean_diff,
        "ci95_low": float(ci_lo),
        "ci95_high": float(ci_hi),
        "t": float(t_stat),
        "p": float(p_val),
        "dz": float(dz),
    }


def main() -> None:
    beh = pd.read_csv(BEH_CSV)
    hmm = pd.read_csv(HMM_LONG_CSV)
    post = pd.read_csv(TRIAL_POST_CSV)
    beh["subj"] = beh["subj"].astype(str)
    beh["block_id"] = beh["block_id"].astype(str)
    hmm["subj"] = hmm["subj"].astype(str)
    hmm["block_id"] = hmm["block_id"].astype(str)
    post["subj"] = post["subj"].astype(str)
    post["block_id"] = post["block_id"].astype(str)
    _safe_num(beh, ["trial_id", "correctness", "rt"])
    _safe_num(hmm, ["trial_id", "candidates_before"])
    _safe_num(post, ["trial_id", "trial_index_1based", "acquisition_trial_core"])

    pcols = post[["subj", "block_id", "trial_id", "trial_index_1based", "acquisition_trial_core"]].dropna()
    d_acquired = beh.merge(pcols, on=["subj", "block_id", "trial_id"], how="inner")
    d_acquired = d_acquired[
        d_acquired["rule_level"].astype(str).isin(["global", "local"])
        & (
            pd.to_numeric(d_acquired["trial_index_1based"], errors="coerce")
            >= pd.to_numeric(d_acquired["acquisition_trial_core"], errors="coerce")
        )
    ].copy()

    rows: List[Dict[str, Any]] = []

    # Panel A: same order as plot _plot_paired significance: ttest_rel(global, local)
    g_rt = d_acquired.groupby(["subj", "rule_level"], as_index=False).agg(val=("rt", "mean")).dropna()
    wide_rt = g_rt.pivot(index="subj", columns="rule_level", values="val").dropna(subset=["global", "local"])
    r_a = _paired_diff_stats(
        wide_rt["global"],
        wide_rt["local"],
        "global - local (paired subject means of RT, s); same t as ttest_rel(global, local)",
        panel="A",
        outcome="RT acquired (HMM core)",
        delta_definition="delta = mean_s(global - local); negative => local slower on average",
    )
    if r_a:
        rows.append(r_a)

    # Panel B
    g_acc = d_acquired.groupby(["subj", "rule_level"], as_index=False).agg(val=("correctness", "mean")).dropna()
    wide_acc = g_acc.pivot(index="subj", columns="rule_level", values="val").dropna(subset=["global", "local"])
    r_b = _paired_diff_stats(
        wide_acc["global"],
        wide_acc["local"],
        "global - local (paired subject means of accuracy)",
        panel="B",
        outcome="Accuracy acquired (HMM core)",
        delta_definition="mean_s(global) - mean_s(local)",
    )
    if r_b:
        rows.append(r_b)

    # Panel C: replicate _plot_uncertainty_modulation
    pcols2 = post[["subj", "block_id", "trial_id", "trial_index_1based", "acquisition_trial_core"]].dropna()
    ds = beh.merge(pcols2, on=["subj", "block_id", "trial_id"], how="inner")
    ds = ds[ds["rule_level"].astype(str).isin(["global", "local"])].copy()
    ti = pd.to_numeric(ds["trial_index_1based"], errors="coerce")
    core = pd.to_numeric(ds["acquisition_trial_core"], errors="coerce")
    ds = ds[np.isfinite(ti) & np.isfinite(core) & (ti < core)].copy()
    hm = hmm[["subj", "block_id", "trial_id", "candidates_before"]].dropna()
    ds = ds.merge(hm, on=["subj", "block_id", "trial_id"], how="inner")
    ds["uncertainty_high"] = (pd.to_numeric(ds["candidates_before"], errors="coerce") > 1).astype(int)
    ds = ds.dropna(subset=["uncertainty_high", "rt"])

    subj_gap_rows = []
    for (subj, uh), g in ds.groupby(["subj", "uncertainty_high"]):
        gm = g.groupby("rule_level")["rt"].mean()
        if ("local" in gm.index) and ("global" in gm.index):
            subj_gap_rows.append(
                {
                    "subj": subj,
                    "uncertainty_high": int(uh),
                    "rt_gap_local_minus_global": float(gm["local"] - gm["global"]),
                }
            )
    gg = pd.DataFrame(subj_gap_rows)
    if not gg.empty:
        wide_c = gg.pivot(index="subj", columns="uncertainty_high", values="rt_gap_local_minus_global")
        if {0, 1}.issubset(set(wide_c.columns)):
            wide_c = wide_c.dropna(subset=[0, 1])
            r_c = _paired_diff_stats(
                wide_c[1],
                wide_c[0],
                "high_uncertainty - low_uncertainty (RT gap local-global, s)",
                panel="C",
                outcome="Pre-core RT gap (trial < core)",
                delta_definition="gap_high - gap_low per subject",
            )
            if r_c:
                rows.append(r_c)

    res = pd.DataFrame(rows)
    if res.empty:
        raise RuntimeError("No metrics computed.")

    res["pFDR_figure28"] = _bh_fdr(res["p"].to_numpy(dtype=float))
    res.to_csv(OUT_CSV, index=False)
    print(f"saved: {OUT_CSV}")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
