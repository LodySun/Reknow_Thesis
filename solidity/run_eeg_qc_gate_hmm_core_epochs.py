"""
New EEG QC gate (replaces legacy block_type × phase3 strata).

Strata: participant × shift_type × epoch_hmm_core
  - shift_type: within_level | cross_level (first_block excluded from primary QC table)
  - epoch_hmm_core (trial-wise, HMM-core aligned):
      pre_first_correct: trial_index_1based < first_correct_trial
      first_correct_to_pre_core: first_correct <= trial < acquisition_trial_core (requires finite core)
      core_onwards: trial >= acquisition_trial_core
      post_first_correct_no_core: first_correct reached but acquisition_trial_core is missing

Merge: eeg_trial_long ↔ hmm_unified_trial_posteriors on (subj, block_id, trial_id).

Outputs:
  trials_trialwise/1s_comp/eeg_paper_results/qc_gate_trial_counts_hmm_core.csv
  trials_trialwise/1s_comp/eeg_paper_results/qc_gate_subject_summary_hmm_core.csv
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

BASE = "base_dir"
EEG_TRIAL = os.path.join(BASE, "trials_trialwise", "1s_comp", "eeg_tables", "eeg_trial_long.csv")
HMM_TRIAL = os.path.join(
    BASE, "trials_trialwise", "1s_comp", "eeg_paper_results", "solidity", "hmm_unified_trial_posteriors.csv"
)
OUT_DIR = os.path.join(BASE, "trials_trialwise", "1s_comp", "eeg_paper_results")
OUT_COUNTS = os.path.join(OUT_DIR, "qc_gate_trial_counts_hmm_core.csv")
OUT_SUBJ = os.path.join(OUT_DIR, "qc_gate_subject_summary_hmm_core.csv")

TRIAL_MIN = 20


def _epoch_hmm_core(row: pd.Series) -> str:
    ti = row.get("trial_index_1based")
    fc = row.get("first_correct_trial")
    acq = row.get("acquisition_trial_core")
    if not np.isfinite(ti) or not np.isfinite(fc):
        return "unknown"
    if ti < fc:
        return "pre_first_correct"
    if not np.isfinite(acq):
        return "post_first_correct_no_core"
    if ti < acq:
        return "first_correct_to_pre_core"
    return "core_onwards"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    eeg = pd.read_csv(EEG_TRIAL)
    hmm = pd.read_csv(HMM_TRIAL)

    for c in ["block_id", "trial_id"]:
        eeg[c] = pd.to_numeric(eeg[c], errors="coerce")
        hmm[c] = pd.to_numeric(hmm[c], errors="coerce")

    use = hmm[
        [
            "subj",
            "block_id",
            "trial_id",
            "trial_index_1based",
            "first_correct_trial",
            "acquisition_trial_core",
            "shift_type",
        ]
    ].copy()
    use["subj"] = use["subj"].astype(str)
    eeg["subj"] = eeg["subj"].astype(str)

    m = eeg.merge(use, on=["subj", "block_id", "trial_id"], how="inner", suffixes=("", "_hmm"))
    m["epoch_hmm_core"] = m.apply(_epoch_hmm_core, axis=1)

    primary = m[m["shift_type"].isin(["within_level", "cross_level"])].copy()
    primary = primary[primary["epoch_hmm_core"] != "unknown"].copy()

    qc = (
        primary.groupby(["subj", "shift_type", "epoch_hmm_core"], as_index=False)
        .size()
        .rename(columns={"size": "n_trials"})
    )
    qc["pass_min_trials"] = qc["n_trials"] >= TRIAL_MIN
    qc.to_csv(OUT_COUNTS, index=False)

    subj = (
        qc.groupby("subj", as_index=False)
        .agg(
            n_cells=("n_trials", "count"),
            n_cells_failed=("pass_min_trials", lambda s: int((~s).sum())),
            min_cell_n=("n_trials", "min"),
        )
    )
    subj.to_csv(OUT_SUBJ, index=False)

    n_strata = len(qc)
    n_subj = qc["subj"].nunique()
    print(f"saved: {OUT_COUNTS} (rows={n_strata})")
    print(f"saved: {OUT_SUBJ} (subjects={n_subj})")
    print("epoch_hmm_core levels:", sorted(primary["epoch_hmm_core"].unique().tolist()))
    print("total trials in primary strata:", int(primary.shape[0]))


if __name__ == "__main__":
    main()
