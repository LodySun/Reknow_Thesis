"""
EEG–behavior alignment via SQLite trial timestamps.

Prefer `align_behavior_from_eeg_events.m` when preprocessed .set files already
encode trial/block on feedback events (ruleblockid, trialidx): that path avoids
depending on the SQLite export for indexing and does not require agreeing with
CTAP DB trial order for eeg_idx.

This script remains useful if you need wall-clock strings (eeg_ts_str) or must
reproduce an older pipeline.
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import sqlite3

# Paths
BASE = "base_dir"
TRIALS_DIR = os.path.join(BASE, "trials")
SQLITE_PATH = os.path.join(
    BASE,
    "prc/final/sql_dbs/rerefAvg/src-4_IC_CORR/export"
    "/rerefAvg_wcst_db--bandpowers--2_Avgref_shorttrial_bandpows.sqlite",
)
OUT_DIR = os.path.join(BASE, "trials_trialwise")  # or a dedicated align dir
SKIP_SUBJECTS = {"reknow011", "reknow020", "reknow023"}

# Alignment: max deviation from baseline offset (seconds) to consider trial "normal"
MAX_OFFSET_DEVIATION_SEC = 2.0

# Time-window scan: half-width in seconds for matching beh trial to EEG timestamp (in relative time)
MATCH_WINDOW_SEC = 1.0

def parse_yyyymmdd_thhmmss(ts_str: str) -> float:
    """Parse 'YYYYMMDDThhmmss' to Unix timestamp (seconds)."""
    try:
        dt = datetime.strptime(ts_str.strip(), "%Y%m%dT%H%M%S")
        return dt.timestamp()
    except (ValueError, TypeError):
        return float("nan")


def get_eeg_timestamps(conn: sqlite3.Connection, subjectnr: int):
    """Get ordered distinct timestamps for one subject. Returns (seconds_series, timestamp_strings)."""
    df = pd.read_sql_query(
        "SELECT DISTINCT timestamp FROM results WHERE subjectnr = ? ORDER BY timestamp",
        conn,
        params=(subjectnr,),
    )
    if df.empty:
        return pd.Series(dtype=float), []
    ts_str = df["timestamp"].astype(str).tolist()
    t_sec = pd.Series([parse_yyyymmdd_thhmmss(s) for s in ts_str])
    return t_sec, ts_str


def get_subjectnr_to_subjectstr(conn: sqlite3.Connection) -> dict:
    """Map subjectnr -> subjectstr (e.g. 1 -> reknow001)."""
    df = pd.read_sql_query("SELECT subjectnr, subjectstr FROM subject", conn)
    return dict(zip(df["subjectnr"], df["subjectstr"].astype(str)))


def match_trials_by_time_window(
    beh_rel: np.ndarray,
    eeg_rel: np.ndarray,
    window_sec: float,
) -> tuple:
    """
    Match behavior trials to EEG trials by scanning in relative time.
    beh_rel[i] = tgt_ts[i] - tgt_ts[0], eeg_rel[j] = eeg_sec[j] - eeg_sec[0].
    Returns:
      beh_to_eeg: for each beh index, matched eeg index or -1
      eeg_to_beh: for each eeg index, matched beh index or -1
      beh_to_diff: for each beh index, time diff (eeg_rel - beh_rel) at match, or nan
    """
    n_beh = len(beh_rel)
    n_eeg = len(eeg_rel)
    beh_to_eeg = [-1] * n_beh
    eeg_to_beh = [-1] * n_eeg
    beh_to_diff = [float("nan")] * n_beh

    # Greedy: for each beh in order, match to closest eeg within window that is still unmatched
    for i in range(n_beh):
        t = float(beh_rel[i])
        best_j = -1
        best_diff = window_sec + 1.0
        for j in range(n_eeg):
            if eeg_to_beh[j] >= 0:
                continue
            d = abs(float(eeg_rel[j]) - t)
            if d <= window_sec and d < best_diff:
                best_diff = d
                best_j = j
        if best_j >= 0:
            beh_to_eeg[i] = best_j
            eeg_to_beh[best_j] = i
            beh_to_diff[i] = float(eeg_rel[best_j]) - t

    return beh_to_eeg, eeg_to_beh, beh_to_diff


def run_alignment():
    os.makedirs(OUT_DIR, exist_ok=True)

    conn = sqlite3.connect(SQLITE_PATH)
    nr_to_str = get_subjectnr_to_subjectstr(conn)

    # subjectstr -> subjectnr
    str_to_nr = {v: k for k, v in nr_to_str.items()}

    rows_summary = []
    all_abnormal = []
    all_missing_info = []
    all_trial_matches: list[pd.DataFrame] = []
    all_orphan_eeg: list[dict] = []

    for subj_id, subj_nr in sorted(str_to_nr.items(), key=lambda x: x[1]):
        if subj_id in SKIP_SUBJECTS:
            all_missing_info.append({"subj": subj_id, "reason": "skipped (info missing)"})
            continue

        beh_path = os.path.join(TRIALS_DIR, f"{subj_id}_trials.csv")
        if not os.path.exists(beh_path):
            all_missing_info.append({"subj": subj_id, "reason": "behavior file not found"})
            continue

        beh = pd.read_csv(beh_path)
        beh["block_num"] = pd.to_numeric(beh["block"], errors="coerce")
        beh_from13 = beh[beh["block_num"] >= 13].sort_values(["block_num", "trial"]).reset_index(drop=True)
        tgt_ts = beh_from13["tgt_ts"].values

        eeg_ts, eeg_ts_str_list = get_eeg_timestamps(conn, subj_nr)
        if isinstance(eeg_ts, pd.Series) and eeg_ts.isna().any():
            all_abnormal.append({
                "subj": subj_id,
                "trial_idx": int(eeg_ts.isna().idxmax()),
                "issue": "EEG timestamp parse error",
            })
        eeg_sec = np.asarray(eeg_ts.values if isinstance(eeg_ts, pd.Series) else eeg_ts)
        if not eeg_ts_str_list:
            eeg_ts_str_list = [""] * max(len(eeg_sec), 0)

        n_beh = len(tgt_ts)
        n_eeg = len(eeg_sec)

        # Time-window scan: relative time from first trial (both streams)
        beh_rel = np.asarray(tgt_ts, dtype=float) - float(tgt_ts[0])
        eeg_rel = np.asarray(eeg_sec, dtype=float) - float(eeg_sec[0]) if n_eeg > 0 else np.array([])
        beh_to_eeg, eeg_to_beh, beh_to_diff = match_trials_by_time_window(
            beh_rel, eeg_rel, MATCH_WINDOW_SEC
        )

        # Per-trial match status for behavior trials
        match_status = [
            "matched" if beh_to_eeg[i] >= 0 else "missing_in_eeg"
            for i in range(n_beh)
        ]
        trial_rows = []
        for i in range(n_beh):
            j = beh_to_eeg[i]
            trial_rows.append({
                "subj": subj_id,
                "block": beh_from13["block"].iloc[i],
                "trial": beh_from13["trial"].iloc[i],
                "trial_idx_0based": i,
                "tgt_ts": round(float(tgt_ts[i]), 3),
                "match_status": match_status[i],
                "eeg_idx": j if j >= 0 else None,
                "eeg_ts_str": eeg_ts_str_list[j] if (j >= 0 and j < len(eeg_ts_str_list)) else "",
                "time_diff_sec": round(beh_to_diff[i], 4) if j >= 0 and not np.isnan(beh_to_diff[i]) else None,
            })
        all_trial_matches.append(pd.DataFrame(trial_rows))

        # Orphan EEG: no behavior trial within window
        for j in range(n_eeg):
            if eeg_to_beh[j] >= 0:
                continue
            all_orphan_eeg.append({
                "subj": subj_id,
                "eeg_idx": j,
                "eeg_ts_str": eeg_ts_str_list[j] if j < len(eeg_ts_str_list) else "",
                "eeg_rel_sec": round(float(eeg_rel[j]), 3) if j < len(eeg_rel) else None,
            })

        n_missing_by_scan = sum(1 for s in match_status if s == "missing_in_eeg")
        n_orphan_eeg = sum(1 for j in range(n_eeg) if eeg_to_beh[j] < 0)

        # Length mismatch (simple count diff, kept for comparison)
        n_missing = max(0, n_beh - n_eeg)  # behavior trials without EEG
        n_extra = max(0, n_eeg - n_beh)    # EEG trials without behavior

        # Time alignment: use block13 first trial as baseline
        # EEG: convert to seconds relative to first EEG timestamp (so we use same scale as behavior if both are "seconds from start")
        # Actually behavior tgt_ts is in experiment clock; EEG is wall clock. So we compare *relative* progression:
        # offset_i = (eeg_sec_i - eeg_sec_0) - (tgt_ts_i - tgt_ts_0) = (eeg_sec_i - tgt_ts_i) - (eeg_sec_0 - tgt_ts_0).
        # So baseline offset = eeg_sec_0 - tgt_ts_0; then for each i, offset_i = eeg_sec_i - tgt_ts_i; deviation_i = offset_i - baseline.
        n_align = min(n_beh, n_eeg)
        abnormal_trials = []
        baseline_offset = None
        max_abs_dev_sec = None

        if n_align > 0:
            # Use first trial as baseline (block13 first trial). Deviation = (eeg_sec[i]-tgt_ts[i]) - (eeg_sec[0]-tgt_ts[0]).
            baseline_offset = float(eeg_sec[0]) - float(tgt_ts[0])
            devs = []
            for i in range(n_align):
                offset_i = float(eeg_sec[i]) - float(tgt_ts[i])
                dev = offset_i - baseline_offset
                devs.append(abs(dev))
                if abs(dev) > MAX_OFFSET_DEVIATION_SEC:
                    abnormal_trials.append({
                        "subj": subj_id,
                        "trial_idx_0based": i,
                        "block": beh_from13["block"].iloc[i],
                        "trial": beh_from13["trial"].iloc[i],
                        "offset_deviation_sec": round(dev, 4),
                        "tgt_ts": round(float(tgt_ts[i]), 3),
                        "eeg_ts_str": eeg_ts_str_list[i] if i < len(eeg_ts_str_list) else "",
                    })
            for a in abnormal_trials:
                all_abnormal.append(a)
            max_abs_dev_sec = round(max(devs), 4) if devs else None

        rows_summary.append({
            "subj": subj_id,
            "subjectnr": subj_nr,
            "n_beh_from_block13": n_beh,
            "n_eeg_distinct_ts": n_eeg,
            "n_missing_eeg": n_missing,
            "n_extra_eeg": n_extra,
            "n_missing_by_scan": n_missing_by_scan,
            "n_orphan_eeg": n_orphan_eeg,
            "n_abnormal_offset": len(abnormal_trials),
            "max_abs_deviation_sec": max_abs_dev_sec,
            "baseline_offset_sec": round(baseline_offset, 4) if baseline_offset is not None else None,
        })

    conn.close()

    summary_df = pd.DataFrame(rows_summary)
    out_summary = os.path.join(OUT_DIR, "eeg_behavior_alignment_summary.csv")
    summary_df.to_csv(out_summary, index=False)
    print(f"Summary -> {out_summary}")
    print(summary_df.to_string(index=False))

    # Trial-level match status (time-window scan)
    if all_trial_matches:
        trial_match_df = pd.concat(all_trial_matches, ignore_index=True)
        out_trials = os.path.join(OUT_DIR, "eeg_behavior_trial_match_status.csv")
        trial_match_df.to_csv(out_trials, index=False)
        print(f"\nTrial match status (window ±{MATCH_WINDOW_SEC}s) -> {out_trials}")
        missing_trials = trial_match_df[trial_match_df["match_status"] == "missing_in_eeg"]
        print(f"  Total trials: {len(trial_match_df)}, missing_in_eeg: {len(missing_trials)}, matched: {len(trial_match_df) - len(missing_trials)}")

    if all_orphan_eeg:
        orphan_df = pd.DataFrame(all_orphan_eeg)
        out_orphan = os.path.join(OUT_DIR, "eeg_behavior_orphan_eeg_trials.csv")
        orphan_df.to_csv(out_orphan, index=False)
        print(f"Orphan EEG trials (no beh in window) -> {out_orphan} ({len(orphan_df)} rows)")

    if all_abnormal:
        abnormal_df = pd.DataFrame(all_abnormal)
        out_abn = os.path.join(OUT_DIR, "eeg_behavior_abnormal_trials.csv")
        abnormal_df.to_csv(out_abn, index=False)
        print(f"\nAbnormal trials (offset deviation > {MAX_OFFSET_DEVIATION_SEC}s) -> {out_abn}")
        print(abnormal_df.head(20).to_string(index=False))
        if len(abnormal_df) > 20:
            print(f"... and {len(abnormal_df) - 20} more")

    # Always report skipped subjects (011, 020, 023) even if not in DB
    for sid in SKIP_SUBJECTS:
        if not any(m.get("subj") == sid for m in all_missing_info):
            all_missing_info.append({"subj": sid, "reason": "skipped (info missing)"})
    if all_missing_info:
        missing_df = pd.DataFrame(all_missing_info)
        out_miss = os.path.join(OUT_DIR, "eeg_behavior_skipped_subjects.csv")
        missing_df.to_csv(out_miss, index=False)
        print(f"\nSkipped / missing -> {out_miss}")
        print(missing_df.to_string(index=False))

    # Per-subject length mismatch as "missing trial" report
    mismatch = summary_df[
        (summary_df["n_missing_eeg"] > 0) | (summary_df["n_extra_eeg"] > 0)
    ]
    if not mismatch.empty:
        print("\n--- Subjects with trial count mismatch (missing/extra) ---")
        print(mismatch.to_string(index=False))

    return summary_df, all_abnormal, all_missing_info


if __name__ == "__main__":
    run_alignment()
