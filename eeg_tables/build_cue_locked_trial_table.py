"""
Build cue-locked trial-level EEG table for within-level vs cross-level shift (strand 3).
Output: eeg_trial_cue_locked.csv with subj, block_id, trial_id, shift_type, and cue-locked
frontal_theta_cue, posterior_alpha_cue, P3a_cue (optional).
Uses same alignment as feedback-locked pipeline; cue epochs selected by same ordinal as feedback.
"""
import os
import numpy as np
import pandas as pd
import mne

BASE = "base_dir"
COMP_TAG = os.environ.get("COMP_TAG", "1s_comp")
EEG_DIR = os.path.join(BASE, "prc", "ctap", "base_wcst", "correct_wcst_5", "tppd", "4_ADJUST_IC_CORR")
# 2s 用旧目录；1s_comp 用独立目录
if COMP_TAG in ["", "2s", "default"]:
    ALIGN_MATCH_CSV = os.path.join(BASE, "trials_trialwise", "eeg_behavior_trial_match_status.csv")
    OUT_DIR = os.path.join(BASE, "trials_trialwise", "eeg_tables")
else:
    ALIGN_MATCH_CSV = os.path.join(BASE, "trials_trialwise", COMP_TAG, "eeg_behavior_trial_match_status.csv")
    OUT_DIR = os.path.join(BASE, "trials_trialwise", COMP_TAG, "eeg_tables")
TRIALWISE_DIR = os.path.join(BASE, "trials_trialwise", "trialwise")
OUT_CSV = os.path.join(OUT_DIR, "eeg_trial_cue_locked.csv")

SKIP_SUBS = {"reknow011", "reknow020", "reknow023"}
FEEDBACK_CODE = "60"
CUE_CODE = "10"
EPOCH_WIN_CUE = (-0.2, 1.0)
BASELINE = (-0.2, 0.0)
BAND_THETA = (4, 7)
BAND_ALPHA = (8, 12)
WIN_THETA_CUE = (0.20, 0.60)
WIN_ALPHA_CUE = (0.40, 1.00)
WIN_P3A_CUE = (0.25, 0.45)
ROI_FRONT = ["Fz", "FC1", "FC2", "Cz", "F3", "F4"]
ROI_ALPHA_POST = ["Pz", "P3", "P4", "Oz", "O1", "O2", "PO9", "PO10"]


def band_power_window(sig: np.ndarray, t: np.ndarray, srate: float, band: tuple, win: tuple) -> float:
    m = (t >= win[0]) & (t <= win[1])
    if not np.any(m):
        return np.nan
    sig = np.asarray(sig[m], dtype=float).reshape(-1)
    if len(sig) < 16 or np.isnan(sig).any():
        return np.nan
    sig = sig - np.nanmean(sig)
    nfft = max(256, 2 ** int(np.ceil(np.log2(len(sig)))))
    x = np.fft.fft(sig, nfft)
    p2 = (np.abs(x) ** 2) / (srate * nfft)
    p1 = p2[: nfft // 2 + 1].copy()
    if len(p1) > 2:
        p1[1:-1] *= 2
    f = np.arange(0, nfft // 2 + 1) * (srate / nfft)
    mask = (f >= band[0]) & (f <= band[1])
    if not np.any(mask):
        return np.nan
    return float(np.trapezoid(p1[mask], f[mask]))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    align = pd.read_csv(ALIGN_MATCH_CSV)
    align["block"] = pd.to_numeric(align["block"], errors="coerce")
    align["trial"] = pd.to_numeric(align["trial"], errors="coerce")
    align["eeg_idx"] = pd.to_numeric(align["eeg_idx"], errors="coerce")

    # Block-level shift_type from trialwise (rule_level per block)
    all_trialwise = os.path.join(BASE, "trials_trialwise", "trialwise", "all_subjects_trialwise.csv")
    if not os.path.exists(all_trialwise):
        raise FileNotFoundError(f"Need {all_trialwise} for rule_level")
    beh = pd.read_csv(all_trialwise)
    beh["block_id"] = pd.to_numeric(beh["block_id"], errors="coerce")
    beh["rule_level"] = beh["rule_level"].astype(str).str.strip()
    block_rule = beh.groupby(["subj", "block_id"], as_index=False)["rule_level"].first()
    block_rule = block_rule.sort_values(["subj", "block_id"]).copy()
    block_rule["prev_rule_level"] = block_rule.groupby("subj")["rule_level"].shift(1)
    block_rule["shift_type"] = np.where(
        block_rule["prev_rule_level"].isna(),
        "missing",
        np.where(block_rule["prev_rule_level"] == block_rule["rule_level"], "within_level", "cross_level"),
    )
    block_rule = block_rule[["subj", "block_id", "shift_type"]].drop_duplicates()

    trial_rows = []
    set_files = sorted([f for f in os.listdir(EEG_DIR) if f.endswith("_reknow_wcst.set")])
    for fn in set_files:
        subj = fn.replace("_reknow_wcst.set", "")
        if subj in SKIP_SUBS:
            continue
        a = align[(align["subj"] == subj) & (align["match_status"] == "matched")].copy()
        if a.empty:
            continue
        a = a.dropna(subset=["block", "trial", "eeg_idx"])
        a["feedback_event_num"] = a["eeg_idx"].astype(int) + 1
        a = a.sort_values("feedback_event_num").drop_duplicates("feedback_event_num", keep="first").reset_index(drop=True)
        if a.empty:
            continue

        raw = mne.io.read_raw_eeglab(os.path.join(EEG_DIR, fn), preload=True, verbose="ERROR")
        ev_cue, id_cue = mne.events_from_annotations(raw, event_id={CUE_CODE: 1}, verbose="ERROR")
        if len(ev_cue) == 0:
            continue
        epochs_cue = mne.Epochs(
            raw,
            ev_cue,
            event_id=id_cue,
            tmin=EPOCH_WIN_CUE[0],
            tmax=EPOCH_WIN_CUE[1],
            baseline=BASELINE,
            preload=True,
            reject_by_annotation=False,
            verbose="ERROR",
        )
        keep_idx0 = a["feedback_event_num"].astype(int).values - 1
        n_common = min(len(epochs_cue), len(a))
        keep_idx0 = keep_idx0[(keep_idx0 >= 0) & (keep_idx0 < n_common)]
        if len(keep_idx0) == 0:
            continue
        a = a.iloc[: len(keep_idx0)].copy()
        epochs_cue = epochs_cue[keep_idx0]

        ch = list(epochs_cue.ch_names)
        idx_front = [ch.index(c) for c in ROI_FRONT if c in ch]
        idx_alpha = [ch.index(c) for c in ROI_ALPHA_POST if c in ch]
        X = epochs_cue.get_data(copy=True) * 1e6
        t = epochs_cue.times
        srate = float(epochs_cue.info["sfreq"])
        m_p3a = (t >= WIN_P3A_CUE[0]) & (t <= WIN_P3A_CUE[1])

        for i in range(len(epochs_cue)):
            sig_front = np.nanmean(X[i, idx_front, :], axis=0) if idx_front else np.full(X.shape[2], np.nan)
            sig_alpha = np.nanmean(X[i, idx_alpha, :], axis=0) if idx_alpha else np.full(X.shape[2], np.nan)
            theta_cue = (
                band_power_window(sig_front, t, srate, BAND_THETA, WIN_THETA_CUE)
                if np.any(np.isfinite(sig_front))
                else np.nan
            )
            alpha_cue = (
                band_power_window(sig_alpha, t, srate, BAND_ALPHA, WIN_ALPHA_CUE)
                if np.any(np.isfinite(sig_alpha))
                else np.nan
            )
            p3a_cue = float(np.nanmean(sig_front[m_p3a])) if np.any(np.isfinite(sig_front)) else np.nan
            b = int(a.iloc[i]["block"])
            tr = int(a.iloc[i]["trial"])
            trial_rows.append(
                {
                    "subj": subj,
                    "block_id": b,
                    "trial_id": tr,
                    "frontal_theta_cue": theta_cue,
                    "posterior_alpha_cue": alpha_cue,
                    "P3a_cue": p3a_cue,
                }
            )
        print(f"cue-locked done {subj}: {len(epochs_cue)} trials")

    if not trial_rows:
        raise RuntimeError("No cue-locked trial rows")
    out = pd.DataFrame(trial_rows)
    out = out.merge(block_rule, on=["subj", "block_id"], how="left")
    out.to_csv(OUT_CSV, index=False)
    print("saved ->", OUT_CSV)
    print("shift_type counts:", out["shift_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
