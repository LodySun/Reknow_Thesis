import os
import numpy as np
import pandas as pd
import mne


BASE = "/Users/lodysun/Desktop/Thesis"
EEG_DIR = "/Users/lodysun/Desktop/Thesis/prc/ctap/base_wcst/correct_wcst_5/tppd/4_ADJUST_IC_CORR"
ALIGN_MATCH_CSV = os.path.join(BASE, "trials_trialwise", "eeg_behavior_trial_match_status.csv")
TRIALWISE_DIR = os.path.join(BASE, "trials_trialwise", "trialwise")
BLOCK_LABEL_CSV = os.path.join(BASE, "trials_trialwise", "hmm_mixture", "hmm_mixture_soft_block_labels.csv")
OUT_DIR = os.path.join(BASE, "trials_trialwise", "eeg_tables")

SKIP_SUBS = {"reknow011", "reknow020", "reknow023"}
FEEDBACK_CODE = "60"
CUE_CODE = "10"
TARGET_CODE = "30"
EPOCH_WIN_SEC = (-0.2, 1.0)
BASELINE = (-0.2, 0.0)

WIN_P3A = (0.30, 0.45)  # frontal/fronto-central (Frontal P3: 300-450 ms)
WIN_P3B = (0.30, 0.50)  # centro-parietal/parietal
BAND_THETA = (4, 7)
BAND_ALPHA = (8, 12)
BAND_BETA = (16, 32)
WIN_THETA_CTRL_MAIN = (0.20, 0.60)     # primary control/update window
WIN_THETA_CTRL_EARLY = (0.20, 0.40)    # sensitivity 1
WIN_THETA_CTRL_LATE = (0.40, 0.70)     # sensitivity 2
WIN_THETA_PREP_EARLY = (0.20, 0.40)    # post-cue sensitivity
WIN_THETA_PREP_LATE = (0.40, 0.70)     # post-cue sensitivity
WIN_THETA_EXEC = (0.00, 0.45)          # post-target execution
WIN_ALPHA_POST_MAIN = (0.30, 0.80)     # feedback-locked posterior alpha
WIN_ALPHA_CUE_PREP = (0.40, 1.00)      # cue-locked preparatory alpha
WIN_FP_BETA_MAIN = (0.30, 0.90)        # fronto-parietal maintenance/implementation

ROI_FRONT = ["Fz", "FC1", "FC2", "Cz", "F3", "F4"]
ROI_PARIETAL = ["Pz", "P3", "P4", "PO9", "PO10", "Oz"]
ROI_ALPHA_POST = ["Pz", "P3", "P4", "Oz", "O1", "O2", "PO9", "PO10"]
ROI_L = ["F3", "F7"]
ROI_R = ["F4", "F8"]


def band_power_1d(sig: np.ndarray, srate: float, band: tuple[float, float]) -> float:
    sig = np.asarray(sig, dtype=float).reshape(-1)
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
    m = (f >= band[0]) & (f <= band[1])
    if not np.any(m):
        return np.nan
    fm = f[m]
    pm = p1[m]
    if pm.size == 1:
        # Single-bin case: trapezoid would return 0, so use PSD * bin width.
        df = float(f[1] - f[0]) if len(f) > 1 else 1.0
        return float(pm[0] * df)
    return float(np.trapezoid(pm, fm))


def band_power_window(sig: np.ndarray, t: np.ndarray, srate: float, band: tuple[float, float], win: tuple[float, float]) -> float:
    m = (t >= win[0]) & (t <= win[1])
    if not np.any(m):
        return np.nan
    return band_power_1d(sig[m], srate, band)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    align = pd.read_csv(ALIGN_MATCH_CSV)
    block_types = pd.read_csv(BLOCK_LABEL_CSV)
    block_types["block_id"] = pd.to_numeric(block_types["block_id"], errors="coerce")

    trial_rows = []
    roi_report = []

    set_files = sorted([f for f in os.listdir(EEG_DIR) if f.endswith("_reknow_wcst.set")])
    for fn in set_files:
        subj = fn.replace("_reknow_wcst.set", "")
        if subj in SKIP_SUBS:
            continue

        trialwise_csv = os.path.join(TRIALWISE_DIR, f"{subj}_trialwise.csv")
        if not os.path.exists(trialwise_csv):
            continue

        beh = pd.read_csv(trialwise_csv)
        beh["block"] = pd.to_numeric(beh["block_id"], errors="coerce")
        beh["trial"] = pd.to_numeric(beh["trial_id"], errors="coerce")
        beh_phase = {(int(b), int(t)): p for b, t, p in zip(beh["block"], beh["trial"], beh["phase"])}

        bt = block_types[block_types["subj"] == subj][["block_id", "strategy_hard"]].drop_duplicates()
        bt_map = {int(b): s for b, s in zip(bt["block_id"], bt["strategy_hard"]) if np.isfinite(b)}

        # Main change: only keep matched rows; do NOT remove abnormal rows.
        a = align[(align["subj"] == subj) & (align["match_status"] == "matched")].copy()
        if a.empty:
            continue
        a["block"] = pd.to_numeric(a["block"], errors="coerce")
        a["trial"] = pd.to_numeric(a["trial"], errors="coerce")
        a["eeg_idx"] = pd.to_numeric(a["eeg_idx"], errors="coerce")
        a = a.dropna(subset=["block", "trial", "eeg_idx"])
        a["feedback_event_num"] = a["eeg_idx"].astype(int) + 1
        a = a.sort_values("feedback_event_num").drop_duplicates("feedback_event_num", keep="first").reset_index(drop=True)
        if a.empty:
            continue

        raw = mne.io.read_raw_eeglab(os.path.join(EEG_DIR, fn), preload=True, verbose="ERROR")
        fb_onsets = [x["onset"] for x in raw.annotations if str(x["description"]).strip() == FEEDBACK_CODE]
        if len(fb_onsets) == 0:
            continue
        nfb = len(fb_onsets)
        a = a[(a["feedback_event_num"] >= 1) & (a["feedback_event_num"] <= nfb)].copy()
        if a.empty:
            continue

        # Epoch all key events
        ev_fb, id_fb = mne.events_from_annotations(raw, event_id={FEEDBACK_CODE: 1}, verbose="ERROR")
        ev_cue, id_cue = mne.events_from_annotations(raw, event_id={CUE_CODE: 1}, verbose="ERROR")
        ev_tgt, id_tgt = mne.events_from_annotations(raw, event_id={TARGET_CODE: 1}, verbose="ERROR")
        if len(ev_fb) == 0 or len(ev_cue) == 0 or len(ev_tgt) == 0:
            continue
        epochs = mne.Epochs(
            raw,
            ev_fb,
            event_id=id_fb,
            tmin=EPOCH_WIN_SEC[0],
            tmax=EPOCH_WIN_SEC[1],
            baseline=BASELINE,
            preload=True,
            reject_by_annotation=False,
            verbose="ERROR",
        )
        epochs_cue = mne.Epochs(
            raw,
            ev_cue,
            event_id=id_cue,
            tmin=-0.2,
            tmax=1.0,
            baseline=BASELINE,
            preload=True,
            reject_by_annotation=False,
            verbose="ERROR",
        )
        epochs_tgt = mne.Epochs(
            raw,
            ev_tgt,
            event_id=id_tgt,
            tmin=-0.2,
            tmax=0.8,
            baseline=BASELINE,
            preload=True,
            reject_by_annotation=False,
            verbose="ERROR",
        )

        keep_idx0 = a["feedback_event_num"].astype(int).values - 1
        n_common = min(len(epochs), len(epochs_cue), len(epochs_tgt))
        keep_idx0 = keep_idx0[(keep_idx0 >= 0) & (keep_idx0 < n_common)]
        if len(keep_idx0) == 0:
            continue
        a = a.iloc[: len(keep_idx0)].copy()
        epochs = epochs[keep_idx0]
        epochs_cue = epochs_cue[keep_idx0]
        epochs_tgt = epochs_tgt[keep_idx0]

        ch = list(epochs.ch_names)
        idx_front = [ch.index(c) for c in ROI_FRONT if c in ch]
        idx_par = [ch.index(c) for c in ROI_PARIETAL if c in ch]
        idx_alpha = [ch.index(c) for c in ROI_ALPHA_POST if c in ch]
        idx_l = [ch.index(c) for c in ROI_L if c in ch]
        idx_r = [ch.index(c) for c in ROI_R if c in ch]

        roi_report.append(
            {
                "subj": subj,
                "p3a_roi_expected": ",".join(ROI_FRONT),
                "p3a_roi_used": ",".join([c for c in ROI_FRONT if c in ch]),
                "p3b_roi_expected": ",".join(ROI_PARIETAL),
                "p3b_roi_used": ",".join([c for c in ROI_PARIETAL if c in ch]),
            }
        )

        X = epochs.get_data(copy=True) * 1e6  # feedback-locked uV
        X_cue = epochs_cue.get_data(copy=True) * 1e6
        X_tgt = epochs_tgt.get_data(copy=True) * 1e6
        t = epochs.times
        t_cue = epochs_cue.times
        t_tgt = epochs_tgt.times
        srate = float(epochs.info["sfreq"])
        m_p3a = (t >= WIN_P3A[0]) & (t <= WIN_P3A[1])
        m_p3b = (t >= WIN_P3B[0]) & (t <= WIN_P3B[1])

        for i in range(len(epochs)):
            sig_front = np.nanmean(X[i, idx_front, :], axis=0) if idx_front else np.full(X.shape[2], np.nan)
            sig_par = np.nanmean(X[i, idx_par, :], axis=0) if idx_par else np.full(X.shape[2], np.nan)
            sig_alpha = np.nanmean(X[i, idx_alpha, :], axis=0) if idx_alpha else np.full(X.shape[2], np.nan)
            sig_front_cue = np.nanmean(X_cue[i, idx_front, :], axis=0) if idx_front else np.full(X_cue.shape[2], np.nan)
            sig_par_cue = np.nanmean(X_cue[i, idx_par, :], axis=0) if idx_par else np.full(X_cue.shape[2], np.nan)
            sig_front_tgt = np.nanmean(X_tgt[i, idx_front, :], axis=0) if idx_front else np.full(X_tgt.shape[2], np.nan)
            sig_l = np.nanmean(X[i, idx_l, :], axis=0) if idx_l else np.full(X.shape[2], np.nan)
            sig_r = np.nanmean(X[i, idx_r, :], axis=0) if idx_r else np.full(X.shape[2], np.nan)

            p3a = float(np.nanmean(sig_front[m_p3a])) if np.any(np.isfinite(sig_front)) else np.nan
            p3b = float(np.nanmean(sig_par[m_p3b])) if np.any(np.isfinite(sig_par)) else np.nan
            theta = (
                band_power_window(sig_front, t, srate, BAND_THETA, WIN_THETA_CTRL_MAIN)
                if np.any(np.isfinite(sig_front))
                else np.nan
            )
            theta_ctrl_early = (
                band_power_window(sig_front, t, srate, BAND_THETA, WIN_THETA_CTRL_EARLY)
                if np.any(np.isfinite(sig_front))
                else np.nan
            )
            theta_ctrl_late = (
                band_power_window(sig_front, t, srate, BAND_THETA, WIN_THETA_CTRL_LATE)
                if np.any(np.isfinite(sig_front))
                else np.nan
            )
            par_theta = band_power_1d(sig_par, srate, BAND_THETA) if np.any(np.isfinite(sig_par)) else np.nan
            alpha = (
                band_power_window(sig_alpha, t, srate, BAND_ALPHA, WIN_ALPHA_POST_MAIN)
                if np.any(np.isfinite(sig_alpha))
                else np.nan
            )
            par_alpha = band_power_1d(sig_par, srate, BAND_ALPHA) if np.any(np.isfinite(sig_par)) else np.nan
            theta_prep_early = band_power_window(sig_front_cue, t_cue, srate, BAND_THETA, WIN_THETA_PREP_EARLY) if np.any(np.isfinite(sig_front_cue)) else np.nan
            theta_prep_late = band_power_window(sig_front_cue, t_cue, srate, BAND_THETA, WIN_THETA_PREP_LATE) if np.any(np.isfinite(sig_front_cue)) else np.nan
            theta_exec = band_power_window(sig_front_tgt, t_tgt, srate, BAND_THETA, WIN_THETA_EXEC) if np.any(np.isfinite(sig_front_tgt)) else np.nan
            par_theta_prep_late = band_power_window(sig_par_cue, t_cue, srate, BAND_THETA, WIN_THETA_PREP_LATE) if np.any(np.isfinite(sig_par_cue)) else np.nan
            alpha_cue_prep = band_power_window(sig_par_cue, t_cue, srate, BAND_ALPHA, WIN_ALPHA_CUE_PREP) if np.any(np.isfinite(sig_par_cue)) else np.nan
            fp_beta_front = band_power_window(sig_front, t, srate, BAND_BETA, WIN_FP_BETA_MAIN) if np.any(np.isfinite(sig_front)) else np.nan
            fp_beta_par = band_power_window(sig_par, t, srate, BAND_BETA, WIN_FP_BETA_MAIN) if np.any(np.isfinite(sig_par)) else np.nan
            # CTI alpha slope across three bins in cue-locked interval
            alpha_bins = []
            centers = np.array([0.2, 0.4, 0.65], dtype=float)
            alpha_bins.append(band_power_window(sig_par_cue, t_cue, srate, BAND_ALPHA, (0.10, 0.30)))
            alpha_bins.append(band_power_window(sig_par_cue, t_cue, srate, BAND_ALPHA, (0.30, 0.50)))
            alpha_bins.append(band_power_window(sig_par_cue, t_cue, srate, BAND_ALPHA, (0.50, 0.80)))
            alpha_bins = np.array(alpha_bins, dtype=float)
            if np.sum(np.isfinite(alpha_bins)) >= 2:
                cc = centers[np.isfinite(alpha_bins)]
                yy = alpha_bins[np.isfinite(alpha_bins)]
                par_alpha_cti_slope = float(np.polyfit(cc, yy, 1)[0])
            else:
                par_alpha_cti_slope = np.nan
            p_l = band_power_1d(sig_l, srate, BAND_ALPHA) if np.any(np.isfinite(sig_l)) else np.nan
            p_r = band_power_1d(sig_r, srate, BAND_ALPHA) if np.any(np.isfinite(sig_r)) else np.nan
            asym = float(np.log(p_l) - np.log(p_r)) if np.isfinite(p_l) and np.isfinite(p_r) and p_l > 0 and p_r > 0 else np.nan

            b = int(a.iloc[i]["block"])
            tr = int(a.iloc[i]["trial"])
            phase = beh_phase.get((b, tr), "unknown")
            bt = bt_map.get(b, "unknown")

            trial_rows.append(
                {
                    "subj": subj,
                    "block_id": b,
                    "trial_id": tr,
                    "phase": phase,
                    "block_type": bt,
                    "feedback_locked_P3a": p3a,
                    "feedback_locked_P3b": p3b,
                    "theta_power": theta,  # frontal theta (P3a ROI)
                    "theta_ctrl_200_400_feedback": theta_ctrl_early,
                    "theta_ctrl_400_700_feedback": theta_ctrl_late,
                    "parietal_theta_power": par_theta,
                    "alpha_power": alpha,
                    "posterior_alpha_400_1000_cue": alpha_cue_prep,
                    "front_beta_300_900_feedback": fp_beta_front,
                    "parietal_beta_300_900_feedback": fp_beta_par,
                    "parietal_alpha_power": par_alpha,
                    "theta_prep_early_100_300_cue": theta_prep_early,
                    "theta_prep_late_500_800_cue": theta_prep_late,
                    "theta_exec_0_450_target": theta_exec,
                    "parietal_theta_prep_late_500_800_cue": par_theta_prep_late,
                    "parietal_alpha_cti_slope_100_800_cue": par_alpha_cti_slope,
                    "frontal_asym_alpha": asym,
                    "qc_kept_after_alignment": 1,
                    "feedback_event_num": int(a.iloc[i]["feedback_event_num"]),
                    "eeg_idx_0based": int(a.iloc[i]["eeg_idx"]),
                }
            )

        print(f"done {subj}: {len(epochs)} trials")

    if not trial_rows:
        raise RuntimeError("No trial rows generated")

    trial = pd.DataFrame(trial_rows)
    trial.to_csv(os.path.join(OUT_DIR, "eeg_trial_long.csv"), index=False)
    pd.DataFrame(roi_report).drop_duplicates().to_csv(
        os.path.join(OUT_DIR, "roi_validation_p3a_p3b.csv"), index=False
    )

    # block long
    block = (
        trial.groupby(["subj", "block_id", "phase", "block_type"], as_index=False)
        .agg(
            n_trials=("trial_id", "count"),
            feedback_locked_P3a=("feedback_locked_P3a", "mean"),
            feedback_locked_P3b=("feedback_locked_P3b", "mean"),
            theta_power=("theta_power", "mean"),
            theta_ctrl_200_400_feedback=("theta_ctrl_200_400_feedback", "mean"),
            theta_ctrl_400_700_feedback=("theta_ctrl_400_700_feedback", "mean"),
            parietal_theta_power=("parietal_theta_power", "mean"),
            alpha_power=("alpha_power", "mean"),
            posterior_alpha_400_1000_cue=("posterior_alpha_400_1000_cue", "mean"),
            front_beta_300_900_feedback=("front_beta_300_900_feedback", "mean"),
            parietal_beta_300_900_feedback=("parietal_beta_300_900_feedback", "mean"),
            parietal_alpha_power=("parietal_alpha_power", "mean"),
            theta_prep_early_100_300_cue=("theta_prep_early_100_300_cue", "mean"),
            theta_prep_late_500_800_cue=("theta_prep_late_500_800_cue", "mean"),
            theta_exec_0_450_target=("theta_exec_0_450_target", "mean"),
            parietal_theta_prep_late_500_800_cue=("parietal_theta_prep_late_500_800_cue", "mean"),
            parietal_alpha_cti_slope_100_800_cue=("parietal_alpha_cti_slope_100_800_cue", "mean"),
            frontal_asym_alpha=("frontal_asym_alpha", "mean"),
        )
    )
    block.to_csv(os.path.join(OUT_DIR, "eeg_block_long.csv"), index=False)

    # subject traits
    rows = []
    for subj, g in block.groupby("subj"):
        row = {"subj": subj}
        for ph in ["search", "acquired"]:
            gg = g[g["phase"] == ph]
            tag = "search" if ph == "search" else "transition"
            for m in [
                "feedback_locked_P3a",
                "feedback_locked_P3b",
                "theta_power",
                "theta_ctrl_200_400_feedback",
                "theta_ctrl_400_700_feedback",
                "alpha_power",
                "posterior_alpha_400_1000_cue",
                "front_beta_300_900_feedback",
                "parietal_beta_300_900_feedback",
                "frontal_asym_alpha",
            ]:
                gv = gg.loc[gg["block_type"] == "gradual_like", m].mean()
                ov = gg.loc[gg["block_type"] == "one_shot_like", m].mean()
                row[f"delta_{m}_{tag}_gradual_minus_one_shot"] = gv - ov
        row["n_blocks_gradual"] = int((g["block_type"] == "gradual_like").sum())
        row["n_blocks_one_shot"] = int((g["block_type"] == "one_shot_like").sum())
        row["n_trials_total"] = int(len(trial[trial["subj"] == subj]))
        rows.append(row)
    subj_traits = pd.DataFrame(rows)
    subj_traits.to_csv(os.path.join(OUT_DIR, "eeg_subject_traits.csv"), index=False)
    print("saved eeg tables ->", OUT_DIR)


if __name__ == "__main__":
    main()

