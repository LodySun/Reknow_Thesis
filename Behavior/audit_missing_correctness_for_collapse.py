import glob
import json
import os

import pandas as pd


BASE = "Your_path"
RAW_PATTERNS = [
    os.path.join(BASE, "trials_trialwise", "trialwise", "reknow*_trialwise.csv"),
    os.path.join(BASE, "trials_trialwise", "reknow*_trialwise.csv"),
]
HMM_TRIAL = os.path.join(BASE, "trials_trialwise", "hmm_long_tables", "hmm_trial_long.csv")
HMM_BLOCK = os.path.join(BASE, "trials_trialwise", "hmm_long_tables", "hmm_block_long.csv")

OUT_DIR = os.path.join(BASE, "trials_trialwise", "behav_paper_results", "missing_correctness_audit")
FINAL_DIR = os.path.join(BASE, "final", "results", "behavior")


def audit_raw_trialwise() -> pd.DataFrame:
    rows = []
    seen = set()
    for pattern in RAW_PATTERNS:
        for path in sorted(glob.glob(pattern)):
            key = os.path.basename(path)
            if key in seen:
                continue
            seen.add(key)
            d = pd.read_csv(path)
            if "correctness" not in d.columns:
                continue
            c = pd.to_numeric(d["correctness"], errors="coerce")
            rows.append(
                {
                    "file": key,
                    "subj": key.replace("_trialwise.csv", ""),
                    "n_trials": int(len(d)),
                    "n_missing_correctness": int(c.isna().sum()),
                    "missing_correctness_rate": float(c.isna().mean()),
                }
            )
    return pd.DataFrame(rows)


def audit_hmm_trial_long() -> pd.DataFrame:
    d = pd.read_csv(HMM_TRIAL)
    c = pd.to_numeric(d["correctness"], errors="coerce")
    flag = pd.to_numeric(d["missing_correctness_flag"], errors="coerce")
    rows = []
    for subj, g in d.groupby("subj", sort=True):
        gc = pd.to_numeric(g["correctness"], errors="coerce")
        gf = pd.to_numeric(g["missing_correctness_flag"], errors="coerce")
        rows.append(
            {
                "subj": subj,
                "n_trials": int(len(g)),
                "n_missing_correctness": int(gc.isna().sum()),
                "n_missing_correctness_flagged": int(gf.sum()),
                "missing_correctness_rate": float(gf.mean()),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["total_trials"] = int(len(d))
    out.attrs["total_missing_correctness"] = int(c.isna().sum())
    out.attrs["total_missing_correctness_flagged"] = int(flag.sum())
    return out

def audit_hmm_block_long() -> pd.DataFrame:
    d = pd.read_csv(HMM_BLOCK)
    if "missing_correctness_rate" not in d.columns:
        return pd.DataFrame()
    r = pd.to_numeric(d["missing_correctness_rate"], errors="coerce")
    rows = []
    for subj, g in d.groupby("subj", sort=True):
        gr = pd.to_numeric(g["missing_correctness_rate"], errors="coerce")
        rows.append(
            {
                "subj": subj,
                "n_blocks": int(len(g)),
                "n_blocks_with_missing_correctness": int((gr > 0).sum()),
                "max_block_missing_correctness_rate": float(gr.max()),
                "mean_block_missing_correctness_rate": float(gr.mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FINAL_DIR, exist_ok=True)

    raw = audit_raw_trialwise()
    hmm_trial = audit_hmm_trial_long()
    hmm_block = audit_hmm_block_long()

    raw_path = os.path.join(OUT_DIR, "raw_trialwise_missing_correctness_by_subject.csv")
    hmm_trial_path = os.path.join(OUT_DIR, "hmm_trial_long_missing_correctness_by_subject.csv")
    hmm_block_path = os.path.join(OUT_DIR, "hmm_block_long_missing_correctness_by_subject.csv")
    summary_path = os.path.join(OUT_DIR, "missing_correctness_audit_summary.json")

    raw.to_csv(raw_path, index=False)
    hmm_trial.to_csv(hmm_trial_path, index=False)
    hmm_block.to_csv(hmm_block_path, index=False)

    summary = {
        "raw_trialwise_n_subjects": int(raw["subj"].nunique()),
        "raw_trialwise_n_trials": int(raw["n_trials"].sum()),
        "raw_trialwise_n_missing_correctness": int(raw["n_missing_correctness"].sum()),
        "raw_trialwise_missing_correctness_rate": float(
            raw["n_missing_correctness"].sum() / raw["n_trials"].sum()
        ),
        "hmm_trial_long_n_subjects": int(hmm_trial["subj"].nunique()),
        "hmm_trial_long_n_trials": int(hmm_trial["n_trials"].sum()),
        "hmm_trial_long_n_missing_correctness": int(hmm_trial["n_missing_correctness"].sum()),
        "hmm_trial_long_n_missing_correctness_flagged": int(hmm_trial["n_missing_correctness_flagged"].sum()),
        "hmm_trial_long_missing_correctness_rate": float(
            hmm_trial["n_missing_correctness_flagged"].sum() / hmm_trial["n_trials"].sum()
        ),
        "hmm_block_long_n_blocks": int(hmm_block["n_blocks"].sum()),
        "hmm_block_long_n_blocks_with_missing_correctness": int(hmm_block["n_blocks_with_missing_correctness"].sum()),
        "hmm_block_long_max_block_missing_correctness_rate": float(hmm_block["max_block_missing_correctness_rate"].max()),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for path in [raw_path, hmm_trial_path, hmm_block_path, summary_path]:
        final_path = os.path.join(FINAL_DIR, os.path.basename(path))
        if path.endswith(".csv"):
            pd.read_csv(path).to_csv(final_path, index=False)
        else:
            with open(path, "r", encoding="utf-8") as src, open(final_path, "w", encoding="utf-8") as dst:
                dst.write(src.read())

    print(json.dumps(summary, indent=2))
    print(f"saved: {raw_path}")
    print(f"saved: {hmm_trial_path}")
    print(f"saved: {hmm_block_path}")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
