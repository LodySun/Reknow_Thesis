import re
import os
import pandas as pd

WS_SPLIT = re.compile(r"\s+")

def parse_log(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = WS_SPLIT.split(line)
          
            # Only keep real data line, exclude start json, etc.
            if len(parts) < 6:
                continue
            try:
                rel_time = float(parts[0])
                ts = float(parts[1])
            except ValueError:
                continue

            if not parts[2].isdigit():
                # If 3rd column is not number then skip
                continue

            num = int(parts[2])
            code = parts[3]
            block = parts[4]
            trial = parts[5]
            rest = " ".join(parts[6:]) if len(parts) > 6 else ""
            
            rows.append({
                "rel_time": rel_time,
                "ts": ts,
                "num": num,
                "code": code,
                "block": block,
                "trial": trial,
                "rest": rest
            })
    return pd.DataFrame(rows)

def parse_rsp_rest(rest: str) -> pd.Series:
    parts = rest.split()
    correct = None
    rule = None
    if len(parts) >= 3 and parts[0] == "RESPONSE":
        if parts[1] in ("0", "1"):
            correct = int(parts[1])
        rule = parts[2]
    return pd.Series({"correct": correct, "rule": rule})

def trials_from_log(log_path: str) -> pd.DataFrame:
    df = parse_log(log_path)

    tgt = (
    df[df["code"] == "TGT"]
    [["block", "trial", "rel_time"]]
    .rename(columns={"rel_time": "tgt_time"})
)

    rsp = (
    df[df["code"] == "RSP"]
    [["block", "trial", "rel_time", "rest"]]
    .rename(columns={"rel_time": "rsp_time"})
)

    rsp[["correct", "rule"]] = rsp["rest"].apply(parse_rsp_rest)

    trials = tgt.merge(rsp.drop(columns=["rest"]), on=["block", "trial"], how="inner")

    # RT = rsp - tgt
    trials["rt"] = (trials["rsp_time"] - trials["tgt_time"]).round(3)

    # Keep the columns needed
    trial_csv = trials[["block", "trial", "rt", "correct", "rule"]].copy()
    return trial_csv

if __name__ == "__main__":
    logs_dir = "logs_dir"
    out_dir = "out_dir"
    os.makedirs(out_dir, exist_ok=True)

    all_trials = []

    for i in range(1, 35):
        subj_id = f"reknow{i:03d}"
        log_path = os.path.join(logs_dir, f"{subj_id}.log")
        if not os.path.exists(log_path):
            print(f"[WARN] log not found, skip: {log_path}")
            continue
        
        print(f"Processing {subj_id} ...")
        trial_df = trials_from_log(log_path)
        trial_df.insert(0, "subj", subj_id)

        out_path = os.path.join(out_dir, f"{subj_id}_trials.csv")
        trial_df.to_csv(out_path, index=False)
        print(f"  saved {len(trial_df)} trials to {out_path}")

        all_trials.append(trial_df)
