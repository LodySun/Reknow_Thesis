import json
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

# Define log file analysis settings
WS_SPLIT = re.compile(r"\s+")

# STM: stimulus state vector, rule = G/L + num
STM_PATTERN = re.compile(r"STIMULUS\s+([0-9]+(?:,[0-9]+){3})\s+RULE\s+([GL]\d)")

# RSP: response correctness, rule, answer card
RSP_PATTERN = re.compile(r"RESPONSE\s+([01])\s+([GL]\d)\s+ANSWER\s+(-?\d+)")

# FDB: feedback label. correct answers, series length
FDB_PATTERN = re.compile(r"FEEDBACK\s+(\w+)\s+(\d+)\s*/\s*(\d+)")

EARLY_FOUND_K = 2                # Early found: first k-consecutive-correct streak window, in here I used the hypothesis to define some index
MIN_FORMAL_BLOCK = 13            # First 12 blocks are practice blocks that don't count
EXCLUDE_REDUCED_BLOCKS = True    # Remove the 20% reduced blocks

RULE_TYPE_MAP = {
    "G1": "obj",
    "G2": "col",
    "L1": "obj",
    "L2": "ori",
}

# Analyze the log files row by row, avoid dirty data
def parse_log(path: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = WS_SPLIT.split(line)
            if len(parts) < 6:
                continue

            try:
                rel_time = float(parts[0])
                ts = float(parts[1])
            except ValueError:
                continue

            if not parts[2].isdigit():
                continue

            rows.append(
                {
                    "rel_time": rel_time,
                    "ts": ts,
                    "num": int(parts[2]),
                    "code": parts[3],
                    "block": parts[4],
                    "trial": parts[5],
                    "rest": " ".join(parts[6:]) if len(parts) > 6 else "",
                }
            )
    return pd.DataFrame(rows)

def _safe_int_list(csv_like: str) -> Optional[List[int]]:
    try:
        return [int(x.strip()) for x in csv_like.split(",")]
    except Exception:
        return None

# Analysis of stm logs
def parse_stm(rest: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "true_rule": None,
        "stimulus_state_vector": None,
        "g1_state": None,
        "g2_state": None,
        "l1_state": None,
        "l2_state": None,
    }
    m = STM_PATTERN.search(rest)
    if not m:
        return out

    vec = _safe_int_list(m.group(1))
    rule = m.group(2)
    out["true_rule"] = rule
    out["stimulus_state_vector"] = json.dumps(vec) if vec is not None else None
    if vec and len(vec) == 4:
        out["g1_state"], out["g2_state"], out["l1_state"], out["l2_state"] = vec
    return out

# Analysis of tgt logs
def parse_tgt(rest: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"target_card_states": None}
    if not rest.startswith("TARGET"):
        return out
    payload = rest[len("TARGET") :].strip()
    card_chunks = [chunk.strip() for chunk in payload.split(";") if chunk.strip()]
    cards: List[List[int]] = []
    for chunk in card_chunks:
        vec = _safe_int_list(chunk)
        if vec is not None:
            cards.append(vec)
    if cards:
        out["target_card_states"] = json.dumps(cards)
    return out

# Analysis of rsp logs
def parse_rsp(rest: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "response_card": None,
        "correctness": None,
        "rsp_rule": None,
    }
    m = RSP_PATTERN.search(rest)
    if not m:
        return out
    out["correctness"] = int(m.group(1))
    out["rsp_rule"] = m.group(2)
    out["response_card"] = int(m.group(3))
    return out


# Analysis of fdb logs
def parse_fdb(rest: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "feedback_label": None,
        "feedback_correct_answers": None,
        "feedback_series_length": None,
    }
    m = FDB_PATTERN.search(rest)
    if not m:
        return out
    out["feedback_label"] = m.group(1)
    out["feedback_correct_answers"] = int(m.group(2))
    out["feedback_series_length"] = int(m.group(3))
    return out

def _rule_level(rule: Any) -> Optional[str]:
    if isinstance(rule, str):
        if rule.startswith("G"):
            return "global"
        if rule.startswith("L"):
            return "local"
    return None


def _rule_type(rule: Any) -> Optional[str]:
    if not isinstance(rule, str):
        return None
    return RULE_TYPE_MAP.get(rule, "unknown")


def assign_phase_one_block(block_df: pd.DataFrame, early_found_k: int = EARLY_FOUND_K) -> pd.DataFrame:
    tmp = block_df.sort_values("block_position").reset_index(drop=True).copy()
    corr = pd.to_numeric(tmp["correctness"], errors="coerce").fillna(0).astype(int)
    k = max(int(early_found_k), 1)

    # Find first window of k consecutive correct (rolling sum == k).
    streak_end_idx = corr.rolling(k, min_periods=k).sum()
    first_streak_end = streak_end_idx[streak_end_idx == k].index.min()

    # No k-consecutive-correct streak in this block -> all search.
    if pd.isna(first_streak_end):
        tmp["phase"] = "search"
        return tmp

    end_idx = int(first_streak_end)
    start_idx = end_idx - k + 1
    phase_values = np.where(
        tmp.index < start_idx,
        "search",
        np.where(tmp.index <= end_idx, "early-found", "late-found"),
    )
    tmp["phase"] = phase_values
    return tmp


def _infer_block_stimulus_condition(trials: pd.DataFrame, tol: float = 1e-3) -> pd.DataFrame:
    if "stimulus_state_vector" not in trials.columns:
        return pd.DataFrame(columns=["block", "stimulus_condition"])

    tmp = trials[["block", "stimulus_state_vector"]].dropna().copy()
    if tmp.empty:
        return pd.DataFrame(columns=["block", "stimulus_condition"])

    def _parse_vec(x: Any) -> Optional[List[int]]:
        try:
            vec = json.loads(str(x))
            if isinstance(vec, list) and len(vec) == 4:
                return [int(v) for v in vec]
        except Exception:
            return None
        return None

    tmp["vec"] = tmp["stimulus_state_vector"].apply(_parse_vec)
    tmp = tmp[tmp["vec"].notna()].copy()
    if tmp.empty:
        return pd.DataFrame(columns=["block", "stimulus_condition"])

    vec_df = pd.DataFrame(tmp["vec"].tolist(), columns=["g1", "g2", "l1", "l2"], index=tmp.index)
    tmp = pd.concat([tmp[["block"]], vec_df], axis=1)
    tmp["g_diff"] = (tmp["g1"] != tmp["g2"]).astype(float)
    tmp["l_diff"] = (tmp["l1"] != tmp["l2"]).astype(float)

    agg = tmp.groupby("block", as_index=False).agg(
        g_diff_rate=("g_diff", "mean"),
        l_diff_rate=("l_diff", "mean"),
    )

    def _label(row: pd.Series) -> str:
        g_ok = float(row["g_diff_rate"]) > tol
        l_ok = float(row["l_diff_rate"]) > tol
        if g_ok and l_ok:
            return "full"
        if g_ok and (not l_ok):
            return "reduced_global"
        if (not g_ok) and l_ok:
            return "reduced_local"
        return "unknown"

    agg["stimulus_condition"] = agg.apply(_label, axis=1)
    return agg[["block", "stimulus_condition"]]


def build_trialwise_dataframe(log_path: str, subj_id: str, include_aux: bool = False) -> pd.DataFrame:
    events = parse_log(log_path)
    if events.empty:
        return pd.DataFrame()

    events["block_num"] = pd.to_numeric(events["block"], errors="coerce")
    events["trial_num"] = pd.to_numeric(events["trial"], errors="coerce")
    events = events[
        (events["block_num"] >= MIN_FORMAL_BLOCK) & (events["trial_num"] >= 1)
    ].copy()

    key_cols = ["block", "trial"]
    base = (
        events[key_cols]
        .drop_duplicates()
        .sort_values(["block", "trial"])
        .reset_index(drop=True)
    )

    stm = events[events["code"] == "STM"][key_cols + ["rel_time", "rest"]].copy()
    stm_parsed = stm["rest"].apply(parse_stm).apply(pd.Series)
    stm = pd.concat(
        [stm[key_cols + ["rel_time"]].rename(columns={"rel_time": "stm_time"}), stm_parsed],
        axis=1,
    )

    tgt = events[events["code"] == "TGT"][key_cols + ["rel_time", "rest"]].copy()
    tgt_parsed = tgt["rest"].apply(parse_tgt).apply(pd.Series)
    tgt = pd.concat(
        [tgt[key_cols + ["rel_time"]].rename(columns={"rel_time": "tgt_time"}), tgt_parsed],
        axis=1,
    )

    rsp = events[events["code"] == "RSP"][key_cols + ["rel_time", "rest"]].copy()
    rsp_parsed = rsp["rest"].apply(parse_rsp).apply(pd.Series)
    rsp = pd.concat(
        [rsp[key_cols + ["rel_time"]].rename(columns={"rel_time": "rsp_time"}), rsp_parsed],
        axis=1,
    )

    fdb = events[events["code"] == "FDB"][key_cols + ["rel_time", "rest"]].copy()
    fdb_parsed = fdb["rest"].apply(parse_fdb).apply(pd.Series)
    fdb = pd.concat(
        [fdb[key_cols + ["rel_time"]].rename(columns={"rel_time": "fdb_time"}), fdb_parsed],
        axis=1,
    )

    trials = base.merge(stm, on=key_cols, how="left")
    trials = trials.merge(tgt, on=key_cols, how="left")
    trials = trials.merge(rsp, on=key_cols, how="left")
    trials = trials.merge(fdb, on=key_cols, how="left")

    trials["subj"] = subj_id
    trials["block_id"] = trials["block"]
    trials["trial_id"] = trials["trial"]
    trials["block_position"] = pd.to_numeric(trials["trial"], errors="coerce")
    trials["rule_level"] = trials["true_rule"].apply(_rule_level)
    trials["rule_type"] = trials["true_rule"].apply(_rule_type)
    trials["rt"] = (trials["rsp_time"] - trials["tgt_time"]).round(3)

    if EXCLUDE_REDUCED_BLOCKS:
        cond_df = _infer_block_stimulus_condition(trials)
        keep_blocks = set(cond_df.loc[cond_df["stimulus_condition"] == "full", "block"].astype(str))
        trials = trials[trials["block"].astype(str).isin(keep_blocks)].copy()

    if trials.empty:
        return pd.DataFrame()

    trials["block_num"] = pd.to_numeric(trials["block"], errors="coerce")
    block_rule = (
        trials.groupby(["block", "block_num"], as_index=False)["true_rule"]
        .agg(lambda s: s.dropna().iloc[0] if not s.dropna().empty else None)
        .sort_values("block_num")
        .reset_index(drop=True)
    )
    block_rule["prev_rule"] = block_rule["true_rule"].shift(1)
    block_rule["is_rule_switch_block"] = (
        block_rule["prev_rule"].notna() & (block_rule["true_rule"] != block_rule["prev_rule"])
    )
    trials = trials.merge(
        block_rule[["block", "is_rule_switch_block", "prev_rule"]],
        on="block",
        how="left",
    )
    trials["trial_since_rule_switch"] = trials["block_position"].where(
        trials["is_rule_switch_block"] == True, pd.NA
    )

    # Assign conservative, discrete phases within each block.
    phased_blocks: List[pd.DataFrame] = []
    for _, bdf in trials.groupby("block_id"):
        phased_blocks.append(assign_phase_one_block(bdf, early_found_k=EARLY_FOUND_K))
    trials = pd.concat(phased_blocks, ignore_index=True)

    # Slim output columns for downstream behavioral analysis.
    ordered_cols = [
        "subj",
        "block_id",
        "trial_id",
        "block_position",
        "trial_since_rule_switch",
        "is_rule_switch_block",
        "phase",
        "true_rule",
        "rule_level",
        "rule_type",
        "stimulus_state_vector",
        "response_card",
        "correctness",
        "feedback_series_length",
        "rt",
    ]
    if include_aux:
        ordered_cols.append("target_card_states")
    out = trials[ordered_cols].copy()
    out["_block_num_sort"] = pd.to_numeric(out["block_id"], errors="coerce")
    out["_trial_num_sort"] = pd.to_numeric(out["trial_id"], errors="coerce")
    out = out.sort_values(["_block_num_sort", "_trial_num_sort"]).reset_index(drop=True)
    out = out.drop(columns=["_block_num_sort", "_trial_num_sort"])
    return out


if __name__ == "__main__":
    logs_dir = "your_logs_dir"
    out_dir = "your_out_dir"
    os.makedirs(out_dir, exist_ok=True)

    all_subjects: List[pd.DataFrame] = []
    for i in range(1, 35):
        subj_id = f"reknow{i:03d}"
        log_path = os.path.join(logs_dir, f"{subj_id}.log")
        if not os.path.exists(log_path):
            print(f"[WARN] missing log: {log_path}")
            continue
        trial_df = build_trialwise_dataframe(log_path, subj_id)
        out_path = os.path.join(out_dir, f"{subj_id}_trialwise.csv")
        trial_df.to_csv(out_path, index=False)
        print(f"saved {len(trial_df)} rows -> {out_path}")
        all_subjects.append(trial_df)

    if all_subjects:
        combined = pd.concat(all_subjects, ignore_index=True)
        combined_path = os.path.join(out_dir, "all_subjects_trialwise.csv")
        combined.to_csv(combined_path, index=False)
        print(f"saved {len(combined)} rows -> {combined_path}")
