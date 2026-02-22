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
    "L2": "col",
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

