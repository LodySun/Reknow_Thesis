import os
from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib import rcParams


BASE_SOL = "base_sol_dir"
METRICS_PATH = os.path.join(BASE_SOL, "solidity_figure_metrics.csv")


def set_helvetica_font():
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]


def p_to_sig(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def update_metrics_file(
    figure_id: str,
    figure_title: str,
    figure_caption: str,
    rows: List[Dict],
    out_path: str = METRICS_PATH,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    normalized = []
    for r in rows:
        rec = {
            "figure_id": figure_id,
            "figure_title": figure_title,
            "figure_caption": figure_caption,
            "panel": r.get("panel", ""),
            "metric_name": r.get("metric_name", ""),
            "p_value": r.get("p_value", np.nan),
            "significance": r.get("significance", ""),
            "effect_or_stat": r.get("effect_or_stat", ""),
            "notes": r.get("notes", ""),
        }
        normalized.append(rec)

    new_df = pd.DataFrame(normalized)
    if os.path.exists(out_path):
        old_df = pd.read_csv(out_path)
        old_df = old_df[old_df["figure_id"] != figure_id].copy()
        out_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        out_df = new_df
    out_df.to_csv(out_path, index=False)
