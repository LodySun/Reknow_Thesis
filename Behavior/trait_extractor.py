import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from trialwise_parser import build_trialwise_dataframe
except ModuleNotFoundError:
    from codes.behav.trialwise_parser import build_trialwise_dataframe

LOGS_DIR = "logs_dir"
OUT_DIR = "out_dir"
RULES = ["G1", "G2", "L1", "L2"]
RULE_IDX = {r: i for i, r in enumerate(RULES)}


def _safe_json_loads(x):
    if not isinstance(x, str) or not x.strip():
        return None
    try:
        return json.loads(x)
    except Exception:
        return None


def _trimmed_mean(series: pd.Series, proportion_to_cut: float = 0.10) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna().sort_values().to_numpy(dtype=float)
    n = len(vals)
    if n == 0:
        return np.nan
    k = int(np.floor(n * proportion_to_cut))
    if 2 * k >= n:
        return float(np.mean(vals))
    trimmed = vals[k : n - k]
    return float(np.mean(trimmed))


def _pooled_sd(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s1 = a.std(ddof=1)
    s2 = b.std(ddof=1)
    n1 = len(a)
    n2 = len(b)
    denom = (n1 + n2 - 2)
    if denom <= 0:
        return np.nan
    return np.sqrt(((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / denom)


def _mean_slope_by_block(df: pd.DataFrame, phase: str) -> float:
    slopes: List[float] = []
    for _, bdf in df[df["phase"] == phase].groupby("block_id"):
        bdf = bdf.sort_values("block_position")
        if len(bdf) < 2:
            continue
        x = pd.to_numeric(bdf["block_position"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(bdf["rt"], errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 2:
            continue
        slopes.append(float(np.polyfit(x[mask], y[mask], 1)[0]))
    return float(np.mean(slopes)) if slopes else np.nan


def _pred_correct_for_card(row: pd.Series, rule: str, card_num: int) -> Optional[int]:
    idx = RULE_IDX.get(rule)
    if idx is None:
        return None
    stim = _safe_json_loads(row.get("stimulus_state_vector"))
    cards = _safe_json_loads(row.get("target_card_states"))
    if not isinstance(stim, list) or len(stim) != 4:
        return None
    if not isinstance(cards, list) or len(cards) < 1:
        return None
    if card_num < 1 or card_num > len(cards):
        return None
    card_vec = cards[card_num - 1]
    if not isinstance(card_vec, list) or len(card_vec) != 4:
        return None
    return int(card_vec[idx] == stim[idx])


def _candidate_pred_correct(row: pd.Series, rule: str) -> Optional[int]:
    """
    Returns predicted correctness (1/0) if the tested rule were true,
    based on chosen response card, stimulus_state_vector and target_card_states.
    """
    resp = row.get("response_card")
    if pd.isna(resp):
        return None

    try:
        chosen = abs(int(resp))  # logs encode wrong answers as negative card ids
    except Exception:
        return None

    return _pred_correct_for_card(row, rule, chosen)


def _search_inference_dynamics(df: pd.DataFrame, subj: str):
    """
    Build trial-level inference dynamics across whole blocks.
    Phase labels are kept from behavior (participant-centered), while
    logic-space variables are computed whenever their domain is valid.
    - n_possible_rules_before/after
    - hypothesis reduction per feedback
    - chosen-card information gain vs maximum possible
    - confirmatory-choice marker
    """
    rows: List[Dict[str, float]] = []

    for block_id, bdf in df.groupby("block_id"):
        bdf = bdf.sort_values("block_position")
        candidates = list(RULES)

        for _, row in bdf.iterrows():
            n_before = len(candidates)
            cards = _safe_json_loads(row.get("target_card_states"))
            n_cards = len(cards) if isinstance(cards, list) else 4

            card_stats = []
            for c in range(1, n_cards + 1):
                r_plus = []
                valid_rules = []
                for r in candidates:
                    pred = _pred_correct_for_card(row, r, c)
                    if pred is None:
                        continue
                    valid_rules.append(r)
                    if pred == 1:
                        r_plus.append(r)
                n_valid = len(valid_rules)
                if n_valid == 0:
                    continue
                n1 = len(r_plus)
                n0 = n_valid - n1
                p_correct = n1 / n_valid
                # E[|C_{t+1}| | action=a] under uniform prior over current candidates:
                # p(correct)*|R+(a)| + (1-p(correct))*(|C_t|-|R+(a)|)
                expected_remaining = (p_correct * n1) + ((1 - p_correct) * n0)
                info_gain = n_valid - expected_remaining
                card_stats.append(
                    {
                        "card": c,
                        "p_correct": p_correct,
                        "info_gain": info_gain,
                    }
                )

            chosen_ig = np.nan
            chosen_pc = np.nan
            info_max = np.nan
            pcorrect_max = np.nan
            info_eff = np.nan
            confirmatory_choice = np.nan
            delta_p = np.nan
            delta_ig = np.nan
            eligible_confirmatory = 0.0

            try:
                chosen_card = abs(int(row["response_card"]))
            except Exception:
                chosen_card = None

            if card_stats:
                _info_max = max(s["info_gain"] for s in card_stats)
                _pcorrect_max = max(s["p_correct"] for s in card_stats)

                tol = 1e-12
                p_argmax = {s["card"] for s in card_stats if s["p_correct"] >= (_pcorrect_max - tol)}
                ig_argmax = {s["card"] for s in card_stats if s["info_gain"] >= (_info_max - tol)}

                # Confirmatory metrics are defined only when choice objectives are separable.
                eligible_confirmatory = float((n_before >= 3) and (p_argmax != ig_argmax))
                if eligible_confirmatory == 1.0:
                    info_max = _info_max
                    pcorrect_max = _pcorrect_max
                    chosen_stat = next((s for s in card_stats if s["card"] == chosen_card), None)
                    if chosen_stat is not None:
                        chosen_ig = chosen_stat["info_gain"]
                        chosen_pc = chosen_stat["p_correct"]
                        info_eff = (chosen_ig / info_max) if info_max > 0 else np.nan
                        delta_p = chosen_pc - pcorrect_max
                        delta_ig = chosen_ig - info_max
                        confirmatory_choice = float((chosen_card in p_argmax) and (chosen_card not in ig_argmax))

            obs = int(row["correctness"]) if not pd.isna(row["correctness"]) else 0
            next_candidates = []
            for r in candidates:
                pred = _candidate_pred_correct(row, r)
                if pred is None or pred == obs:
                    next_candidates.append(r)
            if len(next_candidates) == 0:
                next_candidates = candidates[:]

            n_after = len(next_candidates)
            eligible_reduction = float(n_before > 1)
            if eligible_reduction == 1.0:
                reduction = n_before - n_after
                reduction_ratio = (reduction / n_before) if n_before > 0 else np.nan
            else:
                reduction = np.nan
                reduction_ratio = np.nan

            rows.append(
                {
                    "subj": subj,
                    "block_id": block_id,
                    "trial_id": row["trial_id"],
                    "phase": row.get("phase"),
                    "block_position": row.get("block_position"),
                    "correctness": row.get("correctness"),
                    "n_possible_rules_before": n_before,
                    "n_possible_rules_after": n_after,
                    "hypothesis_reduction": reduction,
                    "hypothesis_reduction_ratio": reduction_ratio,
                    "eligible_reduction_trial": eligible_reduction,
                    "chosen_info_gain": chosen_ig,
                    "max_info_gain": info_max,
                    "information_efficiency": info_eff,
                    "chosen_p_correct": chosen_pc,
                    "max_p_correct": pcorrect_max,
                    "delta_p": delta_p,
                    "delta_ig": delta_ig,
                    "confirmatory_choice": confirmatory_choice,
                    "eligible_confirmatory_trial": eligible_confirmatory,
                }
            )
            candidates = next_candidates

    return pd.DataFrame(rows)


def _build_block_traits(df: pd.DataFrame, dynamics: pd.DataFrame, subj: str) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for block_id, bdf in df.groupby("block_id"):
        bdf = bdf.sort_values("block_position")
        ddf = dynamics[dynamics["block_id"] == block_id].sort_values("block_position")
        if ddf.empty:
            continue

        search_len_k = int((bdf["phase"] == "search").sum())

        # Both onsets use the same coordinate system: block_position (1-based trial index from parser).
        collapse_series = ddf[ddf["n_possible_rules_before"] == 1]
        collapse_onset = (
            pd.to_numeric(collapse_series["block_position"], errors="coerce").iloc[0]
            if not collapse_series.empty
            else np.nan
        )
        collapse_defined = float(not pd.isna(collapse_onset))

        stable_series = bdf[bdf["phase"] != "search"]
        stable_onset = (
            pd.to_numeric(stable_series["block_position"], errors="coerce").iloc[0]
            if not stable_series.empty
            else np.nan
        )
        stable_defined = float(not pd.isna(stable_onset))
        collapse_to_stable = np.nan
        if not pd.isna(collapse_onset) and not pd.isna(stable_onset):
            diff = float(stable_onset - collapse_onset)
            collapse_to_stable = diff if diff >= 0 else np.nan
        collapse_to_stable_defined = float(not pd.isna(collapse_to_stable))

        if not collapse_series.empty:
            post_collapse_fail_rate = float(
                (pd.to_numeric(collapse_series["correctness"], errors="coerce") == 0).mean()
            )
        else:
            post_collapse_fail_rate = np.nan

        search_ddf = ddf[ddf["phase"] == "search"]
        red_vals = pd.to_numeric(
            search_ddf.loc[search_ddf["eligible_reduction_trial"] == 1, "hypothesis_reduction_ratio"],
            errors="coerce",
        ).dropna()
        block_reduction_ratio = float(red_vals.mean()) if len(red_vals) else np.nan

        conf_vals = pd.to_numeric(
            search_ddf.loc[search_ddf["eligible_confirmatory_trial"] == 1, "information_efficiency"],
            errors="coerce",
        ).dropna()
        block_information_efficiency = float(conf_vals.mean()) if len(conf_vals) else np.nan

        cbi_vals = pd.to_numeric(
            search_ddf.loc[search_ddf["eligible_confirmatory_trial"] == 1, "confirmatory_choice"],
            errors="coerce",
        ).dropna()
        block_confirmatory_bias = float(cbi_vals.mean()) if len(cbi_vals) else np.nan
        n_eligible_confirmatory_trials = int(len(cbi_vals))

        rows.append(
            {
                "subj": subj,
                "block_id": block_id,
                "search_len_k": search_len_k,
                "collapse_onset": collapse_onset,
                "collapse_defined": collapse_defined,
                "stable_defined": stable_defined,
                "collapse_to_stable_defined": collapse_to_stable_defined,
                "collapse_to_stable": collapse_to_stable,
                "post_collapse_fail_rate": post_collapse_fail_rate,
                "block_hypothesis_reduction_ratio": block_reduction_ratio,
                "block_information_efficiency": block_information_efficiency,
                "block_confirmatory_bias": block_confirmatory_bias,
                "n_eligible_confirmatory_trials": n_eligible_confirmatory_trials,
            }
        )
    return pd.DataFrame(rows)


def _over_search_index(df: pd.DataFrame) -> Dict[str, float]:
    over_search_error_trials = 0
    total_search_trials = 0
    total_search_errors = 0

    work = df.copy()
    work["block_num"] = pd.to_numeric(work["block_id"], errors="coerce")
    work["trial_num"] = pd.to_numeric(work["trial_id"], errors="coerce")
    work = work.sort_values(["block_num", "trial_num"]).reset_index(drop=True)

    for _, bdf in work.groupby("block_id", sort=False):
        bdf = bdf.sort_values("block_position")
        search = bdf[bdf["phase"] == "search"].copy()
        if search.empty:
            continue
        candidates = set(RULES)

        for _, row in search.iterrows():
            total_search_trials += 1
            obs = int(row["correctness"]) if not pd.isna(row["correctness"]) else 0
            if obs == 0:
                total_search_errors += 1

            # Over-search error: rule is already logically identified,
            # yet subject is still making search-phase errors.
            if len(candidates) == 1 and obs == 0:
                over_search_error_trials += 1

            next_candidates = set()
            for r in candidates:
                pred = _candidate_pred_correct(row, r)
                if pred is None or pred == obs:
                    next_candidates.add(r)
            candidates = next_candidates if next_candidates else candidates

    return {
        "over_search_index": (
            over_search_error_trials / total_search_trials if total_search_trials > 0 else np.nan
        ),
        "over_search_error_fraction": (
            over_search_error_trials / total_search_errors if total_search_errors > 0 else np.nan
        ),
    }


def extract_subject_traits(df: pd.DataFrame, subj: str, dynamics: pd.DataFrame) -> Dict[str, float]:
    # Build block-level traits first, then aggregate to subject-level.
    block_traits = _build_block_traits(df, dynamics, subj)
    out: Dict[str, float] = {"subj": subj}

    if block_traits.empty:
        out["search_len_k_median"] = np.nan
        out["search_len_k_mean"] = np.nan
        out["collapse_to_stable_mean"] = np.nan
        out["collapse_to_stable_trimmed_mean"] = np.nan
        out["collapse_to_stable_p75"] = np.nan
        out["collapse_to_stable_p90"] = np.nan
        out["slow_collapse_to_stable_rate_ge2"] = np.nan
        out["n_collapse_to_stable_defined_blocks"] = 0.0
        out["post_collapse_fail_rate_mean"] = np.nan
        out["hypothesis_space_reduction_ratio"] = np.nan
        out["information_efficiency_mean"] = np.nan
        out["confirmatory_bias_index"] = np.nan
        out["n_eligible_confirmatory_trials"] = 0.0
        return out

    out["search_len_k_median"] = float(pd.to_numeric(block_traits["search_len_k"], errors="coerce").median())
    out["search_len_k_mean"] = float(pd.to_numeric(block_traits["search_len_k"], errors="coerce").mean())
    cts = pd.to_numeric(block_traits["collapse_to_stable"], errors="coerce").dropna()
    out["collapse_to_stable_mean"] = float(cts.mean()) if len(cts) else np.nan
    out["collapse_to_stable_trimmed_mean"] = _trimmed_mean(cts, proportion_to_cut=0.10)
    out["collapse_to_stable_p75"] = float(cts.quantile(0.75)) if len(cts) else np.nan
    out["collapse_to_stable_p90"] = float(cts.quantile(0.90)) if len(cts) else np.nan
    out["slow_collapse_to_stable_rate_ge2"] = float((cts >= 2).mean()) if len(cts) else np.nan
    out["n_collapse_to_stable_defined_blocks"] = float(len(cts))
    out["post_collapse_fail_rate_mean"] = float(
        pd.to_numeric(block_traits["post_collapse_fail_rate"], errors="coerce").mean()
    )
    out["hypothesis_space_reduction_ratio"] = float(
        pd.to_numeric(block_traits["block_hypothesis_reduction_ratio"], errors="coerce").mean()
    )
    out["information_efficiency_mean"] = float(
        pd.to_numeric(block_traits["block_information_efficiency"], errors="coerce").mean()
    )
    out["confirmatory_bias_index"] = float(
        pd.to_numeric(block_traits["block_confirmatory_bias"], errors="coerce").mean()
    )
    out["n_eligible_confirmatory_trials"] = float(
        pd.to_numeric(block_traits["n_eligible_confirmatory_trials"], errors="coerce").sum()
    )
    return out


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    rows: List[Dict[str, float]] = []
    all_dynamics: List[pd.DataFrame] = []
    all_block_traits: List[pd.DataFrame] = []

    for i in range(1, 35):
        subj = f"reknow{i:03d}"
        log_path = os.path.join(LOGS_DIR, f"{subj}.log")
        if not os.path.exists(log_path):
            print(f"[WARN] missing log: {log_path}")
            continue
        trial_df = build_trialwise_dataframe(log_path, subj, include_aux=True)
        dynamics_df = _search_inference_dynamics(trial_df, subj)
        block_traits_df = _build_block_traits(trial_df, dynamics_df, subj)
        rows.append(extract_subject_traits(trial_df, subj, dynamics_df))
        all_dynamics.append(dynamics_df)
        all_block_traits.append(block_traits_df)
        print(f"traits extracted: {subj}")

    if rows:
        trait_df = pd.DataFrame(rows)
        numeric_cols = trait_df.select_dtypes(include=[np.number]).columns
        trait_df[numeric_cols] = trait_df[numeric_cols].round(3)
        out_path = os.path.join(OUT_DIR, "subject_traits.csv")
        trait_df.to_csv(out_path, index=False)
        print(f"saved traits -> {out_path} ({len(trait_df)} subjects)")
    if all_block_traits:
        block_df = pd.concat(all_block_traits, ignore_index=True)
        numeric_cols = block_df.select_dtypes(include=[np.number]).columns
        block_df[numeric_cols] = block_df[numeric_cols].round(3)
        block_path = os.path.join(OUT_DIR, "block_traits.csv")
        block_df.to_csv(block_path, index=False)
        print(f"saved block traits -> {block_path} ({len(block_df)} rows)")
    if all_dynamics:
        dynamics_df = pd.concat(all_dynamics, ignore_index=True)
        dynamics_df["_block_num_sort"] = pd.to_numeric(dynamics_df["block_id"], errors="coerce")
        dynamics_df["_trial_num_sort"] = pd.to_numeric(dynamics_df["trial_id"], errors="coerce")
        dynamics_df = dynamics_df.sort_values(
            ["subj", "_block_num_sort", "_trial_num_sort"]
        ).reset_index(drop=True)
        dynamics_df = dynamics_df.drop(columns=["_block_num_sort", "_trial_num_sort"])
        numeric_cols = dynamics_df.select_dtypes(include=[np.number]).columns
        dynamics_df[numeric_cols] = dynamics_df[numeric_cols].round(3)
        dyn_path = os.path.join(OUT_DIR, "search_inference_dynamics.csv")
        dynamics_df.to_csv(dyn_path, index=False)
        print(f"saved dynamics -> {dyn_path} ({len(dynamics_df)} rows)")
