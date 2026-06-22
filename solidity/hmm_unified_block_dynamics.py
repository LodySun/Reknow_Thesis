import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
except Exception:
    smf = None


BASE = "base_dir"
IN_CSV = os.path.join(BASE, "trials_trialwise", "all_subjects_trialwise.csv")
TRIALWISE_PER_SUBJ_DIR = os.path.join(BASE, "trials_trialwise", "trialwise")
OUT_DIR = os.path.join(BASE, "trials_trialwise", "1s_comp", "eeg_paper_results", "solidity")

# Unified 2-state HMM fitting controls
EPS = 1e-9
MAX_ITER = 60
TOL = 1e-5
N_INIT = 4
RNG_SEED = 20260409

# Core acquisition definition (as requested)
ACQ_THRESH = 0.80
ACQ_MIN_RUN = 2

# Transition width definition
LOW_THRESH = 0.20
HIGH_THRESH = 0.80


@dataclass
class SequenceData:
    subj: str
    block_id: str
    trial_ids: np.ndarray
    y: np.ndarray


def _safe_prob(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 1e-6, 1 - 1e-6)


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    out = mat.copy().astype(float)
    rs = out.sum(axis=1, keepdims=True)
    rs[rs <= 0] = 1.0
    return out / rs


def _logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True) + EPS)
    if axis is None:
        return out.reshape(())
    return np.squeeze(out, axis=axis)


def _prepare_sequences(df: pd.DataFrame) -> List[SequenceData]:
    keep = df[["subj", "block_id", "trial_id", "correctness"]].copy()
    keep["correctness"] = pd.to_numeric(keep["correctness"], errors="coerce")
    keep["trial_num"] = pd.to_numeric(keep["trial_id"], errors="coerce")
    keep["block_num"] = pd.to_numeric(keep["block_id"], errors="coerce")
    keep = keep.sort_values(["subj", "block_num", "trial_num"])

    seqs: List[SequenceData] = []
    for (subj, block_id), bdf in keep.groupby(["subj", "block_id"], sort=False):
        b = bdf[bdf["correctness"].isin([0, 1])].copy()
        if b.empty:
            continue
        seqs.append(
            SequenceData(
                subj=str(subj),
                block_id=str(block_id),
                trial_ids=b["trial_id"].astype(str).to_numpy(),
                y=b["correctness"].astype(int).to_numpy(),
            )
        )
    return seqs


def _emission_prob(y: np.ndarray, e: np.ndarray) -> np.ndarray:
    y01 = y.astype(float).reshape(-1, 1)
    p1 = e.reshape(1, 2)
    return y01 * p1 + (1 - y01) * (1 - p1)


def _forward_backward(
    seq: SequenceData,
    pi: np.ndarray,
    A: np.ndarray,
    e: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    y = seq.y
    T = len(y)
    B = _safe_prob(_emission_prob(y, e))
    log_pi = np.log(_safe_prob(pi))
    log_A = np.log(_safe_prob(A))
    log_B = np.log(B)

    alpha = np.zeros((T, 2), dtype=float)
    beta = np.zeros((T, 2), dtype=float)
    alpha[0] = log_pi + log_B[0]
    for t in range(1, T):
        alpha[t] = log_B[t] + _logsumexp(alpha[t - 1].reshape(2, 1) + log_A, axis=0)

    loglik = float(_logsumexp(alpha[-1], axis=0))
    beta[-1] = 0.0
    for t in range(T - 2, -1, -1):
        beta[t] = _logsumexp(log_A + log_B[t + 1].reshape(1, 2) + beta[t + 1].reshape(1, 2), axis=1)

    gamma_log = alpha + beta
    gamma_log = gamma_log - _logsumexp(gamma_log, axis=1).reshape(-1, 1)
    gamma = np.exp(gamma_log)

    xi = np.zeros((max(T - 1, 0), 2, 2), dtype=float)
    for t in range(T - 1):
        m = alpha[t].reshape(2, 1) + log_A + log_B[t + 1].reshape(1, 2) + beta[t + 1].reshape(1, 2)
        m = m - _logsumexp(m, axis=None)
        xi[t] = np.exp(m)
    return loglik, gamma, xi


def _filtered_posterior(seq: SequenceData, pi: np.ndarray, A: np.ndarray, e: np.ndarray) -> np.ndarray:
    """
    Causal/online posterior p(z_t | y_1:t), no future-trial information.
    """
    y = seq.y
    T = len(y)
    B = _safe_prob(_emission_prob(y, e))
    log_pi = np.log(_safe_prob(pi))
    log_A = np.log(_safe_prob(A))
    log_B = np.log(B)

    alpha = np.zeros((T, 2), dtype=float)
    alpha[0] = log_pi + log_B[0]
    for t in range(1, T):
        alpha[t] = log_B[t] + _logsumexp(alpha[t - 1].reshape(2, 1) + log_A, axis=0)

    alpha_norm = alpha - _logsumexp(alpha, axis=1).reshape(-1, 1)
    return np.exp(alpha_norm)


def _viterbi_states(seq: SequenceData, pi: np.ndarray, A: np.ndarray, e: np.ndarray) -> np.ndarray:
    y = seq.y
    T = len(y)
    B = _safe_prob(_emission_prob(y, e))
    log_pi = np.log(_safe_prob(pi))
    log_A = np.log(_safe_prob(A))
    log_B = np.log(B)

    delta = np.zeros((T, 2), dtype=float)
    psi = np.zeros((T, 2), dtype=int)
    delta[0] = log_pi + log_B[0]
    for t in range(1, T):
        vals = delta[t - 1].reshape(2, 1) + log_A
        psi[t] = np.argmax(vals, axis=0)
        delta[t] = np.max(vals, axis=0) + log_B[t]

    z = np.zeros(T, dtype=int)
    z[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        z[t] = psi[t + 1, z[t + 1]]
    return z


def _enforce_state_order(pi: np.ndarray, A: np.ndarray, e: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # state 1 := acquired => higher correctness emission
    if e[1] >= e[0]:
        return pi, A, e
    perm = np.array([1, 0], dtype=int)
    return pi[perm], A[perm][:, perm], e[perm]


def _fit_hmm_em(seqs: List[SequenceData], rng: np.random.Generator) -> Dict[str, np.ndarray]:
    pi = np.array([0.85, 0.15], dtype=float)
    A = _normalize_rows(np.array([[0.88, 0.12], [0.08, 0.92]], dtype=float))
    e = np.array([rng.uniform(0.10, 0.40), rng.uniform(0.60, 0.95)], dtype=float)
    prev_ll = -np.inf
    best_ll = -np.inf

    for _ in range(MAX_ITER):
        pi_num = np.zeros(2, dtype=float)
        emit_num = np.zeros(2, dtype=float)
        emit_den = np.zeros(2, dtype=float)
        xi_num = np.zeros((2, 2), dtype=float)
        xi_den = np.zeros(2, dtype=float)
        total_ll = 0.0

        for seq in seqs:
            ll, gamma, xi = _forward_backward(seq, pi, A, e)
            total_ll += ll
            pi_num += gamma[0]
            emit_num += (gamma * seq.y.reshape(-1, 1)).sum(axis=0)
            emit_den += gamma.sum(axis=0)
            if len(seq.y) > 1:
                xi_num += xi.sum(axis=0)
                xi_den += gamma[:-1].sum(axis=0)

        pi = _safe_prob(pi_num / (pi_num.sum() + EPS))
        e = _safe_prob(emit_num / (emit_den + EPS))
        for i in range(2):
            if xi_den[i] > 0:
                A[i] = xi_num[i] / xi_den[i]
        A = _normalize_rows(_safe_prob(A))
        pi, A, e = _enforce_state_order(pi, A, e)

        best_ll = max(best_ll, total_ll)
        if np.isfinite(prev_ll) and abs(total_ll - prev_ll) < TOL:
            break
        prev_ll = total_ll
    return {"pi": pi, "A": A, "e": e, "loglik": best_ll}


def fit_unified_hmm(seqs: List[SequenceData], seed: int = RNG_SEED) -> Dict[str, np.ndarray]:
    if len(seqs) == 0:
        raise ValueError("No sequences.")
    master = np.random.default_rng(seed)
    best = None
    best_ll = -np.inf
    for _ in range(N_INIT):
        run_seed = int(master.integers(1, 1_000_000_000))
        params = _fit_hmm_em(seqs, np.random.default_rng(run_seed))
        if params["loglik"] > best_ll:
            best_ll = params["loglik"]
            best = params
    return best


def _first_true(mask: np.ndarray) -> Optional[int]:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    return int(idx[0])


def _first_run_start(mask: np.ndarray, min_run: int = 2) -> Optional[int]:
    run = max(int(min_run), 1)
    cnt = 0
    for i, v in enumerate(mask):
        cnt = cnt + 1 if bool(v) else 0
        if cnt >= run:
            return int(i - run + 1)
    return None


def _mode_or_nan(s: pd.Series):
    s2 = s.dropna().astype(str)
    if s2.empty:
        return np.nan
    return s2.mode().iloc[0]


def _build_block_context(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["block_num"] = pd.to_numeric(d["block_id"], errors="coerce")
    d["trial_num"] = pd.to_numeric(d["trial_id"], errors="coerce")
    d = d.sort_values(["subj", "block_num", "trial_num"])

    rows = (
        d.groupby(["subj", "block_id"], as_index=False)
        .agg(
            block_num=("block_num", "first"),
            rule_level=("rule_level", _mode_or_nan),
            shift_type_raw=("rule_level", _mode_or_nan),
        )
    )

    # set index: split each subject's ordered blocks into 8 sets
    rows["set_index"] = np.nan
    rows["shift_type"] = pd.Series(["unknown"] * len(rows), dtype="object")
    for subj, sdf in rows.groupby("subj", sort=False):
        idx = sdf.sort_values("block_num").index.to_list()
        bids = rows.loc[idx, "block_id"].astype(str).to_numpy()
        block_to_set = {}
        for si, chunk in enumerate(np.array_split(bids, 8), start=1):
            for bid in chunk:
                block_to_set[str(bid)] = float(si)
        rows.loc[idx, "set_index"] = rows.loc[idx, "block_id"].astype(str).map(block_to_set).astype(float)

        # shift_type from consecutive block rule-level changes
        prev_level = None
        for ridx in idx:
            cur_level = rows.at[ridx, "rule_level"]
            if isinstance(cur_level, float) and np.isnan(cur_level):
                rows.at[ridx, "shift_type"] = "unknown"
            elif prev_level is None:
                rows.at[ridx, "shift_type"] = "first_block"
            else:
                rows.at[ridx, "shift_type"] = "within_level" if str(cur_level) == str(prev_level) else "cross_level"
            prev_level = cur_level

    rows["set_bin"] = np.where(rows["set_index"] <= 4, "early_sets1_4", "late_sets5_8")
    return rows[["subj", "block_id", "block_num", "set_index", "set_bin", "rule_level", "shift_type"]].copy()


def _logit(p: np.ndarray) -> np.ndarray:
    p2 = np.clip(p.astype(float), 1e-4, 1 - 1e-4)
    return np.log(p2 / (1 - p2))


def _run_mixed_models(block_df: pd.DataFrame, trial_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    if smf is None:
        return pd.DataFrame(), "statsmodels not available; mixed models were skipped."

    lines: List[str] = []
    model_rows: List[Dict[str, object]] = []

    def _fit_one(dep: str):
        d = block_df[[dep, "set_index", "shift_type", "rule_level", "subj"]].dropna().copy()
        d = d[d["shift_type"] != "first_block"].copy()
        if len(d) < 50:
            lines.append(f"[skip] {dep}: too few rows ({len(d)})")
            return
        try:
            m = smf.mixedlm(f"{dep} ~ set_index + C(shift_type) + C(rule_level)", d, groups=d["subj"], re_formula="~set_index")
            r = m.fit(method="lbfgs", reml=False, maxiter=200, disp=False)
            lines.append(f"[ok] {dep}: converged={r.converged}, n={len(d)}")
            for k in r.params.index:
                model_rows.append(
                    {
                        "model": dep,
                        "term": k,
                        "coef": float(r.params[k]),
                        "pvalue": float(r.pvalues.get(k, np.nan)),
                    }
                )
        except Exception as e:
            lines.append(f"[fail] {dep}: {str(e)}")

    _fit_one("acquisition_lag_core")
    _fit_one("transition_width_20_to_80")
    _fit_one("collapse_to_acquisition_lag")

    # Optional aligned trial-level slope model
    td = trial_df[["logit_p_acq", "rel_trial_fc", "set_bin", "shift_type", "rule_level", "subj"]].dropna().copy()
    td = td[(td["rel_trial_fc"] >= -10) & (td["rel_trial_fc"] <= 10)].copy()
    td = td[td["shift_type"] != "first_block"].copy()
    if len(td) >= 300:
        try:
            m = smf.mixedlm(
                "logit_p_acq ~ rel_trial_fc * C(set_bin) + C(shift_type) + C(rule_level)",
                td,
                groups=td["subj"],
                re_formula="~rel_trial_fc",
            )
            r = m.fit(method="lbfgs", reml=False, maxiter=200, disp=False)
            lines.append(f"[ok] trial_level: converged={r.converged}, n={len(td)}")
            for k in r.params.index:
                model_rows.append(
                    {
                        "model": "logit_p_acq_rel_trial",
                        "term": k,
                        "coef": float(r.params[k]),
                        "pvalue": float(r.pvalues.get(k, np.nan)),
                    }
                )
        except Exception as e:
            lines.append(f"[fail] trial_level: {str(e)}")
    else:
        lines.append(f"[skip] trial_level: too few rows ({len(td)})")

    return pd.DataFrame(model_rows), "\n".join(lines)


def _first_hmm_acquired_trial_id(tdf: pd.DataFrame) -> float:
    """First trial_id in block where Viterbi state is acquired (1); NaN if never acquired."""
    hit = tdf["viterbi_state"].astype(int).to_numpy() == 1
    if not np.any(hit):
        return np.nan
    row0 = tdf.loc[hit].iloc[0]
    return float(pd.to_numeric(row0["trial_id"], errors="coerce"))


def _write_trialwise_phase_csvs(raw: pd.DataFrame, trial_df: pd.DataFrame) -> None:
    """Merge HMM-defined phase into behavior trialwise tables (all_subjects + per-subject for MATLAB)."""
    phase_map = trial_df[["subj", "block_id", "trial_id", "phase"]].copy()
    phase_map["subj"] = phase_map["subj"].astype(str)
    phase_map["block_id"] = phase_map["block_id"].astype(str)
    phase_map["trial_id"] = phase_map["trial_id"].astype(str)

    raw_w = raw.copy()
    raw_w["subj"] = raw_w["subj"].astype(str)
    raw_w["block_id"] = raw_w["block_id"].astype(str)
    raw_w["trial_id"] = raw_w["trial_id"].astype(str)
    if "phase" in raw_w.columns:
        raw_w = raw_w.drop(columns=["phase"])
    raw_w = raw_w.merge(phase_map, on=["subj", "block_id", "trial_id"], how="left")

    raw_w.to_csv(IN_CSV, index=False)

    os.makedirs(TRIALWISE_PER_SUBJ_DIR, exist_ok=True)
    sort_bn = pd.to_numeric(raw_w["block_id"], errors="coerce")
    sort_tn = pd.to_numeric(raw_w["trial_id"], errors="coerce")
    raw_w = raw_w.assign(_bn=sort_bn, _tn=sort_tn).sort_values(["subj", "_bn", "_tn"]).drop(columns=["_bn", "_tn"])
    for subj, sdf in raw_w.groupby("subj", sort=False):
        out_p = os.path.join(TRIALWISE_PER_SUBJ_DIR, f"{subj}_trialwise.csv")
        sdf.to_csv(out_p, index=False)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw = pd.read_csv(IN_CSV)
    seqs = _prepare_sequences(raw)
    if len(seqs) == 0:
        raise RuntimeError("No usable sequences in input.")

    params = fit_unified_hmm(seqs, seed=RNG_SEED)
    pi, A, e = params["pi"], params["A"], params["e"]

    block_context = _build_block_context(raw)
    block_context["block_id"] = block_context["block_id"].astype(str)

    trial_rows: List[Dict[str, object]] = []
    block_rows: List[Dict[str, object]] = []
    for seq in seqs:
        _, gamma, _ = _forward_backward(seq, pi, A, e)
        filt = _filtered_posterior(seq, pi, A, e)
        viterbi = _viterbi_states(seq, pi, A, e)
        p_acq = gamma[:, 1]
        p_acq_f = filt[:, 1]

        fc_idx = _first_true(seq.y == 1)
        acq_core_idx = _first_run_start(p_acq >= ACQ_THRESH, min_run=ACQ_MIN_RUN)
        acq_filt_idx = _first_run_start(p_acq_f >= ACQ_THRESH, min_run=ACQ_MIN_RUN)
        acq_vit_idx = _first_true(viterbi == 1)
        t20 = _first_true(p_acq >= LOW_THRESH)
        t80 = _first_true(p_acq >= HIGH_THRESH)

        acq_trial_core = (acq_core_idx + 1) if acq_core_idx is not None else np.nan
        acq_trial_filtered = (acq_filt_idx + 1) if acq_filt_idx is not None else np.nan
        acq_trial_viterbi = (acq_vit_idx + 1) if acq_vit_idx is not None else np.nan
        first_correct_trial = (fc_idx + 1) if fc_idx is not None else np.nan

        acq_lag = (
            float(acq_trial_core - first_correct_trial)
            if pd.notna(acq_trial_core) and pd.notna(first_correct_trial)
            else np.nan
        )
        acq_lag_filtered = (
            float(acq_trial_filtered - first_correct_trial)
            if pd.notna(acq_trial_filtered) and pd.notna(first_correct_trial)
            else np.nan
        )
        transition_width = (
            float((t80 + 1) - (t20 + 1))
            if (t20 is not None) and (t80 is not None) and (t80 >= t20)
            else np.nan
        )

        # Save trial rows first; rel indices will be filled after block merge.
        for ti in range(len(seq.y)):
            trial_rows.append(
                {
                    "subj": seq.subj,
                    "block_id": seq.block_id,
                    "trial_id": seq.trial_ids[ti],
                    "trial_index_1based": float(ti + 1),
                    "correctness": int(seq.y[ti]),
                    "p_search": float(gamma[ti, 0]),
                    "p_acq": float(gamma[ti, 1]),
                    "p_search_filtered": float(filt[ti, 0]),
                    "p_acq_filtered": float(filt[ti, 1]),
                    "viterbi_state": int(viterbi[ti]),
                }
            )

        block_rows.append(
            {
                "subj": seq.subj,
                "block_id": seq.block_id,
                "n_trials": int(len(seq.y)),
                "first_correct_trial": first_correct_trial,
                "acquisition_trial_core": acq_trial_core,
                "acquisition_trial_filtered": acq_trial_filtered,
                "acquisition_trial_viterbi": acq_trial_viterbi,
                "acquisition_lag_core": acq_lag,
                "acquisition_lag_filtered": acq_lag_filtered,
                "transition_width_20_to_80": transition_width,
            }
        )

    trial_df = pd.DataFrame(trial_rows)
    block_df = pd.DataFrame(block_rows)
    trial_df["block_id"] = trial_df["block_id"].astype(str)
    block_df["block_id"] = block_df["block_id"].astype(str)

    # Phase labels follow the unified 2-state HMM Viterbi path only (z0=search, z1=acquired).
    trial_df["phase"] = np.where(trial_df["viterbi_state"].astype(int).to_numpy() == 0, "search", "acquired")

    collapse_rows: List[Dict[str, object]] = []
    for (sj, bid), g in trial_df.groupby(["subj", "block_id"], sort=False):
        g2 = g.sort_values("trial_index_1based")
        collapse_rows.append(
            {
                "subj": sj,
                "block_id": str(bid),
                "collapse_trial_proxy": _first_hmm_acquired_trial_id(g2),
            }
        )
    collapse_lookup = pd.DataFrame(collapse_rows)

    block_df = (
        block_df.merge(block_context, on=["subj", "block_id"], how="left")
        .merge(collapse_lookup, on=["subj", "block_id"], how="left")
    )
    block_df["collapse_to_acquisition_lag_proxy"] = np.where(
        block_df["collapse_trial_proxy"].notna() & block_df["acquisition_trial_core"].notna(),
        block_df["acquisition_trial_core"] - block_df["collapse_trial_proxy"],
        np.nan,
    )
    # Backward-compatible alias (same value).
    block_df["collapse_to_acquisition_lag"] = block_df["collapse_to_acquisition_lag_proxy"]

    trial_df = trial_df.merge(
        block_df[
            [
                "subj",
                "block_id",
                "first_correct_trial",
                "acquisition_trial_core",
                "set_index",
                "set_bin",
                "rule_level",
                "shift_type",
            ]
        ],
        on=["subj", "block_id"],
        how="left",
    )
    trial_df["rel_trial_fc"] = trial_df["trial_index_1based"] - trial_df["first_correct_trial"]
    trial_df["rel_trial_acq"] = trial_df["trial_index_1based"] - trial_df["acquisition_trial_core"]
    trial_df["logit_p_acq"] = _logit(trial_df["p_acq"].to_numpy())

    _write_trialwise_phase_csvs(raw, trial_df)

    # Subject summary
    subj_df = (
        block_df.groupby("subj", as_index=False)
        .agg(
            n_blocks=("block_id", "count"),
            acquisition_lag_core_median=("acquisition_lag_core", "median"),
            acquisition_lag_filtered_median=("acquisition_lag_filtered", "median"),
            transition_width_median=("transition_width_20_to_80", "median"),
            collapse_to_acq_lag_proxy_median=("collapse_to_acquisition_lag_proxy", "median"),
        )
    )

    model_df = pd.DataFrame(
        [
            {
                "pi_search": float(pi[0]),
                "pi_acquired": float(pi[1]),
                "A_search_to_search": float(A[0, 0]),
                "A_search_to_acquired": float(A[0, 1]),
                "A_acquired_to_search": float(A[1, 0]),
                "A_acquired_to_acquired": float(A[1, 1]),
                "emit_correct_if_search": float(e[0]),
                "emit_correct_if_acquired": float(e[1]),
                "total_loglik": float(params["loglik"]),
                "n_sequences": int(len(seqs)),
                "n_trials_total": int(sum(len(s.y) for s in seqs)),
            }
        ]
    )

    mixed_df, mixed_log = _run_mixed_models(block_df, trial_df)

    trial_path = os.path.join(OUT_DIR, "hmm_unified_trial_posteriors.csv")
    block_path = os.path.join(OUT_DIR, "hmm_unified_block_metrics.csv")
    subj_path = os.path.join(OUT_DIR, "hmm_unified_subject_metrics.csv")
    model_path = os.path.join(OUT_DIR, "hmm_unified_model_params.csv")
    mixed_path = os.path.join(OUT_DIR, "hmm_unified_mixedlm_coeffs.csv")
    log_path = os.path.join(OUT_DIR, "hmm_unified_mixedlm_log.txt")
    defs_path = os.path.join(OUT_DIR, "hmm_unified_metric_definitions.json")

    trial_df.to_csv(trial_path, index=False)
    block_df.to_csv(block_path, index=False)
    subj_df.to_csv(subj_path, index=False)
    model_df.to_csv(model_path, index=False)
    if not mixed_df.empty:
        mixed_df.to_csv(mixed_path, index=False)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(mixed_log + "\n")

    defs = {
        "modeling_principle": "One unified two-state HMM is fit across all blocks; sets are compared downstream at block level.",
        "state_definition": {"z0": "search state", "z1": "acquired state"},
        "phase_trial": "Per-trial label from unified HMM Viterbi decoding only: viterbi_state 0 -> 'search', 1 -> 'acquired'. Written into all_subjects_trialwise.csv and trials_trialwise/trialwise/*_trialwise.csv after this script runs.",
        "acquisition_trial_core": "retrospectively inferred acquired-state onset: first trial starting >=2 consecutive trials with smoothed posterior p_acq > 0.8",
        "acquisition_trial_filtered": "online/causal version: first trial starting >=2 consecutive trials with filtered posterior p_acq_filtered > 0.8",
        "acquisition_trial_viterbi": "first trial where Viterbi path enters acquired state",
        "acquisition_lag_core": "acquisition_trial_core - first_correct_trial",
        "acquisition_lag_filtered": "acquisition_trial_filtered - first_correct_trial",
        "transition_width_20_to_80": "first(p_acq >= 0.8) - first(p_acq >= 0.2)",
        "collapse_trial_proxy": "First trial_id in the HMM sequence where Viterbi enters acquired (state 1); equals acquisition_trial_viterbi when a block reaches acquired at least once.",
        "collapse_to_acquisition_lag_proxy": "acquisition_trial_core - collapse_trial_proxy (difference between posterior-run acquisition onset and Viterbi first-acquired trial)",
        "aligned_trial_model_note": "trial-level model uses logit(p_acq) with rel_trial alignment to first_correct",
    }
    with open(defs_path, "w", encoding="utf-8") as f:
        json.dump(defs, f, ensure_ascii=True, indent=2)

    print(f"saved: {trial_path}")
    print(f"saved: {block_path}")
    print(f"saved: {subj_path}")
    print(f"saved: {model_path}")
    if not mixed_df.empty:
        print(f"saved: {mixed_path}")
    print(f"saved: {log_path}")
    print(f"saved: {defs_path}")
    print(f"updated HMM phase in {IN_CSV} and per-subject CSVs under {TRIALWISE_PER_SUBJ_DIR}")


if __name__ == "__main__":
    main()
