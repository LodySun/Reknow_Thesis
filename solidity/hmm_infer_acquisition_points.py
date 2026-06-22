import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


BASE = "base_dir"
IN_CSV = os.path.join(BASE, "trials_trialwise", "all_subjects_trialwise.csv")
OUT_DIR = os.path.join(BASE, "trials_trialwise", "1s_comp", "eeg_paper_results", "solidity")

EPS = 1e-9
MAX_ITER = 60
TOL = 1e-5
N_INIT = 3
RNG_SEED = 20260404


@dataclass
class SequenceData:
    subj: str
    block_id: str
    trial_ids: np.ndarray  # original trial_id labels
    y: np.ndarray  # binary correctness


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
        y = b["correctness"].astype(int).to_numpy()
        trial_ids = b["trial_id"].astype(str).to_numpy()
        seqs.append(
            SequenceData(
                subj=str(subj),
                block_id=str(block_id),
                trial_ids=trial_ids,
                y=y,
            )
        )
    return seqs


def _emission_prob(y: np.ndarray, e: np.ndarray) -> np.ndarray:
    # P(y_t | z_t=s), shape (T, 2)
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
    # state 1 := acquired, require higher correctness probability
    if e[1] >= e[0]:
        return pi, A, e
    perm = np.array([1, 0], dtype=int)
    pi2 = pi[perm]
    A2 = A[perm][:, perm]
    e2 = e[perm]
    return pi2, A2, e2


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


def fit_with_restarts(seqs: List[SequenceData], seed: int = RNG_SEED) -> Dict[str, np.ndarray]:
    if len(seqs) == 0:
        raise ValueError("No valid sequences found.")
    master = np.random.default_rng(seed)
    best = None
    best_ll = -np.inf
    for _ in range(N_INIT):
        run_seed = int(master.integers(1, 1_000_000_000))
        params = _fit_hmm_em(seqs, np.random.default_rng(run_seed))
        if params["loglik"] > best_ll:
            best = params
            best_ll = params["loglik"]
    return best


def _first_true(mask: np.ndarray) -> Optional[int]:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    return int(idx[0])


def _first_run_start(mask: np.ndarray, min_run: int) -> Optional[int]:
    run = max(int(min_run), 1)
    count = 0
    for i, v in enumerate(mask):
        count = count + 1 if bool(v) else 0
        if count >= run:
            return int(i - run + 1)
    return None


def _acq_defs(gamma: np.ndarray, viterbi: np.ndarray) -> Dict[str, Optional[int]]:
    p_acq = gamma[:, 1]
    ap_post_05 = _first_true(p_acq >= 0.50)
    ap_post_07_run2 = _first_run_start(p_acq >= 0.70, min_run=2)

    first_1 = _first_true(viterbi == 1)
    enter_01 = None
    if first_1 is not None:
        if first_1 == 0:
            enter_01 = 0
        else:
            prev0 = np.where(viterbi[:first_1] == 0)[0]
            enter_01 = int(first_1) if len(prev0) > 0 else int(first_1)

    return {
        "ap_post50_first_idx0": ap_post_05,
        "ap_post70_run2_idx0": ap_post_07_run2,
        "ap_viterbi_enter_idx0": enter_01,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(IN_CSV)
    seqs = _prepare_sequences(df)
    if len(seqs) == 0:
        raise RuntimeError("No usable sequences from input file.")

    params = fit_with_restarts(seqs, seed=RNG_SEED)
    pi, A, e = params["pi"], params["A"], params["e"]

    trial_rows: List[Dict[str, object]] = []
    block_rows: List[Dict[str, object]] = []
    for seq in seqs:
        _, gamma, _ = _forward_backward(seq, pi, A, e)
        vit = _viterbi_states(seq, pi, A, e)
        acq = _acq_defs(gamma, vit)

        for t in range(len(seq.y)):
            trial_rows.append(
                {
                    "subj": seq.subj,
                    "block_id": seq.block_id,
                    "trial_id": seq.trial_ids[t],
                    "trial_index_1based": int(t + 1),
                    "correctness": int(seq.y[t]),
                    "p_search": float(gamma[t, 0]),
                    "p_acquired": float(gamma[t, 1]),
                    "viterbi_state": int(vit[t]),
                }
            )

        block_rows.append(
            {
                "subj": seq.subj,
                "block_id": seq.block_id,
                "n_trials": int(len(seq.y)),
                "ap_post50_first_trial": (acq["ap_post50_first_idx0"] + 1) if acq["ap_post50_first_idx0"] is not None else np.nan,
                "ap_post70_run2_trial": (acq["ap_post70_run2_idx0"] + 1) if acq["ap_post70_run2_idx0"] is not None else np.nan,
                "ap_viterbi_enter_trial": (acq["ap_viterbi_enter_idx0"] + 1) if acq["ap_viterbi_enter_idx0"] is not None else np.nan,
            }
        )

    trial_df = pd.DataFrame(trial_rows)
    block_df = pd.DataFrame(block_rows)
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

    # Optional quick sanity summary at subject level.
    subj_df = (
        block_df.groupby("subj", as_index=False)
        .agg(
            n_blocks=("block_id", "count"),
            ap_post50_median=("ap_post50_first_trial", "median"),
            ap_post70_run2_median=("ap_post70_run2_trial", "median"),
            ap_viterbi_median=("ap_viterbi_enter_trial", "median"),
        )
    )

    trial_path = os.path.join(OUT_DIR, "hmm_trial_posteriors.csv")
    block_path = os.path.join(OUT_DIR, "hmm_acquisition_points_by_block.csv")
    subj_path = os.path.join(OUT_DIR, "hmm_acquisition_points_by_subject.csv")
    model_path = os.path.join(OUT_DIR, "hmm_two_state_model_params.csv")
    meta_path = os.path.join(OUT_DIR, "hmm_acquisition_definitions.json")

    trial_df.to_csv(trial_path, index=False)
    block_df.to_csv(block_path, index=False)
    subj_df.to_csv(subj_path, index=False)
    model_df.to_csv(model_path, index=False)

    defs = {
        "ap_post50_first_trial": "first trial where posterior P(z=acquired|y_1:T) >= 0.50",
        "ap_post70_run2_trial": "first trial starting a run of >=2 consecutive trials with posterior >= 0.70",
        "ap_viterbi_enter_trial": "first trial where Viterbi path enters state z=1 (acquired)",
        "note": "No external hard threshold on number of correct trials is used to force transition.",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(defs, f, ensure_ascii=True, indent=2)

    print(f"saved: {trial_path}")
    print(f"saved: {block_path}")
    print(f"saved: {subj_path}")
    print(f"saved: {model_path}")
    print(f"saved: {meta_path}")


if __name__ == "__main__":
    main()

