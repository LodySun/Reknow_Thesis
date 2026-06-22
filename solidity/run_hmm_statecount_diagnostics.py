import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


BASE = "base_dir"
IN_CSV = os.path.join(BASE, "trials_trialwise", "all_subjects_trialwise.csv")
OUT_DIR = os.path.join(BASE, "trials_trialwise", "1s_comp", "eeg_paper_results", "solidity")

OUT_MODEL_CMP = os.path.join(OUT_DIR, "hmm_statecount_model_comparison.csv")
OUT_BEST_PARAMS = os.path.join(OUT_DIR, "hmm_statecount_best_params_long.csv")
OUT_RUN_DIAG = os.path.join(OUT_DIR, "hmm_statecount_restart_diagnostics.csv")

EPS = 1e-9
MAX_ITER = 60
TOL = 1e-5
N_RESTART = 4
SEED = 20260425
TRAIN_FRAC = 0.8


@dataclass
class Seq:
    subj: str
    block_id: str
    y: np.ndarray


def _safe_prob(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 1e-6, 1 - 1e-6)


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    m = mat.copy().astype(float)
    rs = m.sum(axis=1, keepdims=True)
    rs[rs <= 0] = 1.0
    return m / rs


def _logsumexp(a: np.ndarray, axis=None) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True) + EPS)
    if axis is None:
        return out.reshape(())
    return np.squeeze(out, axis=axis)


def _prepare_sequences(df: pd.DataFrame) -> List[Seq]:
    d = df[["subj", "block_id", "trial_id", "correctness"]].copy()
    d["correctness"] = pd.to_numeric(d["correctness"], errors="coerce")
    d["trial_num"] = pd.to_numeric(d["trial_id"], errors="coerce")
    d["block_num"] = pd.to_numeric(d["block_id"], errors="coerce")
    d = d.sort_values(["subj", "block_num", "trial_num"])
    out: List[Seq] = []
    for (subj, bid), g in d.groupby(["subj", "block_id"], sort=False):
        gg = g[g["correctness"].isin([0, 1])].copy()
        if gg.empty:
            continue
        out.append(Seq(subj=str(subj), block_id=str(bid), y=gg["correctness"].astype(int).to_numpy()))
    return out


def _emission_prob(y: np.ndarray, e: np.ndarray) -> np.ndarray:
    yv = y.astype(float).reshape(-1, 1)
    p1 = e.reshape(1, -1)
    return yv * p1 + (1.0 - yv) * (1.0 - p1)


def _forward_backward(y: np.ndarray, pi: np.ndarray, A: np.ndarray, e: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    k = len(pi)
    T = len(y)
    B = _safe_prob(_emission_prob(y, e))
    log_pi = np.log(_safe_prob(pi))
    log_A = np.log(_safe_prob(A))
    log_B = np.log(B)

    alpha = np.zeros((T, k), dtype=float)
    beta = np.zeros((T, k), dtype=float)
    alpha[0] = log_pi + log_B[0]
    for t in range(1, T):
        alpha[t] = log_B[t] + _logsumexp(alpha[t - 1].reshape(k, 1) + log_A, axis=0)

    ll = float(_logsumexp(alpha[-1], axis=0))
    beta[-1] = 0.0
    for t in range(T - 2, -1, -1):
        beta[t] = _logsumexp(log_A + log_B[t + 1].reshape(1, k) + beta[t + 1].reshape(1, k), axis=1)

    gamma_log = alpha + beta
    gamma_log = gamma_log - _logsumexp(gamma_log, axis=1).reshape(-1, 1)
    gamma = np.exp(gamma_log)

    xi = np.zeros((max(T - 1, 0), k, k), dtype=float)
    for t in range(T - 1):
        x = alpha[t].reshape(k, 1) + log_A + log_B[t + 1].reshape(1, k) + beta[t + 1].reshape(1, k)
        x = x - _logsumexp(x, axis=None)
        xi[t] = np.exp(x)
    return ll, gamma, xi


def _order_states(pi: np.ndarray, A: np.ndarray, e: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(e)
    return pi[order], A[order][:, order], e[order]


def _init_params(k: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if k == 1:
        return np.array([1.0]), np.array([[1.0]]), np.array([rng.uniform(0.2, 0.8)])
    pi = _safe_prob(rng.dirichlet(np.ones(k)))
    A = _normalize_rows(_safe_prob(rng.dirichlet(np.ones(k), size=k)))
    e = np.sort(rng.uniform(0.05, 0.95, size=k))
    return pi, A, e


def _fit_em(seqs: List[Seq], k: int, rng: np.random.Generator) -> Dict[str, object]:
    pi, A, e = _init_params(k, rng)
    prev_ll = -np.inf
    converged = False
    iters = 0
    ll_trace: List[float] = []

    for it in range(MAX_ITER):
        iters = it + 1
        pi_num = np.zeros(k, dtype=float)
        emit_num = np.zeros(k, dtype=float)
        emit_den = np.zeros(k, dtype=float)
        xi_num = np.zeros((k, k), dtype=float)
        xi_den = np.zeros(k, dtype=float)
        total_ll = 0.0

        for s in seqs:
            ll, gamma, xi = _forward_backward(s.y, pi, A, e)
            total_ll += ll
            pi_num += gamma[0]
            emit_num += (gamma * s.y.reshape(-1, 1)).sum(axis=0)
            emit_den += gamma.sum(axis=0)
            if len(s.y) > 1:
                xi_num += xi.sum(axis=0)
                xi_den += gamma[:-1].sum(axis=0)

        pi = _safe_prob(pi_num / (pi_num.sum() + EPS))
        e = _safe_prob(emit_num / (emit_den + EPS))
        if k > 1:
            for i in range(k):
                if xi_den[i] > 0:
                    A[i] = xi_num[i] / (xi_den[i] + EPS)
            A = _normalize_rows(_safe_prob(A))
        pi, A, e = _order_states(pi, A, e)

        ll_trace.append(total_ll)
        if np.isfinite(prev_ll) and abs(total_ll - prev_ll) < TOL:
            converged = True
            break
        prev_ll = total_ll

    return {
        "pi": pi,
        "A": A,
        "e": e,
        "loglik": float(ll_trace[-1]) if ll_trace else -np.inf,
        "iters": int(iters),
        "converged": int(converged),
        "ll_trace_last3": ll_trace[-3:] if len(ll_trace) >= 3 else ll_trace,
    }


def _dataset_loglik(seqs: List[Seq], pi: np.ndarray, A: np.ndarray, e: np.ndarray) -> float:
    s = 0.0
    for seq in seqs:
        ll, _, _ = _forward_backward(seq.y, pi, A, e)
        s += ll
    return float(s)


def _n_params(k: int) -> int:
    # pi: k-1, transition: k*(k-1), emission Bernoulli: k
    return (k - 1) + k * (k - 1) + k


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw = pd.read_csv(IN_CSV)
    seqs = _prepare_sequences(raw)
    if not seqs:
        raise RuntimeError("No usable sequences.")

    master = np.random.default_rng(SEED)
    n = len(seqs)
    idx = np.arange(n)
    master.shuffle(idx)
    cut = int(np.floor(TRAIN_FRAC * n))
    tr_idx = set(idx[:cut].tolist())
    te_idx = set(idx[cut:].tolist())
    train = [seqs[i] for i in range(n) if i in tr_idx]
    test = [seqs[i] for i in range(n) if i in te_idx]

    run_rows = []
    cmp_rows = []
    param_rows = []

    for k in [1, 2, 3]:
        fits = []
        for r in range(N_RESTART):
            seed = int(master.integers(1, 1_000_000_000))
            fit = _fit_em(train, k=k, rng=np.random.default_rng(seed))
            fit["restart_id"] = r + 1
            fit["seed"] = seed
            fits.append(fit)
            run_rows.append(
                {
                    "k_states": k,
                    "restart_id": r + 1,
                    "seed": seed,
                    "train_loglik": fit["loglik"],
                    "n_iters": fit["iters"],
                    "converged": fit["converged"],
                    "ll_trace_last3": "|".join([f"{x:.6f}" for x in fit["ll_trace_last3"]]),
                }
            )

        best = sorted(fits, key=lambda x: x["loglik"], reverse=True)[0]
        pi, A, e = best["pi"], best["A"], best["e"]

        train_ll = float(best["loglik"])
        test_ll = _dataset_loglik(test, pi, A, e)
        all_ll = _dataset_loglik(seqs, pi, A, e)
        p = _n_params(k)
        n_obs_train = int(sum(len(s.y) for s in train))
        aic = 2 * p - 2 * train_ll
        bic = np.log(max(n_obs_train, 1)) * p - 2 * train_ll

        cmp_rows.append(
            {
                "k_states": k,
                "n_train_blocks": len(train),
                "n_test_blocks": len(test),
                "n_train_trials": n_obs_train,
                "n_test_trials": int(sum(len(s.y) for s in test)),
                "n_params": p,
                "best_train_loglik": train_ll,
                "test_loglik": test_ll,
                "all_loglik_refit_eval": all_ll,
                "aic_train": aic,
                "bic_train": bic,
                "restart_loglik_mean": float(np.mean([f["loglik"] for f in fits])),
                "restart_loglik_sd": float(np.std([f["loglik"] for f in fits], ddof=1)) if len(fits) > 1 else 0.0,
                "converged_restarts": int(np.sum([f["converged"] for f in fits])),
                "total_restarts": N_RESTART,
                "best_iters": int(best["iters"]),
                "best_converged": int(best["converged"]),
            }
        )

        for i, v in enumerate(pi):
            param_rows.append({"k_states": k, "param_group": "initial_pi", "param_name": f"pi_state{i}", "value": float(v)})
        for i, v in enumerate(e):
            param_rows.append({"k_states": k, "param_group": "emission", "param_name": f"emit_correct_if_state{i}", "value": float(v)})
        for i in range(k):
            for j in range(k):
                param_rows.append(
                    {
                        "k_states": k,
                        "param_group": "transition_A",
                        "param_name": f"A_state{i}_to_state{j}",
                        "value": float(A[i, j]),
                    }
                )

    pd.DataFrame(run_rows).to_csv(OUT_RUN_DIAG, index=False)
    pd.DataFrame(cmp_rows).sort_values("k_states").to_csv(OUT_MODEL_CMP, index=False)
    pd.DataFrame(param_rows).sort_values(["k_states", "param_group", "param_name"]).to_csv(OUT_BEST_PARAMS, index=False)

    print(f"saved: {OUT_RUN_DIAG}")
    print(f"saved: {OUT_MODEL_CMP}")
    print(f"saved: {OUT_BEST_PARAMS}")
    print(pd.DataFrame(cmp_rows).sort_values('k_states').to_string(index=False))


if __name__ == "__main__":
    main()
