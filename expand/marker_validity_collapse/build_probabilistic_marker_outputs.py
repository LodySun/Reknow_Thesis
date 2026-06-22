import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon


BASE = "base_dir"
IN_BLOCK = os.path.join(BASE, "collapse_methods_block_level.csv")
IN_MODELS = os.path.join(BASE, "marker_independence_models.csv")

OUT_DISS = os.path.join(BASE, "expand1_certainty_commitment_dissociation_probabilistic.csv")
OUT_ORDER = os.path.join(BASE, "expand1_event_order_proportions_probabilistic.csv")
OUT_MODELS = os.path.join(BASE, "expand2_marker_independence_models_probabilistic.csv")
OUT_FIG = os.path.join(BASE, "figure_expand_transition_core_claims_probabilistic.png")


def _summarize_vector(name: str, x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    return {
        "metric": name,
        "n_blocks": int(len(x)),
        "mean": float(np.mean(x)) if len(x) else np.nan,
        "median": float(np.median(x)) if len(x) else np.nan,
        "sd": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
        "prop_gt_0": float(np.mean(x > 0)) if len(x) else np.nan,
    }


def _safe_p_to_sig(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def main():
    d = pd.read_csv(IN_BLOCK)
    m = pd.read_csv(IN_MODELS)
    for c in [
        "first_correct_trial",
        "acquisition_trial_core",
        "collapse_hard",
        "collapse_prob_tau08",
        "collapse_prob_tau09",
    ]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Primary revised collapse marker: probabilistic tau=0.8
    d = d.dropna(subset=["first_correct_trial", "acquisition_trial_core", "collapse_prob_tau08"]).copy()
    d["lag_collapse_to_fc_prob08"] = d["first_correct_trial"] - d["collapse_prob_tau08"]
    d["lag_fc_to_core"] = d["acquisition_trial_core"] - d["first_correct_trial"]
    d["lag_collapse_to_core_prob08"] = d["acquisition_trial_core"] - d["collapse_prob_tau08"]

    # Sensitivity reference: original hard collapse
    d["lag_collapse_to_fc_hard"] = d["first_correct_trial"] - d["collapse_hard"]
    d["lag_collapse_to_core_hard"] = d["acquisition_trial_core"] - d["collapse_hard"]

    # Summary table
    rows = []
    rows.append(_summarize_vector("lag_collapse_to_fc_prob08", d["lag_collapse_to_fc_prob08"].to_numpy(dtype=float)))
    rows.append(_summarize_vector("lag_fc_to_core", d["lag_fc_to_core"].to_numpy(dtype=float)))
    rows.append(_summarize_vector("lag_collapse_to_core_prob08", d["lag_collapse_to_core_prob08"].to_numpy(dtype=float)))
    rows.append(_summarize_vector("lag_collapse_to_fc_hard", d["lag_collapse_to_fc_hard"].to_numpy(dtype=float)))
    rows.append(_summarize_vector("lag_collapse_to_core_hard", d["lag_collapse_to_core_hard"].to_numpy(dtype=float)))

    x = d["lag_collapse_to_fc_prob08"].to_numpy(dtype=float)
    y = d["lag_fc_to_core"].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() >= 10:
        tt, pp = ttest_rel(y[ok], x[ok], nan_policy="omit")
        try:
            ww, pww = wilcoxon(y[ok], x[ok], zero_method="wilcox")
        except Exception:
            ww, pww = np.nan, np.nan
        rows.append(
            {
                "metric": "lag_fc_to_core_vs_lag_collapse_to_fc_prob08",
                "n_blocks": int(ok.sum()),
                "mean": float(np.mean(y[ok] - x[ok])),
                "median": float(np.median(y[ok] - x[ok])),
                "sd": float(np.std(y[ok] - x[ok], ddof=1)),
                "prop_gt_0": float(np.mean((y[ok] - x[ok]) > 0)),
                "test": "paired_t",
                "t_stat": float(tt),
                "p_ttest": float(pp),
                "w_stat": float(ww) if np.isfinite(ww) else np.nan,
                "p_wilcoxon": float(pww) if np.isfinite(pww) else np.nan,
            }
        )
    out_diss = pd.DataFrame(rows)
    out_diss.to_csv(OUT_DISS, index=False)

    # Event-order proportions
    d["order_collapse_prob_vs_fc"] = np.where(
        d["collapse_prob_tau08"] < d["first_correct_trial"],
        "collapse_before_fc",
        np.where(
            d["collapse_prob_tau08"] == d["first_correct_trial"],
            "collapse_eq_fc",
            "collapse_after_fc",
        ),
    )
    d["order_fc_vs_core"] = np.where(
        d["first_correct_trial"] < d["acquisition_trial_core"],
        "fc_before_core",
        np.where(
            d["first_correct_trial"] == d["acquisition_trial_core"],
            "fc_eq_core",
            "fc_after_core",
        ),
    )
    order_rows = []
    for col in ["order_collapse_prob_vs_fc", "order_fc_vs_core"]:
        vc = d[col].value_counts(dropna=False)
        n = int(vc.sum())
        for k, v in vc.items():
            order_rows.append({"comparison": col, "category": str(k), "n_blocks": int(v), "prop": float(v / n)})
    out_order = pd.DataFrame(order_rows)
    out_order.to_csv(OUT_ORDER, index=False)

    # Keep marker-independence models in same bundle with explicit name
    m.to_csv(OUT_MODELS, index=False)

    # Figure: A dissociation, B order proportions, C model coefficients
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    fig, axs = plt.subplots(1, 3, figsize=(14.2, 4.6), constrained_layout=False)

    # A: lag boxplot
    a1 = d["lag_collapse_to_fc_prob08"].to_numpy(dtype=float)
    a2 = d["lag_fc_to_core"].to_numpy(dtype=float)
    a1 = a1[np.isfinite(a1)]
    a2 = a2[np.isfinite(a2)]
    axs[0].boxplot([a1, a2], labels=["collapse->transition\nfirst-correct\n(prob, tau=0.8)", "transition\nfirst-correct->core"], widths=0.56)
    axs[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axs[0].set_title("A. Dissociation with probabilistic collapse")
    axs[0].set_ylabel("Lag (trials)")
    axs[0].grid(axis="y", linestyle="--", alpha=0.30)

    # B: order proportions
    bdat = out_order[out_order["comparison"] == "order_collapse_prob_vs_fc"].copy()
    ord_map = ["collapse_before_fc", "collapse_eq_fc", "collapse_after_fc"]
    bdat = bdat.set_index("category").reindex(ord_map).reset_index()
    axs[1].bar(np.arange(len(bdat)), bdat["prop"].to_numpy(dtype=float), color=["#2ca25f", "#9ecae1", "#ef3b2c"])
    axs[1].set_xticks(np.arange(len(bdat)))
    axs[1].set_xticklabels(["before", "equal", "after"])
    axs[1].set_ylim(0, 1.0)
    axs[1].set_ylabel("Proportion")
    axs[1].set_title("B. Collapse timing vs transition first-correct")
    axs[1].grid(axis="y", linestyle="--", alpha=0.30)

    # C: coefficients from independence models
    cdat = m[m["term"].isin(["collapse_to_fc", "feedback_locked_FRN", "feedback_locked_P3b"])].copy()
    cdat = cdat[np.isfinite(pd.to_numeric(cdat["beta"], errors="coerce"))].copy()
    if not cdat.empty:
        cdat = cdat.reset_index(drop=True)
        y = np.arange(len(cdat))
        beta = pd.to_numeric(cdat["beta"], errors="coerce").to_numpy(dtype=float)
        se = pd.to_numeric(cdat["se_cluster"], errors="coerce").to_numpy(dtype=float)
        lo = beta - 1.96 * se
        hi = beta + 1.96 * se
        for i in range(len(cdat)):
            axs[2].plot([lo[i], hi[i]], [y[i], y[i]], color="#4b5563", linewidth=2)
            axs[2].scatter([beta[i]], [y[i]], color="#111827", s=28)
        axs[2].set_yticks(y)
        axs[2].set_yticklabels(
            [f"{r['model']} | {r['term']} ({_safe_p_to_sig(float(r['p']))})" for _, r in cdat.iterrows()],
            fontsize=8,
        )
    axs[2].axvline(0, color="black", linewidth=1)
    axs[2].set_title("C. Marker-independence models")
    axs[2].set_xlabel("Beta (95% CI)")
    axs[2].grid(axis="x", linestyle="--", alpha=0.30)

    fig.suptitle("Probabilistic-collapse revision and marker-independence checks", fontsize=13, y=0.99)
    fig.subplots_adjust(wspace=0.45, left=0.05, right=0.99, top=0.84, bottom=0.12)
    fig.savefig(OUT_FIG, dpi=320, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"saved: {OUT_DISS}")
    print(f"saved: {OUT_ORDER}")
    print(f"saved: {OUT_MODELS}")
    print(f"saved: {OUT_FIG}")


if __name__ == "__main__":
    main()
