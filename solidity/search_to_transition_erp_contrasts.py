import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from figure_style_and_metrics import p_to_sig, set_helvetica_font, update_metrics_file
from transition_stages import assign_transition_stage


BASE = "/Users/lodysun/Desktop/Thesis/trials_trialwise/1s_comp/eeg_paper_results/solidity"
TRIAL_POST_CSV = os.path.join(BASE, "hmm_unified_trial_posteriors.csv")
EEG_TRIAL_CSV = "/Users/lodysun/Desktop/Thesis/trials_trialwise/1s_comp/eeg_tables/eeg_trial_long.csv"
OUT_DIR = BASE
set_helvetica_font()

FEATURES = ["feedback_locked_FRN", "feedback_locked_P3b"]
CATEGORIES = ["search_error", "transition_pre_core_correct", "transition_first_correct"]


def _load_merged():
    eeg = pd.read_csv(EEG_TRIAL_CSV)
    post = pd.read_csv(TRIAL_POST_CSV)
    eeg["subj"] = eeg["subj"].astype(str)
    eeg["block_id"] = eeg["block_id"].astype(str)
    eeg["trial_id"] = pd.to_numeric(eeg["trial_id"], errors="coerce")
    post["subj"] = post["subj"].astype(str)
    post["block_id"] = post["block_id"].astype(str)
    post["trial_id"] = pd.to_numeric(post["trial_id"], errors="coerce")

    m = eeg.merge(
        post[
            [
                "subj",
                "block_id",
                "trial_id",
                "correctness",
                "trial_index_1based",
                "first_correct_trial",
                "acquisition_trial_core",
                "set_index",
            ]
        ],
        on=["subj", "block_id", "trial_id"],
        how="inner",
    )
    for c in ["correctness", "trial_index_1based", "first_correct_trial", "acquisition_trial_core", "set_index"] + FEATURES:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    return m


def _categorize(m: pd.DataFrame) -> pd.DataFrame:
    d = m.copy()
    # Trial-stage classification: single source of truth (transition_stages).
    # Variant: index = trial_index_1based, correctness filter; only the three
    # CATEGORIES are kept (acquired / search_correct intentionally omitted via order).
    d["category"] = assign_transition_stage(
        d,
        trial_col="trial_index_1based",
        correct_suffix=True,
        order=("search_error", "transition_pre_core", "transition_first_correct"),
    )
    d = d[d["category"].isin(CATEGORIES)].copy()
    return d


def _block_stage_means(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subj, bid), g in d.groupby(["subj", "block_id"], sort=False):
        set_idx = float(pd.to_numeric(g["set_index"], errors="coerce").dropna().iloc[0]) if g["set_index"].notna().any() else np.nan
        for feat in FEATURES:
            for cat in CATEGORIES:
                vals = g.loc[g["category"] == cat, feat].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                rows.append(
                    {
                        "subj": subj,
                        "block_id": str(bid),
                        "set_index": set_idx,
                        "feature": feat,
                        "category": cat,
                        "value_mean": float(np.nanmean(vals)) if len(vals) else np.nan,
                        "n_trials": int(len(vals)),
                    }
                )
    return pd.DataFrame(rows)


def _paired_tests(stage: pd.DataFrame) -> pd.DataFrame:
    pair_defs = [
        ("transition_first_correct", "transition_pre_core_correct"),
        ("transition_pre_core_correct", "search_error"),
        ("transition_first_correct", "search_error"),
    ]
    subset_defs = {
        "all_blocks": np.ones(len(stage), dtype=bool),
        "early_sets_1_2": pd.to_numeric(stage["set_index"], errors="coerce").isin([1, 2]),
    }
    rows = []
    for sub_name, mask in subset_defs.items():
        sd = stage[mask].copy()
        for feat in FEATURES:
            sf = sd[sd["feature"] == feat]
            for a, b in pair_defs:
                pa = sf[sf["category"] == a][["subj", "block_id", "value_mean"]].rename(columns={"value_mean": "va"})
                pb = sf[sf["category"] == b][["subj", "block_id", "value_mean"]].rename(columns={"value_mean": "vb"})
                m = pa.merge(pb, on=["subj", "block_id"], how="inner").dropna()
                if len(m) < 8:
                    continue
                t, p = ttest_rel(m["va"], m["vb"], nan_policy="omit")
                rows.append(
                    {
                        "subset": sub_name,
                        "feature": feat,
                        "comparison": f"{a}_minus_{b}",
                        "n_blocks": int(len(m)),
                        "mean_diff": float((m["va"] - m["vb"]).mean()),
                        "t": float(t),
                        "p": float(p),
                    }
                )
    return pd.DataFrame(rows)


def _plot(stage: pd.DataFrame, tests: pd.DataFrame, out_png: str):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), constrained_layout=False)
    order = CATEGORIES
    x = np.arange(len(order))
    labels = ["Search\nerror", "Transition\npre-core correct", "Transition\nfirst correct"]
    colors = {"feedback_locked_FRN": "#6a51a3", "feedback_locked_P3b": "#1f78b4"}

    early = stage[pd.to_numeric(stage["set_index"], errors="coerce").isin([1, 2])].copy()
    for ax, feat in zip(axes, FEATURES):
        sf = early[early["feature"] == feat]
        subj_mean = sf.groupby(["subj", "category"], as_index=False)["value_mean"].mean()
        piv = subj_mean.pivot_table(index="subj", columns="category", values="value_mean")
        mu, se = [], []
        for cat in order:
            arr = piv[cat].to_numpy(dtype=float) if cat in piv.columns else np.array([])
            arr = arr[np.isfinite(arr)]
            mu.append(np.nanmean(arr) if len(arr) else np.nan)
            se.append(np.nanstd(arr, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.nan)
        mu = np.array(mu, dtype=float)
        se = np.array(se, dtype=float)
        ax.errorbar(x, mu, yerr=1.96 * se, fmt="o-", color=colors[feat], linewidth=2.2, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"{feat} (early sets 1-2)")
        ax.set_ylabel("Amplitude (uV)")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        # Add significant contrasts at top (early sets 1-2), with stacked brackets.
        pairs = [
            ("transition_pre_core_correct_minus_search_error", 0, 1),
            ("transition_first_correct_minus_transition_pre_core_correct", 1, 2),
            ("transition_first_correct_minus_search_error", 0, 2),
        ]
        qf = tests[(tests["subset"] == "early_sets_1_2") & (tests["feature"] == feat)].copy()
        y_ref = float(np.nanmax(mu + 1.96 * np.nan_to_num(se, nan=0.0)))
        level = 0
        for cname, x1, x2 in pairs:
            qq = qf[qf["comparison"] == cname]
            if qq.empty:
                continue
            pv = float(qq.iloc[0]["p"])
            if not np.isfinite(pv) or pv >= 0.05:
                continue
            sig = p_to_sig(pv)
            y0 = y_ref + 0.12 + level * 0.20
            ax.plot([x1, x1, x2, x2], [y0 - 0.05, y0, y0, y0 - 0.05], color="black", linewidth=1)
            ax.text((x1 + x2) / 2, y0 + 0.015, sig, ha="center", va="bottom", fontsize=11, fontweight="bold")
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, max(hi, y0 + 0.13))
            level += 1
    fig.suptitle("Search-to-Transition ERP Contrasts", fontsize=14, y=0.98)
    fig.savefig(out_png, dpi=320, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main():
    fig_title = "Search-to-Transition ERP Contrasts"
    fig_caption = (
        "Early-set FRN and P3b contrasts across search-error, transition pre-core correct, and transition first-correct categories."
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    m = _load_merged()
    d = _categorize(m)
    stage = _block_stage_means(d)
    tests = _paired_tests(stage)

    p1 = os.path.join(OUT_DIR, "search_transition_erp_stage_means_long.csv")
    p2 = os.path.join(OUT_DIR, "search_transition_erp_pair_tests.csv")
    p3 = os.path.join(OUT_DIR, "figure26_search_to_transition_erp_contrasts.png")

    stage.to_csv(p1, index=False)
    tests.to_csv(p2, index=False)
    _plot(stage, tests, p3)
    metric_rows = []
    if not tests.empty:
        for _, r in tests.iterrows():
            pv = float(r["p"]) if np.isfinite(r["p"]) else np.nan
            metric_rows.append(
                {
                    "panel": str(r.get("feature", "")),
                    "metric_name": f"{r.get('subset', '')}:{r.get('comparison', '')}",
                    "p_value": pv,
                    "significance": p_to_sig(pv),
                    "effect_or_stat": f"t={float(r.get('t', np.nan)):.3f}; mean_diff={float(r.get('mean_diff', np.nan)):.3f}",
                    "notes": f"n_blocks={int(r.get('n_blocks', 0))}",
                }
            )
    else:
        metric_rows.append(
            {
                "panel": "FRN/P3b",
                "metric_name": "descriptive_three_way_plot",
                "p_value": np.nan,
                "significance": "NA",
                "effect_or_stat": "",
                "notes": "No paired tests available.",
            }
        )
    update_metrics_file(
        figure_id="figure26",
        figure_title=fig_title,
        figure_caption=fig_caption,
        rows=metric_rows,
    )

    print(f"saved: {p1}")
    print(f"saved: {p2}")
    print(f"saved: {p3}")
    if not tests.empty:
        print(tests.sort_values(["subset", "feature", "p"]).to_string(index=False))


if __name__ == "__main__":
    main()
