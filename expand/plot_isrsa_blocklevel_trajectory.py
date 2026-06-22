import os

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import spearmanr


matplotlib.use("Agg")

BASE = "base_dir"
EXP = os.path.join(BASE, "expand")

BEH_CSV = os.path.join(EXP, "isrsa_two_blocklevel_behavior_subject_features.csv")
NEU_CSV = os.path.join(EXP, "isrsa_two_blocklevel_neural_subject_features.csv")
RESULT_CSV = os.path.join(EXP, "isrsa_two_blocklevel_rdm_results.csv")

OUT_PNG = os.path.join(EXP, "figure_isrsa_blocklevel_trajectory.png")
OUT_PDF = os.path.join(EXP, "figure_isrsa_blocklevel_trajectory.pdf")
OUT_POINTS = os.path.join(EXP, "isrsa_blocklevel_trajectory_pairwise_points.csv")

BLOCK_NUMS = list(range(13, 109))
NEURAL_FEATURES = [
    "neu_p3b_transition_fc_minus_acquired_core",
    "neu_frn_p3b_balance_fc",
]


def set_helvetica_font() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8


def zscore_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        x = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x)
        if ok.sum() == 0:
            out[col] = 0.0
            continue
        mu = np.nanmean(x)
        sd = np.nanstd(x, ddof=1)
        out[col] = (x - mu) / sd if np.isfinite(sd) and sd > 0 else 0.0
    return out


def upper(mat: np.ndarray) -> np.ndarray:
    return mat[np.triu_indices_from(mat, k=1)]


def nan_euclidean_rdm(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    n, p = x.shape
    out = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        out[i, i] = 0.0
        for j in range(i + 1, n):
            ok = np.isfinite(x[i]) & np.isfinite(x[j])
            n_ok = int(ok.sum())
            if n_ok == 0:
                continue
            diff = x[i, ok] - x[j, ok]
            dist = np.sqrt(np.sum(diff * diff) * p / n_ok)
            out[i, j] = dist
            out[j, i] = dist
    return out


def p_text(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.4f}".rstrip("0").rstrip(".")


def make_rdms(behavior_cols: list[str], neural_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    beh = pd.read_csv(BEH_CSV)
    neu = pd.read_csv(NEU_CSV)
    beh["subj"] = beh["subj"].astype(str)
    neu["subj"] = neu["subj"].astype(str)
    d = beh[["subj"] + behavior_cols].merge(neu[["subj"] + neural_cols], on="subj", how="inner")
    d = d.dropna(subset=behavior_cols + neural_cols, how="all").reset_index(drop=True)
    bz = zscore_cols(d[["subj"] + behavior_cols], behavior_cols)
    nz = zscore_cols(d[["subj"] + neural_cols], neural_cols)
    brdm = nan_euclidean_rdm(bz[behavior_cols].to_numpy(dtype=float))
    nrdm = nan_euclidean_rdm(nz[neural_cols].to_numpy(dtype=float))
    return d, brdm, nrdm


def draw_heatmap(ax, mat: np.ndarray, title: str, vmax: float):
    im = ax.imshow(mat, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=10.5, pad=7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def draw_scatter(ax, x: np.ndarray, y: np.ndarray, rho: float, p_perm: float, p_fdr: float, label: str):
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    ax.scatter(x, y, s=17, color="#2b6f9f", alpha=0.42, edgecolors="none")
    if len(x) > 2:
        coef = np.polyfit(x, y, deg=1)
        xx = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
        ax.plot(xx, coef[0] * xx + coef[1], color="#d95f02", linewidth=2.0)
    ax.set_title(f"{label}\nRDM correspondence", fontsize=10.5, pad=7)
    ax.set_xlabel("Behavioral dissimilarity")
    ax.set_ylabel("Neural dissimilarity")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.28)
    txt = (
        f"Spearman rho = {rho:.3f}\n"
        f"p_perm = {p_text(p_perm)}\n"
        f"FDR q = {p_text(p_fdr)}\n"
        f"pairs = {len(x)}"
    )
    ax.text(
        0.04,
        0.96,
        txt,
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=9.2,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#d0d0d0", linewidth=0.8, alpha=0.92),
    )


def main() -> None:
    set_helvetica_font()
    os.makedirs(EXP, exist_ok=True)

    results = pd.read_csv(RESULT_CSV)
    specs = [
        {
            "key": "block_nonzero_96block_nan_euclidean",
            "label": "A. Transition-delay presence",
            "behavior_title": "Behavior RDM\n96-block delay-presence trajectory",
            "behavior_cols": [f"block{i}_transition_nonzero" for i in BLOCK_NUMS],
        },
        {
            "key": "block_loglag_96block_nan_euclidean",
            "label": "B. Acquisition-lag magnitude",
            "behavior_title": "Behavior RDM\n96-block log1p lag trajectory",
            "behavior_cols": [f"block{i}_lag_log1p" for i in BLOCK_NUMS],
        },
    ]
    neural_cols = [f"block{i}_{feat}" for i in BLOCK_NUMS for feat in NEURAL_FEATURES]

    fig = plt.figure(figsize=(11.8, 7.2), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.22], height_ratios=[1, 1], wspace=0.34, hspace=0.42)

    point_rows = []
    for row, spec in enumerate(specs):
        d, brdm, nrdm = make_rdms(spec["behavior_cols"], neural_cols)
        order_score = np.nanmean(d[spec["behavior_cols"]].to_numpy(dtype=float), axis=1)
        order = np.argsort(order_score)
        brdm_show = brdm[np.ix_(order, order)]
        nrdm_show = nrdm[np.ix_(order, order)]

        res = results[results["behavior_rdm"] == spec["key"]].iloc[0]
        rho = float(res["rho_spearman_uppertri"])
        p_perm = float(res["p_perm"])
        p_fdr = float(res["p_fdr_two"])

        ax_b = fig.add_subplot(gs[row, 0])
        ax_n = fig.add_subplot(gs[row, 1])
        ax_s = fig.add_subplot(gs[row, 2])

        b_vmax = float(np.nanpercentile(upper(brdm), 95))
        n_vmax = float(np.nanpercentile(upper(nrdm), 95))
        im_b = draw_heatmap(ax_b, brdm_show, spec["behavior_title"], b_vmax)
        im_n = draw_heatmap(
            ax_n,
            nrdm_show,
            "Neural RDM\n96-block P3b transition/acquired + FRN-P3b",
            n_vmax,
        )

        cb_b = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.025)
        cb_b.ax.tick_params(labelsize=8)
        cb_n = fig.colorbar(im_n, ax=ax_n, fraction=0.046, pad=0.025)
        cb_n.ax.tick_params(labelsize=8)

        bx = upper(brdm)
        ny = upper(nrdm)
        draw_scatter(ax_s, bx, ny, rho, p_perm, p_fdr, spec["label"])

        for pair_idx, (beh_dist, neu_dist) in enumerate(zip(bx, ny), start=1):
            point_rows.append(
                {
                    "behavior_rdm": spec["key"],
                    "pair_index": pair_idx,
                    "behavior_dissimilarity": beh_dist,
                    "neural_dissimilarity": neu_dist,
                }
            )

    fig.suptitle(
        "Block-level IS-RSA: behavioral acquisition trajectories and neural transition dynamics",
        fontsize=14,
        y=0.985,
    )
    fig.savefig(OUT_PNG, dpi=340, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    pd.DataFrame(point_rows).to_csv(OUT_POINTS, index=False)
    print(f"saved: {OUT_PNG}")
    print(f"saved: {OUT_PDF}")
    print(f"saved: {OUT_POINTS}")

    check = []
    for spec in specs:
        d, brdm, nrdm = make_rdms(spec["behavior_cols"], neural_cols)
        ok = np.isfinite(upper(brdm)) & np.isfinite(upper(nrdm))
        rho, _ = spearmanr(upper(brdm)[ok], upper(nrdm)[ok])
        check.append({"behavior_rdm": spec["key"], "rho_recomputed_for_plot": float(rho), "n_subjects": int(len(d))})
    print(pd.DataFrame(check).to_string(index=False))


if __name__ == "__main__":
    main()
