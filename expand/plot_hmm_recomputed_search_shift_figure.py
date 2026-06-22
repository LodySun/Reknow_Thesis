import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel


BASE = "base_dir"
IN_BLOCK = os.path.join(BASE, "hmm_based_search_metrics_block_level.csv")
OUT_PNG = os.path.join(BASE, "figure2_behavior_shift_position_hmm_recomputed.png")

# Keep style aligned to the original Figure2 generator.
CANDY = ["#9FC5E8", "#F4B6C2", "#B6D7A8", "#FFE599", "#C9B6E4", "#A7D8DE"]
BAR_WIDTH = 0.42
BAR_X_STEP = 0.78
BAR_JITTER = 0.045
BAR_BOX_ASPECT = 0.72


def _p_to_sig(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _paired_p(subj_level: pd.DataFrame, xcol: str, ycol: str, a, b) -> float:
    aa = subj_level[subj_level[xcol] == a][["subj", ycol]].rename(columns={ycol: "a"})
    bb = subj_level[subj_level[xcol] == b][["subj", ycol]].rename(columns={ycol: "b"})
    m = aa.merge(bb, on="subj", how="inner").dropna()
    if len(m) < 6:
        return np.nan
    res = ttest_rel(m["a"], m["b"], nan_policy="omit")
    return float(res.pvalue)


def grouped_bar(ax, df, xcol, ycol, order, title, ylabel, sig_pairs=None):
    sub = df[df[xcol].isin(order)].groupby(["subj", xcol], as_index=False)[ycol].mean()
    g = sub.groupby(xcol)[ycol].agg(["mean", "sem"]).reindex(order)
    x = np.arange(len(order), dtype=float) * BAR_X_STEP
    bar_colors = (CANDY * ((len(order) // len(CANDY)) + 1))[: len(order)]
    ax.bar(
        x,
        g["mean"].values,
        yerr=g["sem"].values,
        width=BAR_WIDTH,
        capsize=4,
        alpha=0.9,
        color=bar_colors,
        edgecolor="#666666",
        linewidth=0.8,
    )

    rng = np.random.default_rng(20260320)
    for i, cat in enumerate(order):
        vals = sub[sub[xcol] == cat][ycol].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-BAR_JITTER, BAR_JITTER, size=len(vals))
        ax.scatter(
            np.full(len(vals), x[i]) + jitter,
            vals,
            s=28,
            alpha=0.55,
            marker="o",
            facecolors="black",
            edgecolors="black",
            linewidths=0.35,
            zorder=3,
        )

    pts = sub[sub[xcol].isin(order)][ycol].dropna().to_numpy(dtype=float)
    bar_low = float(np.nanmin(g["mean"].values - np.nan_to_num(g["sem"].values, nan=0.0)))
    bar_high = float(np.nanmax(g["mean"].values + np.nan_to_num(g["sem"].values, nan=0.0)))
    pmin = float(np.nanmin(pts)) if len(pts) else bar_low
    pmax = float(np.nanmax(pts)) if len(pts) else bar_high
    ymin_raw = float(min(bar_low, pmin))
    ymax_raw = float(max(bar_high, pmax))
    yspan = max(ymax_raw - ymin_raw, 1e-9)
    y_bottom = ymin_raw - 0.05 * yspan
    y_top = ymax_raw + 0.05 * yspan

    if sig_pairs:
        data_top = float(max(bar_high, pmax))
        data_bot = float(min(bar_low, pmin))
        yr = max(data_top - data_bot, 1e-6)
        valid_pairs = [(a, b) for (a, b) in sig_pairs if a in order and b in order]
        n_br = len(valid_pairs)
        step = (0.13 + 0.06 * max(0, n_br - 1)) * yr
        base = data_top + 0.10 * yr
        for k, (a, b) in enumerate(valid_pairs):
            i1 = order.index(a)
            i2 = order.index(b)
            xa = x[i1]
            xb = x[i2]
            p = _paired_p(sub, xcol, ycol, a, b)
            label = _p_to_sig(p)
            y = base + k * step
            drop = 0.012 * yr
            ax.plot([xa, xa, xb, xb], [y - drop, y, y, y - drop], color="black", linewidth=1.4)
            ax.text(
                (xa + xb) / 2,
                y + 0.02 * yr,
                label,
                ha="center",
                va="bottom",
                fontsize=10,
                color="#7A1F1F",
            )
            y_top = max(y_top, y + 0.34 * yr)
        if n_br > 0:
            y_top = max(y_top, base + max(n_br - 1, 0) * step + 0.36 * yr)

    ax.set_ylim(y_bottom, y_top)
    ax.set_xlim(float(np.min(x) - 0.32), float(np.max(x) + 0.32))
    ax.set_xticks(x)
    ax.set_xticklabels(["within-level", "cross-level"], rotation=15)
    ax.set_box_aspect(BAR_BOX_ASPECT)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)


def _add_panel_label(ax, label: str):
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")


def main() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    d = pd.read_csv(IN_BLOCK)
    d["subj"] = d["subj"].astype(str)
    d["shift_type"] = d["shift_type"].astype(str)
    for c in ["search_len", "first_correct_latency"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["shift_type"].isin(["within_level", "cross_level"])].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=False)
    grouped_bar(
        axes[0],
        d,
        "shift_type",
        "search_len",
        ["within_level", "cross_level"],
        "Search Length By Shift Type (core - 1)",
        "Search Length (Trials; core - 1)",
        sig_pairs=[("within_level", "cross_level")],
    )
    grouped_bar(
        axes[1],
        d,
        "shift_type",
        "first_correct_latency",
        ["within_level", "cross_level"],
        "First-Correct Latency By Shift Type",
        "First-Correct Latency (Trials)",
        sig_pairs=[("within_level", "cross_level")],
    )
    _add_panel_label(axes[0], "A")
    _add_panel_label(axes[1], "B")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.93, wspace=0.20)
    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
