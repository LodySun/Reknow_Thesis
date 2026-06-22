import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from figure_style_and_metrics import update_metrics_file


BASE = Path("base_dir")
OUT_DIR = BASE / "trials_trialwise" / "1s_comp" / "eeg_paper_results" / "solidity"
OUT_PNG = OUT_DIR / "figure29_hmm_methods_schematic.png"


def make_figure():
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
    rcParams["figure.dpi"] = 220
    rcParams["mathtext.fontset"] = "dejavusans"

    fig, ax = plt.subplots(figsize=(14.4, 7.2))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.4)
    ax.axis("off")

    ax.text(
        7.5,
        7.9,
        "Hidden Markov Model for Acquisition Dynamics",
        ha="center",
        va="center",
        fontsize=27,
        fontweight="bold",
    )
    ax.text(
        7.5,
        7.42,
        "Latent-state transition from search to acquired across trials",
        ha="center",
        va="center",
        fontsize=19,
    )

    xs = [2.55, 4.65, 6.75, 8.85, 10.95, 13.05]
    y_hidden = 5.8
    y_obs = 2.95

    hidden_labels = [
        "$z_1$\nsearch",
        "$z_2$\nsearch",
        "$z_3$\nsearch",
        "$z_4$\nacquired",
        "$z_5$\nacquired",
        "$z_6$\nacquired",
    ]
    obs_labels = [
        "$y_1$\nerror",
        "$y_2$\ncorrect",
        "$y_3$\nerror",
        "$y_4$\ncorrect",
        "$y_5$\ncorrect",
        "$y_6$\ncorrect",
    ]

    ax.text(0.8, y_hidden + 0.03, "Hidden\nstates", ha="left", va="center", fontsize=18, fontweight="bold")
    ax.text(0.8, y_obs, "Observed\nfeedback", ha="left", va="center", fontsize=18, fontweight="bold")

    ax.text(7.8, 6.65, r"State transitions   $P(z_t \mid z_{t-1})$", ha="center", va="center", fontsize=16)
    ax.text(7.8, 4.35, "Emissions\n" + r"$P(y_t \mid z_t)$", ha="center", va="center", fontsize=16, linespacing=1.1)

    box_w, box_h = 1.38, 1.0
    for i, (x, lab) in enumerate(zip(xs, hidden_labels)):
        face = "#f4f4f4" if i < 3 else "#ffffff"
        box = FancyBboxPatch(
            (x - box_w / 2, y_hidden - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.03,rounding_size=0.16",
            linewidth=1.9,
            edgecolor="black",
            facecolor=face,
        )
        ax.add_patch(box)
        ax.text(x, y_hidden, lab, ha="center", va="center", fontsize=15.5)

    r = 0.45
    for x, lab in zip(xs, obs_labels):
        circ = Circle((x, y_obs), r, linewidth=1.9, edgecolor="black", facecolor="white")
        ax.add_patch(circ)
        ax.text(x, y_obs, lab, ha="center", va="center", fontsize=13.5)

    for i in range(len(xs) - 1):
        arr = FancyArrowPatch(
            (xs[i] + box_w / 2 + 0.02, y_hidden),
            (xs[i + 1] - box_w / 2 - 0.02, y_hidden),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.6,
            color="black",
        )
        ax.add_patch(arr)

    for x in xs:
        arr = FancyArrowPatch(
            (x, y_hidden - box_h / 2 - 0.06),
            (x, y_obs + r + 0.06),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.45,
            color="black",
        )
        ax.add_patch(arr)

    seq_arrow = FancyArrowPatch(
        (2.1, 1.45), (13.45, 1.45), arrowstyle="-|>", mutation_scale=15, linewidth=1.25, color="black"
    )
    ax.add_patch(seq_arrow)
    ax.text(7.8, 1.76, "trial sequence", ha="center", va="bottom", fontsize=14, style="italic")

    ax.text(
        7.8,
        0.78,
        r"$z_t \in \{0,1\}$  ·  0 = search  ·  1 = acquired  ·  "
        r"$P(y_t=1 \mid z_t=1) > P(y_t=1 \mid z_t=0)$",
        ha="center",
        va="center",
        fontsize=13.5,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0.6)
    fig.savefig(OUT_PNG, bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    update_metrics_file(
        figure_id="figure29",
        figure_title="Hidden Markov Model for Acquisition Dynamics",
        figure_caption="Methods schematic of latent-state transitions and emission mapping in the unified two-state HMM.",
        rows=[
            {
                "panel": "schematic",
                "metric_name": "methods_diagram",
                "p_value": np.nan,
                "significance": "NA",
                "effect_or_stat": "",
                "notes": "No inferential statistics displayed.",
            }
        ],
    )
    print(f"[OK] Saved: {OUT_PNG}")


if __name__ == "__main__":
    make_figure()
