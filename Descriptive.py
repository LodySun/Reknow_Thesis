import pandas as pd
from pathlib import Path
import numpy as np

TRIALS_DIR = Path("Your_path")

def label_phase_one_block(block_df: pd.DataFrame) -> pd.DataFrame:
    tmp = block_df.sort_values("trial").reset_index(drop=True)

    is_correct = (tmp["correct"] == 1).astype(int)
    roll = is_correct.rolling(3, min_periods=3).sum()

# 3 correct trial streak means rule found
    found_idx = roll[roll == 3].index.min()

    if pd.isna(found_idx):
        tmp["phase"] = "search"
    else:
        tmp["phase"] = np.where(tmp.index < found_idx, "search", "found")

    return tmp

def analyze_one_subject(df: pd.DataFrame, subj_id: str):
    mean_rt_by_block = (
        df.groupby("block")["rt"]
          .mean()
          .reset_index()
    )
    mean_rt_by_block.insert(0, "subj", subj_id)

    mean_rt_by_rule = (
        df.groupby("rule")["rt"]
          .mean()
          .reset_index()
    )
    mean_rt_by_rule.insert(0, "subj", subj_id)

    blocks = []
    for _, block_df in df.groupby("block"):
        blocks.append(label_phase_one_block(block_df))
    df_with_phase = pd.concat(blocks, ignore_index=True)

    mean_rt_by_phase = (
        df_with_phase.groupby("phase")["rt"]
                     .mean()
                     .reset_index()
    )
    mean_rt_by_phase.insert(0, "subj", subj_id)

    # Under each rule, search & found avg RT
    mean_rt_by_rule_phase = (
        df_with_phase.groupby(["rule", "phase"])["rt"]
        .mean()
        .reset_index()
    )
    mean_rt_by_rule_phase.insert(0, "subj", subj_id)

    # Focusing on search：block, then between blocks
    search_df = df_with_phase[df_with_phase["phase"] == "search"].copy()
    search_df["block_num"] = pd.to_numeric(search_df["block"], errors="coerce")
    search_df["trial_num"] = pd.to_numeric(search_df["trial"], errors="coerce")
    search_df = search_df.sort_values(["block_num", "trial_num"]).reset_index(drop=True)
  
    block_search = (
        search_df.groupby("block", as_index=False)
        .agg(
            block_num=("block_num", "first"),
            search_rt=("rt", "mean"),
            search_fail_count=("correct", lambda s: (s == 0).sum()),
            rule=("rule", lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
        )
        .sort_values("block_num")
        .reset_index(drop=True)
    )
    block_search["prev_rule"] = block_search["rule"].shift(1)

    def to_group(rule: str):
        if isinstance(rule, str):
            if rule.startswith("G"):
                return "G"
            if rule.startswith("L"):
                return "L"
        return ""

    block_search["prev_group"] = block_search["prev_rule"].apply(to_group)
    block_search["curr_group"] = block_search["rule"].apply(to_group)

    # Rule-change:
    # G_to_L, L_to_G, G_to_G(Only G1<->G2), L_to_L(Only L1<->L2)
    block_search["change_type"] = np.select(
        [
            (block_search["prev_group"] == "G") & (block_search["curr_group"] == "L"),
            (block_search["prev_group"] == "L") & (block_search["curr_group"] == "G"),
            (block_search["prev_group"] == "G") & (block_search["curr_group"] == "G") & (block_search["prev_rule"] != block_search["rule"]),
            (block_search["prev_group"] == "L") & (block_search["curr_group"] == "L") & (block_search["prev_rule"] != block_search["rule"]),
        ],
        ["G_to_L", "L_to_G", "G_to_G", "L_to_L"],
        default="NO_CHANGE"
    )

    mean_search_rt_by_change = (
        block_search[block_search["change_type"] != "NO_CHANGE"]
        .groupby("change_type")["search_rt"]
        .mean()
        .reset_index()
        .rename(columns={"search_rt": "rt"})
    )
    mean_search_rt_by_change.insert(0, "subj", subj_id)

    # With different rule change(Inside G/L, between G & L, average fail)
    mean_search_fail_by_change = (
        block_search[block_search["change_type"] != "NO_CHANGE"]
        .groupby("change_type")["search_fail_count"]
        .mean()
        .reset_index()
        .rename(columns={"search_fail_count": "mean_fail_before_found"})
    )
    mean_search_fail_by_change.insert(0, "subj", subj_id)

    return (
        mean_rt_by_block,
        mean_rt_by_rule,
        mean_rt_by_phase,
        mean_search_rt_by_change,
        mean_rt_by_rule_phase,
        mean_search_fail_by_change,
    )


if __name__ == "__main__":
    from pathlib import Path
    TRIALS_DIR = Path("Your_path")

    all_block_stats = []
    all_rule_stats = []
    all_phase_stats = []
    all_search_change_stats = []
    all_rule_phase_stats = []
    all_search_fail_change_stats = []

    for i in range(1, 35):
        subj_id = f"reknow{i:03d}"
        csv_path = TRIALS_DIR / f"{subj_id}_trials.csv"
        if not csv_path.exists():
            print(f"Warning: Can't find: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        b, r, p, c, rp, f = analyze_one_subject(df, subj_id)
        all_block_stats.append(b)
        all_rule_stats.append(r)
        all_phase_stats.append(p)
        all_search_change_stats.append(c)
        all_rule_phase_stats.append(rp)
        all_search_fail_change_stats.append(f)

    if all_block_stats:
        pd.concat(all_block_stats, ignore_index=True).to_csv(
            TRIALS_DIR / "block_mean_rt.csv", index=False
        )
        pd.concat(all_rule_stats, ignore_index=True).to_csv(
            TRIALS_DIR / "rule_mean_rt.csv", index=False
        )
        pd.concat(all_phase_stats, ignore_index=True).to_csv(
            TRIALS_DIR / "phase_mean_rt.csv", index=False
        )
        pd.concat(all_search_change_stats, ignore_index=True).to_csv(
            TRIALS_DIR / "search_rt_by_rule_change.csv", index=False
        )
        pd.concat(all_rule_phase_stats, ignore_index=True).to_csv(
            TRIALS_DIR / "rule_phase_mean_rt.csv", index=False
        )
        pd.concat(all_search_fail_change_stats, ignore_index=True).to_csv(
            TRIALS_DIR / "search_fail_by_rule_change.csv", index=False
        )
        
        print("Exported block_mean_rt.csv, rule_mean_rt.csv, phase_mean_rt.csv, search_rt_by_rule_change.csv, rule_phase_mean_rt.csv, search_fail_by_rule_change.csv")
    else:
        print("Error")
