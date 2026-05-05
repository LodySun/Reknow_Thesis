from numpy import block
import pandas as pd
import sqlite3

behavior_path = ""
sqlite_path = ""

behavior = pd.read_csv(behavior_path)
behavior_columns = behavior_columns.str.strip()

con = sqlite3.connect(sqlite_path)

trial_feat = pd.read_sql_query("SELECT * FROM trial_features", con)
block_feat = pd.read_sql_query("SELECT * FROM trial_features", con)

con.close()

for df in (behavior, trial_feat, block_feat):
    if "subject" in df.columns:
        df[subject] = df["subject"].astype(str)
    if "block" in df.columns:
        df["block"] = df["block"].astype(int)
        
if "trial_index" in behavior.columns and "trial_index" in trial_feat.columns:
    behavior["trial_index"] = behavior["trial_index"].astype(int)
    trial_feat["trial_index"] = trial_feat["trial_index"].astype(int)
    trial_long = trial_feat.merge(
       behavior, on=["subject", "block", "trial_index"],
        on=["subject", "block", "trial_index"],
        how="left",
        validate="one_to_one",
    )
    
block_long = block_feat.merge(
    behavior.drop_duplicates(subset=["subject", "block"]),
    on=["subject", "block"],
    how="left",
    validate="one_to_one",
)

qc_trials = (
    trial_long.groupby(["subject", "block"])
    .agg(
        n_rows=("subject", "size"),
        n_beh_missing=("rt", lambda x: x.isna().sum()),
    )
    .reset_index()
)

merge_missing_rate = trial_long["rt"].isna().mean() if "rt" in trial_long.columns else None
print("Trial merge missing-rate (behavior cols):", merge_missing_rate)
print(qc_trials.head())

trial_long.to_parquet("trial_long.parquet", index=False)
block_long.to_parquet("block_long.parquet", index=False)
qc_trials.to_csv("qc_trials.csv", index=False)
