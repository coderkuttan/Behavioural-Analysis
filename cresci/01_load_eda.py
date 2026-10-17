import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(__file__)
UDIR = os.path.join(BASE, "users")

FILES = {
    "genuine_accounts_users.csv": ("genuine", 0),
    "fake_followers_users.csv": ("fake_followers", 1),
    "social_spambots_1_users.csv": ("social_spambots_1", 1),
    "social_spambots_2_users.csv": ("social_spambots_2", 1),
    "social_spambots_3_users.csv": ("social_spambots_3", 1),
}

dfs = []
for fname, (campaign, label) in FILES.items():
    path = os.path.join(UDIR, fname)
    df = pd.read_csv(path, dtype=str, quotechar='"', on_bad_lines="skip", low_memory=False)
    df["campaign"] = campaign
    df["label"] = label
    dfs.append(df)
    print(f"{fname}: {len(df)} rows, cols={len(df.columns)}")

all_df = pd.concat(dfs, ignore_index=True, sort=False)
print("\nTotal rows:", len(all_df))
print("Label counts:\n", all_df["label"].value_counts())
print("Campaign counts:\n", all_df["campaign"].value_counts())

# Check which reference-date columns exist per campaign
for col in ["created_at", "crawled_at", "updated", "timestamp"]:
    print(f"\nColumn '{col}' non-null by campaign:")
    print(all_df.groupby("campaign")[col].apply(lambda s: s.notna().sum()) if col in all_df.columns else "MISSING ENTIRELY")

all_df.to_pickle(os.path.join(BASE, "merged_raw.pkl"))
print("\nSaved merged_raw.pkl")
