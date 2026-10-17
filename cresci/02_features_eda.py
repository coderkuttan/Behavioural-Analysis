import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(__file__)
df = pd.read_pickle(os.path.join(BASE, "merged_raw.pkl"))

# numeric coercion
for c in ["statuses_count","followers_count","friends_count","favourites_count","listed_count"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# reference date per campaign = that campaign's own max(updated) timestamp (their crawl snapshot)
df["updated_dt"] = pd.to_datetime(df["updated"], errors="coerce")
ref_date = df.groupby("campaign")["updated_dt"].transform("max")
df["created_dt"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True, format="mixed")
df["updated_dt"] = pd.to_datetime(df["updated_dt"], utc=True)
ref_date = pd.to_datetime(ref_date, utc=True)
df["account_age_days"] = (ref_date - df["created_dt"]).dt.total_seconds() / 86400.0

df["friend_follower_ratio"] = df["friends_count"] / (df["followers_count"] + 1)
df["posting_rate"] = df["statuses_count"] / df["account_age_days"].clip(lower=1)
df["has_url"] = df["url"].notna().astype(int)
df["has_description"] = df["description"].fillna("").str.strip().str.len().gt(0).astype(int)
df["default_profile_image"] = pd.to_numeric(df["default_profile_image"], errors="coerce").fillna(0).astype(int)
df["geo_enabled"] = pd.to_numeric(df["geo_enabled"], errors="coerce").fillna(0).astype(int)
df["listed_zero"] = (df["listed_count"].fillna(0) == 0).astype(int)

cand = ["friend_follower_ratio","posting_rate","account_age_days","listed_count",
        "has_url","has_description","default_profile_image","geo_enabled","listed_zero",
        "followers_count","friends_count","statuses_count","favourites_count"]

print("=== Medians by label (0=genuine, 1=bot) ===")
print(df.groupby("label")[cand].median(numeric_only=True).T)

print("\n=== Means by label ===")
print(df.groupby("label")[cand].mean(numeric_only=True).T)

print("\n=== Percentiles of genuine class (for threshold-setting) ===")
for c in ["friend_follower_ratio","posting_rate","account_age_days","listed_count"]:
    g = df[df.label==0][c].dropna()
    print(c, "p50", g.quantile(.5), "p90", g.quantile(.9), "p95", g.quantile(.95), "p99", g.quantile(.99))

print("\n=== By campaign (bots), medians ===")
print(df.groupby("campaign")[cand].median(numeric_only=True).T)

# missingness check
print("\n=== Missingness of created_dt after parse ===")
print(df.groupby("campaign")["created_dt"].apply(lambda s: s.isna().sum()))
print("\n=== Missingness of account_age_days ===")
print(df.groupby("campaign")["account_age_days"].apply(lambda s: s.isna().sum()))
print(df.groupby("campaign")["account_age_days"].describe())

df.to_pickle(os.path.join(BASE, "merged_features_raw.pkl"))
