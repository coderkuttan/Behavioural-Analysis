import pandas as pd
import numpy as np
import os, json

BASE = os.path.dirname(__file__)
df = pd.read_pickle(os.path.join(BASE, "merged_features_raw.pkl"))

genuine = df[df.label == 0]

def pct(s, q):
    return s.dropna().quantile(q)

# ---- thresholds derived ONLY from the genuine class (documented, reproducible) ----
ffr_cap = pct(genuine.friend_follower_ratio, 0.99)          # ~8.76 -> use as cap
age_ref = genuine.account_age_days.median()                  # ~1510 days
pr_lo = pct(np.log1p(genuine.posting_rate), 0.10)
pr_hi = pct(np.log1p(genuine.posting_rate), 0.90)
eng_ratio_genuine_med = (genuine.favourites_count / (genuine.statuses_count + 1)).median()

THRESH = dict(ffr_cap=float(ffr_cap), age_ref=float(age_ref),
              pr_lo=float(pr_lo), pr_hi=float(pr_hi),
              eng_ratio_ref=float(eng_ratio_genuine_med))
print("Derived thresholds (from genuine class only):", THRESH)

def s_friend_follower(d):
    return np.clip(d.friend_follower_ratio / ffr_cap, 0, 1)

def s_age(d):
    return np.clip(1 - d.account_age_days / age_ref, 0, 1)

def s_posting_rate(d):
    x = np.log1p(d.posting_rate)
    below = np.clip((pr_lo - x) / max(pr_lo, 1e-6), 0, 1)          # too quiet
    above = np.clip((x - pr_hi) / max(pr_hi, 1e-6), 0, 1)          # too spammy
    return np.clip(np.maximum(below, above), 0, 1)

def s_engagement_void(d):
    ratio = d.favourites_count / (d.statuses_count + 1)
    return np.clip(1 - ratio / eng_ratio_genuine_med, 0, 1)

def s_profile_void(d):
    return ((1 - d.has_url) + (1 - d.has_description) + d.default_profile_image) / 3.0

def s_social_proof_void(d):
    return (d.listed_zero + (1 - d.geo_enabled)) / 2.0

df["s_ffr"] = s_friend_follower(df)
df["s_age"] = s_age(df)
df["s_pr"] = s_posting_rate(df)
df["s_eng"] = s_engagement_void(df)
df["s_prof"] = s_profile_void(df)
df["s_social"] = s_social_proof_void(df)

SIGCOLS = ["s_ffr","s_age","s_pr","s_eng","s_prof","s_social"]
print("\nAny NaNs in signals?")
print(df[SIGCOLS].isna().sum())

# drop rows with unparseable core fields
before = len(df)
df = df.dropna(subset=SIGCOLS + ["label"]).reset_index(drop=True)
print(f"\nDropped {before-len(df)} rows with missing signal values; {len(df)} remain")

print("\nSignal means by label:")
print(df.groupby("label")[SIGCOLS].mean())

df.to_pickle(os.path.join(BASE, "features_final.pkl"))
with open(os.path.join(BASE, "thresholds.json"), "w") as f:
    json.dump(THRESH, f, indent=2)
print("\nSaved features_final.pkl and thresholds.json")
print("Final label counts:\n", df.label.value_counts())
print("Final campaign counts:\n", df.campaign.value_counts())
