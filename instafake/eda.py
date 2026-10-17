import json, os
import pandas as pd
import numpy as np

BASE = os.path.dirname(__file__)

def load(fn, label_field, label_val_true=1):
    with open(os.path.join(BASE, fn), encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

auto = load("automatedAccountData.json", "automatedBehaviour")
nonauto = load("nonautomatedAccountData.json", "automatedBehaviour")
auto["label"] = 1
nonauto["label"] = 0
df = pd.concat([auto, nonauto], ignore_index=True)
print("Total:", len(df), "label counts:\n", df.label.value_counts())

# derived fields
def span_days(times):
    if not isinstance(times, list) or len(times) < 2:
        return np.nan
    return (max(times) - min(times)) / 86400.0

df["post_span_days"] = df.mediaUploadTimes.apply(span_days)
df["posting_rate"] = df.userMediaCount / df.post_span_days.clip(lower=1)
df["avg_likes"] = df.mediaLikeNumbers.apply(lambda x: np.mean(x) if isinstance(x, list) and len(x)>0 else np.nan)
df["avg_comments"] = df.mediaCommentNumbers.apply(lambda x: np.mean(x) if isinstance(x, list) and len(x)>0 else np.nan)
df["engagement_rate"] = df.avg_likes / (df.userFollowerCount + 1)
df["friend_follower_ratio"] = df.userFollowingCount / (df.userFollowerCount + 1)
df["has_url"] = df.userHasExternalUrl.astype(int)
df["has_bio"] = (df.userBiographyLength > 0).astype(int)
df["has_highlights"] = df.userHasHighlighReels.astype(int)
df["digit_frac"] = df.usernameDigitCount / df.usernameLength.replace(0,1)
df["comments_disabled_frac"] = df.mediaCommentsAreDisabled.apply(lambda x: np.mean(x) if isinstance(x, list) and len(x)>0 else 0)

cand = ["userFollowerCount","userFollowingCount","friend_follower_ratio","posting_rate",
        "engagement_rate","avg_likes","avg_comments","userBiographyLength","has_url",
        "has_bio","has_highlights","digit_frac","userTagsCount","comments_disabled_frac",
        "userMediaCount"]

print("\n=== Medians by label ===")
print(df.groupby("label")[cand].median(numeric_only=True).T)
print("\n=== Means by label ===")
print(df.groupby("label")[cand].mean(numeric_only=True).T)

print("\nMissing post_span_days by label:\n", df.groupby("label").post_span_days.apply(lambda s: s.isna().sum()))

df.to_pickle(os.path.join(BASE, "insta_merged.pkl"))
