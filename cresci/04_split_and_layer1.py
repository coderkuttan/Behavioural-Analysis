import pandas as pd
import numpy as np
import os, json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve

BASE = os.path.dirname(__file__)
df = pd.read_pickle(os.path.join(BASE, "features_final.pkl")).reset_index(drop=True)
SIGCOLS = ["s_ffr","s_age","s_pr","s_eng","s_prof","s_social"]

SEED = 42
# 70/10/20 stratified split, same proportions as the original paper, by campaign+label
# so each split contains all campaigns proportionally
train_idx, temp_idx = train_test_split(df.index, test_size=0.30, stratify=df[["label","campaign"]].astype(str).agg("_".join, axis=1), random_state=SEED)
val_idx, test_idx = train_test_split(temp_idx, test_size=2/3, stratify=df.loc[temp_idx, ["label","campaign"]].astype(str).agg("_".join, axis=1), random_state=SEED)

df["split"] = "train"
df.loc[val_idx, "split"] = "val"
df.loc[test_idx, "split"] = "test"
print(df.groupby(["split","label"]).size())
print("\nTotal:", len(df), "train:", len(train_idx), "val:", len(val_idx), "test:", len(test_idx))

# ---- Layer-1 weight fitting: univariate AUC on VALIDATION split only (non-circular) ----
val = df.loc[val_idx]
aucs = {}
for c in SIGCOLS:
    aucs[c] = roc_auc_score(val.label, val[c])
print("\nUnivariate AUC per signal (validation only):", aucs)

# weight proportional to (AUC - 0.5), floor at 0, then renormalize to sum 1
raw_w = {c: max(aucs[c]-0.5, 0.0) for c in SIGCOLS}
s = sum(raw_w.values())
weights = {c: raw_w[c]/s for c in SIGCOLS}
print("\nLayer-1 weights (validation-fit, non-circular):", weights)

df["fraud_score"] = sum(df[c]*weights[c] for c in SIGCOLS)
val = df.loc[val_idx]

# threshold sweep on validation
from sklearn.metrics import f1_score
best_f1, best_tau = -1, None
sweep = []
for tau in np.arange(0.05, 0.85, 0.01):
    pred = (val["fraud_score"] >= tau).astype(int)
    f1 = f1_score(val.label, pred)
    sweep.append((round(tau,2), f1))
    if f1 > best_f1:
        best_f1, best_tau = f1, tau
print(f"\nBest validation F1={best_f1:.4f} at tau={best_tau:.2f}")

# deployed tau: pick a slightly conservative one near the plateau (mirror original narrative)
# find plateau within 0.005 of max
plateau = [t for t,f in sweep if best_f1 - f <= 0.005]
print("Plateau (within 0.005 of max):", min(plateau), "-", max(plateau))

deployed_tau = round(min(plateau), 2)  # deliberately permissive: low end of plateau
gate_pred_all = (df["fraud_score"] >= deployed_tau).astype(int)
gate_rate = gate_pred_all.mean()
print(f"Deployed tau={deployed_tau}, gate admits {gate_rate*100:.1f}% of all {len(df)} accounts")

results = dict(
    n_total=len(df), n_train=len(train_idx), n_val=len(val_idx), n_test=len(test_idx),
    aucs=aucs, weights=weights, best_val_f1=best_f1, best_val_tau=float(best_tau),
    plateau_lo=float(min(plateau)), plateau_hi=float(max(plateau)),
    deployed_tau=float(deployed_tau), gate_rate=float(gate_rate),
    n_gated=int(gate_pred_all.sum())
)
with open(os.path.join(BASE, "layer1_results.json"), "w") as f:
    json.dump(results, f, indent=2)

df["gated"] = gate_pred_all
df.to_pickle(os.path.join(BASE, "features_split.pkl"))
print("\nSaved features_split.pkl, layer1_results.json")
