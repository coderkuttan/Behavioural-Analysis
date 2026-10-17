"""Isolated-node blind spot: size it exactly and test two cheap remedies (reviewer comment 7).

(a) Fallback: accounts rejected by the Layer-1 gate are scored by the already-trained
    Random Forest (06_models.py) instead of the GCN; gated accounts keep the GCN label.
(b) Gate: how many accounts/bots a more permissive tau would admit (validation split).
No model is retrained. Outputs: experiments/isolated_remedies_results.json
"""
import os, json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cresci")
OUT = os.path.dirname(os.path.abspath(__file__))
SIGCOLS = ["s_ffr", "s_age", "s_pr", "s_eng", "s_prof", "s_social"]

df = pd.read_pickle(os.path.join(BASE, "features_split.pkl")).reset_index(drop=True)
gcn_pred = np.load(os.path.join(BASE, "gcn_preds.npy"))
with open(os.path.join(BASE, "models.pkl"), "rb") as f:
    rf = pickle.load(f)["rf"]

y = df.label.to_numpy()
test = (df.split == "test").to_numpy()
val = (df.split == "val").to_numpy()
gated = df.gated.to_numpy() == 1  # every gated account has >=1 edge (graph_stats.json)


def metrics(mask, pred):
    yy, pp = y[mask], pred[mask]
    p, r, f1, _ = precision_recall_fscore_support(yy, pp, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(yy, pp, labels=[0, 1]).ravel()
    return dict(n=int(mask.sum()), bots=int(yy.sum()), tp=int(tp), fp=int(fp), fn=int(fn),
                precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4),
                fpr=round(fp / (fp + tn), 4))


out = {}
out["gcn_all_test"] = metrics(test, gcn_pred)
out["gcn_gated_test"] = metrics(test & gated, gcn_pred)
out["gcn_isolated_test"] = metrics(test & ~gated, gcn_pred)

rf_pred = rf.predict(df[SIGCOLS].to_numpy())
hybrid = np.where(gated, gcn_pred, rf_pred)
out["rf_on_isolated_test"] = metrics(test & ~gated, rf_pred)
out["hybrid_gcn_plus_rf_fallback_test"] = metrics(test, hybrid)

fs = df.fraud_score.to_numpy()
gate = []
for tau in [0.32, 0.30, 0.28, 0.25, 0.20]:
    adm = fs >= tau
    gate.append(dict(tau=tau,
                     val_bot_recall_of_gate=round(float(adm[val & (y == 1)].mean()), 4),
                     admitted_all=int(adm.sum()), admitted_frac=round(float(adm.mean()), 4),
                     ungated_bots_test=int((~adm & test & (y == 1)).sum()),
                     admitted_genuine_test=int((adm & test & (y == 0)).sum())))
out["gate_tau_options"] = gate

for k, v in out.items():
    print(k, v)
with open(os.path.join(OUT, "isolated_remedies_results.json"), "w") as f:
    json.dump(out, f, indent=2)
