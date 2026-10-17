import pandas as pd, numpy as np, os, json, pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score

BASE = os.path.dirname(__file__)
OUT = os.path.join(os.path.dirname(os.path.dirname(BASE)), "..")  # not used; write directly to paper folder
PAPER_DIR = r"F:\christ\sem 5\research paper"

df = pd.read_pickle(os.path.join(BASE, "features_split.pkl")).reset_index(drop=True)
SIGCOLS = ["s_ffr","s_age","s_pr","s_eng","s_prof","s_social"]
test = df[df.split=="test"]
yte = test.label.to_numpy()

with open(os.path.join(BASE, "models.pkl"), "rb") as f:
    M = pickle.load(f)
rf, xg, mlp, scaler = M["rf"], M["xg"], M["mlp"], M["scaler"]
Xte = test[SIGCOLS].to_numpy()

gcn_probs = np.load(os.path.join(BASE, "gcn_probs.npy"))
test_mask = (df.split=="test").to_numpy()
gcn_score_te = gcn_probs[test_mask]

scores = {
    "Random Forest": rf.predict_proba(Xte)[:,1],
    "XGBoost": xg.predict_proba(Xte)[:,1],
    "MLP": mlp.predict_proba(scaler.transform(Xte))[:,1],
    "GCN (Layer 2)": gcn_score_te,
    "FraudScore alone": test["fraud_score"].to_numpy(),
}

# ---- Figure: ROC + PR curves ----
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for name, sc in scores.items():
    fpr, tpr, _ = roc_curve(yte, sc)
    a = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={a:.3f})", linewidth=1.3)
axes[0].plot([0,1],[0,1],'k--',linewidth=0.7)
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC (X test set)"); axes[0].legend(fontsize=6.5, loc="lower right")

for name, sc in scores.items():
    prec, rec, _ = precision_recall_curve(yte, sc)
    ap = auc(rec, prec)
    axes[1].plot(rec, prec, label=f"{name} (AP={ap:.3f})", linewidth=1.3)
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall (X test set)"); axes[1].legend(fontsize=6.5, loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(PAPER_DIR, "combo_roc_pr.png"), dpi=200)
plt.close()
print("Saved combo_roc_pr.png")

# ---- Figure: threshold sweep (recompute) + leave-one-campaign-out bars ----
val = df[df.split=="val"]
from sklearn.metrics import f1_score
taus = np.arange(0.05, 0.85, 0.01)
f1s = [f1_score(val.label, (val.fraud_score>=t).astype(int)) for t in taus]
with open(os.path.join(BASE,"layer1_results.json")) as f: L1=json.load(f)
with open(os.path.join(BASE,"leave_one_out_results.json")) as f: LOO=json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(9,4))
axes[0].plot(taus, f1s, color="tab:blue")
axes[0].axvspan(L1["plateau_lo"], L1["plateau_hi"], color="tab:blue", alpha=0.15, label="within 0.005 of max")
axes[0].axvline(L1["deployed_tau"], color="tab:red", linestyle="--", label=f"deployed τ={L1['deployed_tau']}")
axes[0].set_xlabel("FraudScore gate threshold τ"); axes[0].set_ylabel("Validation F1")
axes[0].set_title("Layer-1 threshold sweep (X)"); axes[0].legend(fontsize=7)

camps = list(LOO["held_out"].keys())
labels = ["Fake\nfollowers","Election\nspambots\n(2014)","Hashtag-promo\nspambots","Product-ad\nspambots"]
in_dist = [LOO["in_distribution_control"][c] for c in camps]
held_out = [LOO["held_out"][c]["held_out_campaign_recall"] for c in camps]
x = np.arange(len(camps)); w=0.35
axes[1].bar(x-w/2, in_dist, w, label="In-distribution", color="tab:green")
axes[1].bar(x+w/2, held_out, w, label="Held out (unseen campaign)", color="tab:red")
axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=7.5)
axes[1].set_ylabel("Recall on that campaign")
axes[1].set_title("Leave-one-campaign-out generalization")
axes[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(PAPER_DIR, "combo_thresh_loo.png"), dpi=200)
plt.close()
print("Saved combo_thresh_loo.png")

# ---- Figure: feature importance + degree distribution ----
with open(os.path.join(BASE,"baseline_results.json")) as f: BR=json.load(f)
edges_global = np.load(os.path.join(BASE, "edges_global.npy"))
N = len(df)
deg = np.zeros(N, dtype=np.int64)
for a,b in edges_global:
    deg[a]+=1; deg[b]+=1
deg_nonzero = deg[deg>0]

fig, axes = plt.subplots(1, 2, figsize=(9,4))
labels_sig = ["FriendFollower\nImbalance","AccountAge\nScore","PostingRate\nAnomaly","Engagement\nVoid*","Profile\nVoid","SocialProof\nVoid"]
rf_imp = [BR["rf_importance"][c] for c in SIGCOLS]
xg_imp = [BR["xg_importance"][c] for c in SIGCOLS]
x = np.arange(6); w=0.35
axes[0].bar(x-w/2, rf_imp, w, label="Random Forest", color="tab:blue")
axes[0].bar(x+w/2, xg_imp, w, label="XGBoost", color="tab:orange")
axes[0].set_xticks(x); axes[0].set_xticklabels(labels_sig, fontsize=6.8, rotation=20, ha="right")
axes[0].set_ylabel("Importance"); axes[0].set_title("Feature importance (X)\n* = strongest real discriminator")
axes[0].legend(fontsize=7)
axes[0].margins(x=0.02)

axes[1].hist(deg_nonzero, bins=60, color="tab:purple", alpha=0.8)
axes[1].set_xlabel("Node degree (fingerprint edges)"); axes[1].set_ylabel("Number of accounts")
axes[1].set_title(f"Degree distribution\n({len(deg_nonzero)} non-isolated gated accounts)")
plt.tight_layout()
plt.savefig(os.path.join(PAPER_DIR, "combo_feat_degree.png"), dpi=200)
plt.close()
print("Saved combo_feat_degree.png")

# ---- Figure: cross-platform (Instagram) comparison ----
with open(os.path.join(os.path.dirname(BASE), "instafake", "instagram_results.json")) as f:
    IG = json.load(f)
fig, axes = plt.subplots(1, 2, figsize=(9,4))
models_ig = [r["model"] for r in IG["results"]]
f1_ig = [r["f1"] for r in IG["results"]]
auc_ig = [r["auc"] for r in IG["results"]]
x = np.arange(len(models_ig)); w=0.35
axes[0].bar(x-w/2, f1_ig, w, label="F1", color="tab:green")
axes[0].bar(x+w/2, auc_ig, w, label="AUC", color="tab:cyan")
axes[0].set_xticks(x); axes[0].set_xticklabels(models_ig, fontsize=7, rotation=20)
axes[0].set_ylim(0,1); axes[0].set_title("Instagram cross-platform test results"); axes[0].legend(fontsize=7)

sig_names = ["FriendFollower\nImbalance","PostingRate\nAnomaly","Engagement\nVoid","Profile\nVoid","SocialProof\nVoid"]
dirs = [IG["directions"][c] for c in ["s_ffr","s_pr","s_eng","s_prof","s_social"]]
aucs_cal = [IG["aucs"][c] for c in ["s_ffr","s_pr","s_eng","s_prof","s_social"]]
colors = ["tab:red" if d==-1 else "tab:blue" for d in dirs]
axes[1].bar(np.arange(5), aucs_cal, color=colors)
axes[1].set_xticks(np.arange(5)); axes[1].set_xticklabels(sig_names, fontsize=6.8, rotation=15, ha="right")
axes[1].set_ylabel("Calibrated validation AUC")
axes[1].set_title("Instagram signal transfer\n(red = direction flipped vs. X)")
axes[1].axhline(0.5, color="k", linestyle=":", linewidth=0.7)
plt.tight_layout()
plt.savefig(os.path.join(PAPER_DIR, "combo_crossplatform.png"), dpi=200)
plt.close()
print("Saved combo_crossplatform.png")
