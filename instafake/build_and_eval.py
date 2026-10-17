import pandas as pd
import numpy as np
import os, json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

BASE = os.path.dirname(__file__)
df = pd.read_pickle(os.path.join(BASE, "insta_merged.pkl")).reset_index(drop=True)

# fill missing posting_rate (no usable span) with 0 posts/day floor -> treat as maximally dormant-suspicious later via sign calibration
df["posting_rate"] = df["posting_rate"].fillna(0.0)
df["engagement_rate"] = df["engagement_rate"].fillna(0.0)

SEED = 42
train_idx, temp_idx = train_test_split(df.index, test_size=0.40, stratify=df.label, random_state=SEED)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=df.loc[temp_idx].label, random_state=SEED)
df["split"] = "train"; df.loc[val_idx,"split"]="val"; df.loc[test_idx,"split"]="test"
print(df.groupby(["split","label"]).size())

genuine_train = df[(df.label==0) & (df.split=="train")]

def pct(s,q): return s.dropna().quantile(q)
ffr_cap = pct(genuine_train.friend_follower_ratio, .99)
pr_lo = pct(np.log1p(genuine_train.posting_rate), .10)
pr_hi = pct(np.log1p(genuine_train.posting_rate), .90)
eng_ref = genuine_train.engagement_rate.median()

def s_ffr(d): return np.clip(d.friend_follower_ratio/ffr_cap, 0, 1)
def s_pr(d):
    x = np.log1p(d.posting_rate)
    below = np.clip((pr_lo-x)/max(pr_lo,1e-6),0,1)
    above = np.clip((x-pr_hi)/max(pr_hi,1e-6),0,1)
    return np.clip(np.maximum(below,above),0,1)
def s_eng(d): return np.clip(1 - d.engagement_rate/max(eng_ref,1e-6), 0, 1)
def s_prof(d): return ((1-d.has_url)+(1-d.has_bio)+(1-d.has_highlights))/3.0
def s_social(d): return (d.digit_frac.clip(0,1) + d.comments_disabled_frac.clip(0,1))/2.0

df["s_ffr"]=s_ffr(df); df["s_pr"]=s_pr(df); df["s_eng"]=s_eng(df)
df["s_prof"]=s_prof(df); df["s_social"]=s_social(df)
RAW_SIGCOLS = ["s_ffr","s_pr","s_eng","s_prof","s_social"]

val = df.loc[val_idx]
# calibrate direction + weight per signal using validation AUC (non-circular); flip sign if inverse direction scores higher
directions = {}
aucs = {}
for c in RAW_SIGCOLS:
    a = roc_auc_score(val.label, val[c])
    if a < 0.5:
        directions[c] = -1
        aucs[c] = 1-a
    else:
        directions[c] = 1
        aucs[c] = a
print("\nDirections (1=as-defined suspicious-high, -1=flipped) and calibrated AUC:")
for c in RAW_SIGCOLS: print(f"  {c}: dir={directions[c]:+d} auc={aucs[c]:.4f}")

# final signal = raw or (1-raw) depending on direction, so higher always = more suspicious
for c in RAW_SIGCOLS:
    df[f"f_{c}"] = df[c] if directions[c]==1 else (1-df[c])
FINAL = [f"f_{c}" for c in RAW_SIGCOLS]

raw_w = {c: max(aucs[c.replace('f_','')]-0.5,0) for c in FINAL}
s = sum(raw_w.values())
weights = {c: raw_w[c]/s for c in FINAL}
print("\nWeights:", weights)

df["fraud_score"] = sum(df[c]*weights[c] for c in FINAL)
val = df.loc[val_idx]
best_f1, best_tau = -1, 0.5
for tau in np.arange(0.05,0.9,0.01):
    f1 = f1_score(val.label, (val.fraud_score>=tau).astype(int))
    if f1>best_f1: best_f1, best_tau = f1, tau
print(f"\nLayer-1 best val F1={best_f1:.4f} at tau={best_tau:.2f}")

train = df[df.split=="train"]; test = df[df.split=="test"]
Xtr,ytr = train[FINAL].to_numpy(), train.label.to_numpy()
Xte,yte = test[FINAL].to_numpy(), test.label.to_numpy()

def ev(name, pred, score, y):
    p,r,f1,_ = precision_recall_fscore_support(y,pred,average="binary",zero_division=0)
    auc = roc_auc_score(y,score)
    tn,fp,fn,tp = confusion_matrix(y,pred).ravel()
    fpr = fp/(fp+tn)
    res = dict(model=name, precision=round(p,4), recall=round(r,4), f1=round(f1,4), auc=round(auc,4), fpr=round(fpr,4))
    print(res); return res

results = []
fs_pred = (test.fraud_score>=best_tau).astype(int).to_numpy()
results.append(ev("FraudScore alone", fs_pred, test.fraud_score.to_numpy(), yte))

rf = RandomForestClassifier(n_estimators=200,max_depth=15,min_samples_split=5,min_samples_leaf=2,class_weight="balanced",random_state=42,n_jobs=-1)
rf.fit(Xtr,ytr); results.append(ev("Random Forest", rf.predict(Xte), rf.predict_proba(Xte)[:,1], yte))
rf_imp = dict(zip(FINAL, rf.feature_importances_.round(4).tolist()))

xg = xgb.XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,eval_metric="logloss",random_state=42,n_jobs=-1)
xg.fit(Xtr,ytr); results.append(ev("XGBoost", xg.predict(Xte), xg.predict_proba(Xte)[:,1], yte))
xg_imp = dict(zip(FINAL, xg.feature_importances_.round(4).tolist()))

sc = StandardScaler().fit(Xtr)
mlp = MLPClassifier(hidden_layer_sizes=(64,32),activation="relu",solver="adam",alpha=0.001,batch_size=64,
                     learning_rate_init=0.001,max_iter=500,early_stopping=True,validation_fraction=0.1,random_state=42)
mlp.fit(sc.transform(Xtr),ytr); results.append(ev("MLP", mlp.predict(sc.transform(Xte)), mlp.predict_proba(sc.transform(Xte))[:,1], yte))

out = dict(results=results, rf_importance=rf_imp, xg_importance=xg_imp,
           directions=directions, aucs=aucs, weights=weights,
           best_val_f1=best_f1, best_val_tau=float(best_tau),
           n_train=len(train), n_val=len(val_idx), n_test=len(test))
with open(os.path.join(BASE,"instagram_results.json"),"w") as f:
    json.dump(out, f, indent=2)
print("\nSaved instagram_results.json")
