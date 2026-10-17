import pandas as pd
import numpy as np
import os, json, time
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import xgboost as xgb

BASE = os.path.dirname(__file__)
df = pd.read_pickle(os.path.join(BASE, "features_split.pkl")).reset_index(drop=True)
SIGCOLS = ["s_ffr","s_age","s_pr","s_eng","s_prof","s_social"]

train = df[df.split=="train"]
test = df[df.split=="test"]
Xtr, ytr = train[SIGCOLS].to_numpy(), train.label.to_numpy()
Xte, yte = test[SIGCOLS].to_numpy(), test.label.to_numpy()
print(f"train n={len(train)} (pos={ytr.sum()}), test n={len(test)} (pos={yte.sum()})")

def eval_model(name, ypred, yscore, ytrue):
    p,r,f1,_ = precision_recall_fscore_support(ytrue, ypred, average="binary", zero_division=0)
    auc = roc_auc_score(ytrue, yscore)
    tn,fp,fn,tp = confusion_matrix(ytrue, ypred).ravel()
    fpr = fp/(fp+tn)
    res = dict(model=name, precision=round(p,4), recall=round(r,4), f1=round(f1,4),
               auc=round(auc,4), fpr=round(fpr,4))
    print(res)
    return res

results = []

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5,
                             min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(Xtr, ytr)
rf_pred = rf.predict(Xte)
rf_score = rf.predict_proba(Xte)[:,1]
results.append(eval_model("Random Forest", rf_pred, rf_score, yte))
rf_importance = dict(zip(SIGCOLS, rf.feature_importances_.round(4).tolist()))

# XGBoost
xg = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=-1)
xg.fit(Xtr, ytr)
xg_pred = xg.predict(Xte)
xg_score = xg.predict_proba(Xte)[:,1]
results.append(eval_model("XGBoost", xg_pred, xg_score, yte))
xg_importance = dict(zip(SIGCOLS, xg.feature_importances_.round(4).tolist()))

# MLP
scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
mlp = MLPClassifier(hidden_layer_sizes=(64,32), activation="relu", solver="adam",
                     alpha=0.001, batch_size=64, learning_rate_init=0.001, max_iter=500,
                     early_stopping=True, validation_fraction=0.1, random_state=42)
mlp.fit(Xtr_s, ytr)
mlp_pred = mlp.predict(Xte_s)
mlp_score = mlp.predict_proba(Xte_s)[:,1]
results.append(eval_model("MLP", mlp_pred, mlp_score, yte))

# Isolation Forest (contamination = true train prevalence)
contam = ytr.mean()
isof = IsolationForest(n_estimators=200, contamination=min(contam,0.5), random_state=42, n_jobs=-1)
isof.fit(Xtr)
isof_raw = isof.predict(Xte)  # 1 normal, -1 anomaly
isof_pred = (isof_raw == -1).astype(int)
isof_score = -isof.score_samples(Xte)
results.append(eval_model("Isolation Forest", isof_pred, isof_score, yte))

# One-Class SVM (fit on genuine TRAIN only)
genuine_train = Xtr[ytr==0]
oc_scaler = StandardScaler().fit(genuine_train)
ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
ocsvm.fit(oc_scaler.transform(genuine_train))
oc_raw = ocsvm.predict(oc_scaler.transform(Xte))
oc_pred = (oc_raw == -1).astype(int)
oc_score = -ocsvm.decision_function(oc_scaler.transform(Xte))
results.append(eval_model("One-Class SVM", oc_pred, oc_score, yte))

# Layer 1 fixed-weight FraudScore alone at deployed tau
with open(os.path.join(BASE, "layer1_results.json")) as f:
    L1 = json.load(f)
tau = L1["deployed_tau"]
fs_pred = (test["fraud_score"] >= tau).astype(int).to_numpy()
results.append(eval_model(f"FraudScore alone (tau={tau})", fs_pred, test["fraud_score"].to_numpy(), yte))

with open(os.path.join(BASE, "baseline_results.json"), "w") as f:
    json.dump(dict(results=results, rf_importance=rf_importance, xg_importance=xg_importance,
                    contamination=float(contam)), f, indent=2)

# save trained model artifacts needed downstream
import pickle
with open(os.path.join(BASE, "models.pkl"), "wb") as f:
    pickle.dump(dict(rf=rf, xg=xg, mlp=mlp, scaler=scaler), f)

print("\nSaved baseline_results.json and models.pkl")
