import pandas as pd
import numpy as np
import os, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

BASE = os.path.dirname(__file__)
df = pd.read_pickle(os.path.join(BASE, "features_split.pkl")).reset_index(drop=True)
SIGCOLS = ["s_ffr","s_age","s_pr","s_eng","s_prof","s_social"]

genuine = df[df.label==0]
genuine_test = df[(df.label==0) & (df.split=="test")]
campaigns = [c for c in df.campaign.unique() if c != "genuine"]

results = {}
for held_out in campaigns:
    train_pool = df[(df.label==0) & (df.split.isin(["train","val"]))]
    train_bots = df[(df.label==1) & (df.split.isin(["train","val"])) & (df.campaign != held_out)]
    train = pd.concat([train_pool, train_bots])
    test_bots = df[df.campaign == held_out]  # ALL of this campaign's accounts, unseen during training
    test = pd.concat([genuine_test, test_bots])

    Xtr, ytr = train[SIGCOLS].to_numpy(), train.label.to_numpy()
    Xte, yte = test[SIGCOLS].to_numpy(), test.label.to_numpy()

    rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5,
                                 min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    score = rf.predict_proba(Xte)[:,1]
    p,r,f1,_ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
    auc = roc_auc_score(yte, score)

    # recall specifically on the held-out campaign's own accounts
    campaign_mask = (test.campaign == held_out).to_numpy()
    campaign_recall = pred[campaign_mask].mean()  # fraction of held-out-campaign accounts flagged fraud

    results[held_out] = dict(
        n_train=len(train), n_train_bots=len(train_bots),
        n_held_out_campaign=int(campaign_mask.sum()),
        overall_precision=round(p,4), overall_recall=round(r,4), overall_f1=round(f1,4), overall_auc=round(auc,4),
        held_out_campaign_recall=round(float(campaign_recall),4)
    )
    print(held_out, "->", results[held_out])

# in-distribution control: same pipeline but campaign included in training (upper bound)
Xtr_all = df[(df.label==1) & (df.split.isin(["train","val"]))]
Xtr_full = pd.concat([df[(df.label==0)&(df.split.isin(["train","val"]))], Xtr_all])
rf_full = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5,
                                  min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)
rf_full.fit(Xtr_full[SIGCOLS].to_numpy(), Xtr_full.label.to_numpy())
control = {}
for camp in campaigns:
    camp_df = df[df.campaign==camp]
    pred = rf_full.predict(camp_df[SIGCOLS].to_numpy())
    control[camp] = round(float(pred.mean()),4)
print("\nIn-distribution control recall per campaign (campaign included in training):", control)

with open(os.path.join(BASE, "leave_one_out_results.json"), "w") as f:
    json.dump(dict(held_out=results, in_distribution_control=control), f, indent=2)
print("\nSaved leave_one_out_results.json")
