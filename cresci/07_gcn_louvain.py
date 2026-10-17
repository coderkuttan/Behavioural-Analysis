import pandas as pd
import numpy as np
import os, json, time
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

BASE = os.path.dirname(__file__)
df = pd.read_pickle(os.path.join(BASE, "features_split.pkl")).reset_index(drop=True)
SIGCOLS = ["s_ffr","s_age","s_pr","s_eng","s_prof","s_social"]
edges_global = np.load(os.path.join(BASE, "edges_global.npy"))

N = len(df)
X = torch.tensor(df[SIGCOLS].to_numpy(), dtype=torch.float32)
y = torch.tensor(df["label"].to_numpy(), dtype=torch.long)

ei = np.concatenate([edges_global, edges_global[:, ::-1]], axis=0).T  # undirected
edge_index = torch.tensor(ei, dtype=torch.long)

train_mask = torch.tensor((df.split=="train").to_numpy())
val_mask = torch.tensor((df.split=="val").to_numpy())
test_mask = torch.tensor((df.split=="test").to_numpy())

data = Data(x=X, edge_index=edge_index, y=y)
print(f"Graph: {N} nodes, {edge_index.shape[1]} directed edges (incl. both directions)")
print(f"train={train_mask.sum().item()} val={val_mask.sum().item()} test={test_mask.sum().item()}")

class GCN(torch.nn.Module):
    def __init__(self, in_dim=6, h1=64, h2=32, out_dim=2, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, h1, add_self_loops=True)
        self.conv2 = GCNConv(h1, h2, add_self_loops=True)
        self.lin = torch.nn.Linear(h2, out_dim)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

torch.manual_seed(42)
model = GCN()
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def f1_of(mask, logits):
    pred = logits[mask].argmax(1).numpy()
    true = y[mask].numpy()
    p,r,f1,_ = precision_recall_fscore_support(true, pred, average="binary", zero_division=0)
    return f1

best_val_f1, best_state, patience, wait = -1, None, 20, 0
t0=time.time()
for epoch in range(150):
    model.train()
    opt.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[train_mask], y[train_mask])
    loss.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        vf1 = f1_of(val_mask, out)
    if vf1 > best_val_f1:
        best_val_f1, best_state, wait = vf1, {k:v.clone() for k,v in model.state_dict().items()}, 0
    else:
        wait += 1
    if epoch % 10 == 0:
        print(f"epoch {epoch}: loss={loss.item():.4f} val_f1={vf1:.4f} best={best_val_f1:.4f}")
    if wait >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
print(f"GCN training time: {time.time()-t0:.1f}s, best val F1={best_val_f1:.4f}")

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    probs = out.exp()[:,1].numpy()
    preds = out.argmax(1).numpy()

yte = y[test_mask].numpy()
pte = preds[test_mask.numpy()]
ste = probs[test_mask.numpy()]
p,r,f1,_ = precision_recall_fscore_support(yte, pte, average="binary", zero_division=0)
auc = roc_auc_score(yte, ste)
tn,fp,fn,tp = confusion_matrix(yte, pte).ravel()
fpr = fp/(fp+tn)
gcn_result = dict(model="GCN (Layer 2)", precision=round(p,4), recall=round(r,4), f1=round(f1,4),
                   auc=round(auc,4), fpr=round(fpr,4))
print("GCN test result:", gcn_result)

# isolated node stats within test mask
deg = np.zeros(N, dtype=np.int64)
for a,b in edges_global:
    deg[a]+=1; deg[b]+=1
iso_in_test = int(((deg==0) & test_mask.numpy()).sum())
print(f"Isolated nodes in test mask: {iso_in_test} / {test_mask.sum().item()} ({iso_in_test/test_mask.sum().item()*100:.1f}%)")

# GCN performance restricted to non-isolated test nodes only, vs isolated-only
non_iso_test = test_mask.numpy() & (deg>0)
iso_test = test_mask.numpy() & (deg==0)
def sub_eval(mask):
    yy = y.numpy()[mask]; pp = preds[mask]; ss = probs[mask]
    if len(np.unique(yy))<2 or mask.sum()==0:
        return None
    p,r,f1,_ = precision_recall_fscore_support(yy, pp, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(yy, ss)
    except Exception:
        auc = float("nan")
    return dict(n=int(mask.sum()), precision=round(p,4), recall=round(r,4), f1=round(f1,4), auc=round(auc,4))

gcn_noniso = sub_eval(non_iso_test)
gcn_iso = sub_eval(iso_test)
print("GCN on non-isolated test nodes:", gcn_noniso)
print("GCN on isolated test nodes:", gcn_iso)

# ---- Louvain community detection over the same graph ----
G = nx.Graph()
G.add_nodes_from(range(N))
G.add_edges_from(edges_global.tolist())
t0=time.time()
communities = nx.community.louvain_communities(G, seed=42, resolution=1.0)
print(f"Louvain: {len(communities)} communities, time={time.time()-t0:.1f}s")
comm_of = np.full(N, -1, dtype=np.int64)
for cid, c in enumerate(communities):
    for node in c:
        comm_of[node] = cid

# label each community by majority FraudScore-gate classification among its TRAIN nodes only
gated = df["gated"].to_numpy()
train_arr = train_mask.numpy()
comm_label = {}
for cid in range(len(communities)):
    members_train = np.where((comm_of==cid) & train_arr)[0]
    if len(members_train)==0:
        comm_label[cid] = 0  # default genuine if no train evidence
        continue
    comm_label[cid] = int(gated[members_train].mean() >= 0.5)
# isolated nodes (comm_of == -1, i.e. never got a community since no edges) -> use own gate label
louvain_pred = np.array([comm_label.get(c, None) if c!=-1 else None for c in comm_of])
iso_mask_all = (deg==0)
louvain_pred = np.where(iso_mask_all, gated, [comm_label[c] for c in comm_of])

lyte = y.numpy()[test_mask.numpy()]
lpte = louvain_pred[test_mask.numpy()].astype(int)
p,r,f1,_ = precision_recall_fscore_support(lyte, lpte, average="binary", zero_division=0)
tn,fp,fn,tp = confusion_matrix(lyte, lpte).ravel()
fpr = fp/(fp+tn)
# use fraud_score as continuous score proxy for AUC (community label is binary only)
louvain_score = df["fraud_score"].to_numpy()[test_mask.numpy()]
auc = roc_auc_score(lyte, louvain_score)
louvain_result = dict(model="Louvain", precision=round(p,4), recall=round(r,4), f1=round(f1,4),
                       auc=round(auc,4), fpr=round(fpr,4))
print("Louvain test result:", louvain_result)

agree = (lpte == pte).mean()
print(f"Louvain vs GCN prediction agreement on test set: {agree*100:.2f}%")

out = dict(gcn_result=gcn_result, gcn_noniso=gcn_noniso, gcn_iso=gcn_iso,
           iso_in_test=iso_in_test, n_test=int(test_mask.sum().item()),
           louvain_result=louvain_result, n_communities=len(communities),
           louvain_gcn_agreement=float(agree), best_val_f1=float(best_val_f1))
with open(os.path.join(BASE, "layer2_results.json"), "w") as f:
    json.dump(out, f, indent=2)

np.save(os.path.join(BASE, "comm_of.npy"), comm_of)
np.save(os.path.join(BASE, "gcn_preds.npy"), preds)
np.save(os.path.join(BASE, "gcn_probs.npy"), probs)
torch.save(model.state_dict(), os.path.join(BASE, "gcn_model.pt"))
print("\nSaved layer2_results.json, comm_of.npy, gcn_preds.npy, gcn_probs.npy, gcn_model.pt")
