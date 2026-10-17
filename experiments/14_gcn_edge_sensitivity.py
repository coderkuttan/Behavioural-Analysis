"""GCN sensitivity to the edge rule (reviewer comment 5).

Retrains the exact GCN of 07_gcn_louvain.py (same architecture, optimiser, seed,
early stopping) on graphs built with a different tolerance theta / edge cut-off.
GCNConv(cached=True) reuses the normalised adjacency between epochs; on a fixed
graph this gives the same result as the original and only saves time.
Usage: python 14_gcn_edge_sensitivity.py <theta> <cutoff>
Outputs: experiments/gcn_sensitivity_<theta>_<cutoff>.json
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

theta, cutoff = float(sys.argv[1]), float(sys.argv[2])

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cresci")
OUT = os.path.dirname(os.path.abspath(__file__))
SIGCOLS = ["s_ffr", "s_age", "s_pr", "s_eng", "s_prof", "s_social"]
EDGE_SIGS = ["s_ffr", "s_age", "s_pr", "s_eng"]
df = pd.read_pickle(os.path.join(BASE, "features_split.pkl")).reset_index(drop=True)
with open(os.path.join(BASE, "graph_stats.json")) as f:
    edge_w = json.load(f)["edge_weights"]  # same validation-derived, capped weights
w_arr = np.array([edge_w[c] for c in EDGE_SIGS], dtype=np.float32)

gated_idx = df.index[df.gated == 1].to_numpy()
m = len(gated_idx)
Xg = df.loc[gated_idx, EDGE_SIGS].to_numpy(dtype=np.float32)
pi, pj = [], []
for s in range(0, m, 800):
    e = min(s + 800, m)
    acc = np.zeros((e - s, m), dtype=np.float32)
    for k in range(4):
        acc += (np.abs(Xg[s:e, k:k + 1] - Xg[:, k].reshape(1, -1)) < theta).astype(np.float32) * w_arr[k]
    li, lj = np.nonzero(acc >= cutoff)
    gi = li + s
    keep = lj > gi
    pi.append(gi[keep]); pj.append(lj[keep])
edges = np.stack([gated_idx[np.concatenate(pi)], gated_idx[np.concatenate(pj)]], axis=1)
print(f"theta={theta} cutoff={cutoff}: {len(edges):,} edges", flush=True)

N = len(df)
X = torch.tensor(df[SIGCOLS].to_numpy(), dtype=torch.float32)
y = torch.tensor(df["label"].to_numpy(), dtype=torch.long)
edge_index = torch.tensor(np.concatenate([edges, edges[:, ::-1]], axis=0).T.copy(), dtype=torch.long)
del edges
train_mask = torch.tensor((df.split == "train").to_numpy())
val_mask = torch.tensor((df.split == "val").to_numpy())
test_mask = torch.tensor((df.split == "test").to_numpy())
data = Data(x=X, edge_index=edge_index, y=y)


class GCN(torch.nn.Module):
    def __init__(self, in_dim=6, h1=64, h2=32, out_dim=2, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, h1, add_self_loops=True, cached=True)
        self.conv2 = GCNConv(h1, h2, add_self_loops=True, cached=True)
        self.lin = torch.nn.Linear(h2, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return F.log_softmax(self.lin(x), dim=1)


torch.manual_seed(42)
model = GCN()
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def f1_of(mask, logits):
    return precision_recall_fscore_support(y[mask].numpy(), logits[mask].argmax(1).numpy(),
                                           average="binary", zero_division=0)[2]


best_val_f1, best_state, patience, wait = -1, None, 20, 0
t0 = time.time()
for epoch in range(150):
    model.train(); opt.zero_grad()
    out = model(data.x, data.edge_index)
    F.nll_loss(out[train_mask], y[train_mask]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        vf1 = f1_of(val_mask, model(data.x, data.edge_index))
    if vf1 > best_val_f1:
        best_val_f1, best_state, wait = vf1, {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        wait += 1
    print(f"epoch {epoch} val_f1={vf1:.4f} ({time.time()-t0:.0f}s)", flush=True)
    if wait >= patience:
        break
train_time = time.time() - t0

model.load_state_dict(best_state); model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
probs, preds = out.exp()[:, 1].numpy(), out.argmax(1).numpy()
tm = test_mask.numpy(); yy = y.numpy()
p, r, f1, _ = precision_recall_fscore_support(yy[tm], preds[tm], average="binary", zero_division=0)
tn, fp, fn, tp = confusion_matrix(yy[tm], preds[tm]).ravel()
res = dict(theta=theta, cutoff=cutoff, n_edges=int(edge_index.shape[1] // 2),
           epochs=epoch + 1, train_time_s=round(train_time, 1), best_val_f1=round(float(best_val_f1), 4),
           test=dict(precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4),
                     auc=round(roc_auc_score(yy[tm], probs[tm]), 4), fpr=round(fp / (fp + tn), 4)))
print(res)
with open(os.path.join(OUT, f"gcn_sensitivity_{theta}_{cutoff}.json"), "w") as f:
    json.dump(res, f, indent=2)
