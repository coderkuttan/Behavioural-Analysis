"""Edge-threshold sensitivity sweep (reviewer comment 5).

Rebuilds the fingerprint graph of 05_build_graph.py for several proximity
tolerances theta (and one stricter edge cut-off), then reruns Louvain and the
campaign-recovery analysis of 07/08 on each graph. Edge rule, weights, gate and
seed are identical to the published run; only theta / cut-off change.
Outputs: experiments/edge_sweep_results.json
"""
import os, json, time, gc
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cresci")
OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 42

df = pd.read_pickle(os.path.join(BASE, "features_split.pkl")).reset_index(drop=True)
with open(os.path.join(BASE, "layer1_results.json")) as f:
    L1 = json.load(f)

EDGE_SIGS = ["s_ffr", "s_age", "s_pr", "s_eng"]


def cap_and_renorm(w, cap=0.35, iters=5):
    w = dict(w)
    for _ in range(iters):
        over = {k: v for k, v in w.items() if v > cap}
        if not over:
            break
        excess = sum(v - cap for v in over.values())
        for k in over:
            w[k] = cap
        under = {k: v for k, v in w.items() if v < cap}
        usum = sum(under.values())
        for k in under:
            w[k] += excess * (under[k] / usum)
    return w


raw_w = {c: max(L1["aucs"][c] - 0.5, 0.0) for c in EDGE_SIGS}
s = sum(raw_w.values())
edge_w = cap_and_renorm({c: raw_w[c] / s for c in EDGE_SIGS})
w_arr = np.array([edge_w[c] for c in EDGE_SIGS], dtype=np.float32)

gated_idx = df.index[df.gated == 1].to_numpy()
m = len(gated_idx)
X = df.loc[gated_idx, EDGE_SIGS].to_numpy(dtype=np.float32)


def build_edges(theta, cutoff, chunk=800):
    """Same rule as 05_build_graph.py, returned as global node-index pairs (i<j)."""
    t0 = time.time()
    parts_i, parts_j = [], []
    for start in range(0, m, chunk):
        end = min(start + chunk, m)
        acc = np.zeros((end - start, m), dtype=np.float32)
        for k in range(len(EDGE_SIGS)):
            diff = np.abs(X[start:end, k:k + 1] - X[:, k].reshape(1, -1))
            acc += (diff < theta).astype(np.float32) * w_arr[k]
        li, lj = np.nonzero(acc >= cutoff)
        gi = li + start
        keep = lj > gi
        parts_i.append(gi[keep]); parts_j.append(lj[keep])
    ei = np.concatenate(parts_i); ej = np.concatenate(parts_j)
    return np.stack([gated_idx[ei], gated_idx[ej]], axis=1), time.time() - t0


bots_mask = (df.label == 1).to_numpy()
camp = df["campaign"].to_numpy()
CONFIGS = [(0.01, 0.5), (0.02, 0.5), (0.03, 0.5), (0.04, 0.5), (0.05, 0.5),
           (0.04, 0.75)]
MAX_LOUVAIN_EDGES = 16_000_000  # keep networkx within RAM on a 16 GB machine

results = []
for theta, cutoff in CONFIGS:
    edges, t_build = build_edges(theta, cutoff)
    E = len(edges)
    deg = np.bincount(edges.ravel(), minlength=len(df))
    noniso = deg > 0
    n_noniso = int(noniso.sum())
    test = (df.split == "test").to_numpy()
    row = dict(theta=theta, cutoff=cutoff, n_edges=int(E), build_time_s=round(t_build, 1),
               non_isolated_gated=n_noniso,
               isolated_gated=int(m - n_noniso),
               mean_degree=round(float(deg[noniso].mean()), 1) if n_noniso else 0.0,
               median_degree=float(np.median(deg[noniso])) if n_noniso else 0.0,
               density=round(E / (n_noniso * (n_noniso - 1) / 2), 4) if n_noniso > 1 else 0.0,
               isolated_test=int((~noniso & test).sum()),
               edge_index_mb=round(E * 2 * 2 * 8 / 1e6, 1))  # PyG int64, both directions
    print(f"theta={theta} cutoff={cutoff}: {E:,} edges, mean deg {row['mean_degree']}, "
          f"density {row['density']}, build {t_build:.1f}s", flush=True)

    if 0 < E <= MAX_LOUVAIN_EDGES:
        G = nx.Graph()
        G.add_nodes_from(range(len(df)))  # as in 07: isolated accounts are singleton nodes
        G.add_edges_from(edges.tolist())
        t0 = time.time()
        comms = nx.community.louvain_communities(G, seed=SEED, resolution=1.0)
        row["louvain_time_s"] = round(time.time() - t0, 1)
        del G; gc.collect()
        comms = [c for c in comms if len(c) > 1]  # drop the isolated singletons
        comm_of = np.full(len(df), -1, dtype=np.int64)
        for cid, c in enumerate(comms):
            comm_of[list(c)] = cid
        sizes = np.bincount(comm_of[comm_of >= 0])
        row["n_communities"] = int(len(comms))
        row["n_communities_ge10"] = int((sizes >= 10).sum())
        bm = bots_mask & (comm_of >= 0)
        row["nmi_bots_in_graph"] = round(float(normalized_mutual_info_score(camp[bm], comm_of[bm])), 4)
        rec = {}
        for cname in sorted(set(camp[bots_mask])):
            cm = bots_mask & (camp == cname)
            in_g = cm & (comm_of >= 0)
            if in_g.sum() == 0:
                rec[cname] = dict(coverage=0.0, purity=0.0); continue
            vals, cnts = np.unique(comm_of[in_g], return_counts=True)
            top = vals[cnts.argmax()]
            cov = cnts.max() / cm.sum()                    # same as Table IV "Recovered"
            pur = cnts.max() / (comm_of == top).sum()      # share of that community that is this campaign
            rec[cname] = dict(coverage=round(float(cov), 4), purity=round(float(pur), 4),
                              dominant_size=int((comm_of == top).sum()))
        row["recovery"] = rec
        row["campaigns_recovered_ge70"] = int(sum(v["coverage"] >= 0.70 for v in rec.values()))
        print(f"   Louvain: {len(comms)} comms ({row['n_communities_ge10']} with >=10), "
              f"NMI {row['nmi_bots_in_graph']}, recovered {row['campaigns_recovered_ge70']}/4, "
              f"{row['louvain_time_s']}s", flush=True)
    results.append(row)
    del edges; gc.collect()

with open(os.path.join(OUT, "edge_sweep_results.json"), "w") as f:
    json.dump(dict(edge_weights=edge_w, m_gated=int(m), results=results), f, indent=2)
print("Saved edge_sweep_results.json")
