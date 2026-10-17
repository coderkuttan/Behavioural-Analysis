import pandas as pd
import numpy as np
import os, json, time

BASE = os.path.dirname(__file__)
df = pd.read_pickle(os.path.join(BASE, "features_split.pkl")).reset_index(drop=True)
with open(os.path.join(BASE, "layer1_results.json")) as f:
    L1 = json.load(f)

# edge-rule signals: the 4 genuinely continuous signals only. s_social and s_prof are
# averages of 2-3 booleans (only a handful of achievable values), so an "absolute proximity"
# rule on them produces trivial exact-tie edges between unrelated accounts and inflates
# density; excluding them mirrors the original design's exclusion of the boolean-flavoured
# CrossPlatformVoid signal from its edge rule for the same underlying reason.
EDGE_SIGS = ["s_ffr","s_age","s_pr","s_eng"]
aucs = L1["aucs"]
raw_w = {c: max(aucs[c]-0.5, 0.0) for c in EDGE_SIGS}
s = sum(raw_w.values())
edge_w = {c: raw_w[c]/s for c in EDGE_SIGS}
# cap max weight at 0.35 (renormalize) so no single signal alone can pass the 0.5 edge threshold,
# preserving the "at least two signals must agree" adversarial property from the original design
def cap_and_renorm(w, cap=0.35, iters=5):
    w = dict(w)
    for _ in range(iters):
        over = {k:v for k,v in w.items() if v > cap}
        if not over:
            break
        excess = sum(v-cap for v in over.values())
        for k in over: w[k] = cap
        under = {k:v for k,v in w.items() if v < cap}
        usum = sum(under.values())
        for k in under: w[k] += excess * (under[k]/usum)
    return w
edge_w = cap_and_renorm(edge_w)
print("Edge-rule weights (capped, validation-derived):", edge_w, "sum=", sum(edge_w.values()))
theta = float(os.environ.get("EDGE_THETA", "0.10"))  # uniform proximity tolerance per signal

gated_idx = df.index[df.gated == 1].to_numpy()
m = len(gated_idx)
print(f"Gated candidates m={m} of {len(df)} total ({m/len(df)*100:.1f}%)")

X = df.loc[gated_idx, EDGE_SIGS].to_numpy(dtype=np.float32)
w_arr = np.array([edge_w[c] for c in EDGE_SIGS], dtype=np.float32)

t0 = time.time()
edges_i, edges_j = [], []
CHUNK = 800
for start in range(0, m, CHUNK):
    end = min(start+CHUNK, m)
    block = X[start:end]  # (b, 5)
    acc = np.zeros((end-start, m), dtype=np.float32)
    for k in range(len(EDGE_SIGS)):
        diff = np.abs(block[:, k:k+1] - X[:, k].reshape(1, -1))  # (b, m)
        acc += (diff < theta).astype(np.float32) * w_arr[k]
    # only upper triangle relative to global index to avoid dup + self
    for local_i in range(end-start):
        global_i = start + local_i
        row = acc[local_i]
        js = np.where(row >= 0.5)[0]
        js = js[js > global_i]
        edges_i.extend([global_i]*len(js))
        edges_j.extend(js.tolist())
    if start % 4000 == 0:
        print(f"  processed {end}/{m} rows, elapsed {time.time()-t0:.1f}s, edges so far {len(edges_i)}")

print(f"Total edge construction time: {time.time()-t0:.1f}s")
print(f"Edges found: {len(edges_i)}")

edges_global = np.stack([gated_idx[edges_i], gated_idx[edges_j]], axis=1)
np.save(os.path.join(BASE, "edges_global.npy"), edges_global)

deg = np.zeros(len(df), dtype=np.int64)
for a,b in edges_global:
    deg[a]+=1; deg[b]+=1
non_isolated = int((deg>0).sum())
print(f"Non-isolated nodes: {non_isolated} of {m} gated ({non_isolated/m*100:.1f}%)")
print(f"Mean degree (non-isolated): {deg[deg>0].mean():.1f}, median {np.median(deg[deg>0]):.0f}, max {deg.max()}")
possible_pairs = m*(m-1)/2
print(f"Density (of induced subgraph on {non_isolated} non-isolated nodes): {len(edges_global)/(non_isolated*(non_isolated-1)/2):.4f}")

graph_stats = dict(m_gated=int(m), n_edges=int(len(edges_global)), non_isolated=non_isolated,
                    mean_degree=float(deg[deg>0].mean()), median_degree=float(np.median(deg[deg>0])),
                    max_degree=int(deg.max()), edge_weights=edge_w, theta=theta,
                    density=float(len(edges_global)/(non_isolated*(non_isolated-1)/2)))
with open(os.path.join(BASE, "graph_stats.json"), "w") as f:
    json.dump(graph_stats, f, indent=2)
print("\nSaved edges_global.npy and graph_stats.json")
