import pandas as pd
import numpy as np
import os, json
from sklearn.metrics import normalized_mutual_info_score

BASE = os.path.dirname(__file__)
df = pd.read_pickle(os.path.join(BASE, "features_split.pkl")).reset_index(drop=True)
comm_of = np.load(os.path.join(BASE, "comm_of.npy"))
df["comm"] = comm_of

bots = df[df.label==1].copy()
print("Bot campaigns and sizes:")
print(bots.campaign.value_counts())

# For each real campaign (= a real coordinated "ring"), what fraction of its members
# fall in the single Louvain community that captures the largest share of that campaign?
recovery = {}
for camp, g in bots.groupby("campaign"):
    g_comm = g[g.comm != -1]  # exclude isolated (non-gated) members
    if len(g_comm)==0:
        recovery[camp] = dict(n=len(g), n_in_graph=0, recovered_frac=0.0, recovered_ge_70=False)
        continue
    top_comm, top_n = g_comm.comm.value_counts().idxmax(), g_comm.comm.value_counts().max()
    frac = top_n / len(g)  # denominator = ALL campaign members (isolated ones count against recovery)
    recovery[camp] = dict(n=int(len(g)), n_in_graph=int(len(g_comm)),
                           dominant_community=int(top_comm), members_in_dominant=int(top_n),
                           recovered_frac=round(float(frac),4), recovered_ge_70=bool(frac>=0.70))

print("\nPer-campaign ('ring') recovery (>=70% criterion, mirroring the original R(k)):")
for k,v in recovery.items():
    print(k, v)

# Global clustering quality: NMI between Louvain community assignment and true campaign
# label, restricted to bot accounts that received a community (i.e. were gated & non-isolated)
bots_graph = bots[bots.comm != -1]
nmi = normalized_mutual_info_score(bots_graph.campaign, bots_graph.comm)
print(f"\nNMI(Louvain community, true campaign) over {len(bots_graph)} non-isolated bot accounts: {nmi:.4f}")

# Also: fraction of each campaign that even survives the Layer-1 gate (entered the graph at all)
gate_survival = bots.groupby("campaign")["gated"].mean().round(4).to_dict()
print("\nFraction of each campaign surviving the Layer-1 gate:", gate_survival)

out = dict(recovery=recovery, nmi=float(nmi), gate_survival=gate_survival,
           n_bot_accounts=int(len(bots)), n_bot_in_graph=int(len(bots_graph)))
with open(os.path.join(BASE, "ring_recovery_results.json"), "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved ring_recovery_results.json")
