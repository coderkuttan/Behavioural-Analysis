import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(5.6, 7.2))
ax.set_xlim(0, 11.4); ax.set_ylim(1.5, 11.0); ax.axis("off")

def box(cx, cy, w, h, text, fc="#dfe9f5", ec="#2b5b84", fs=9.2):
    x, y = cx - w/2, cy - h/2
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                        linewidth=1.2, edgecolor=ec, facecolor=fc)
    ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, linespacing=1.4)
    return (cx, y + h, cx, y)  # (top_x, top_y, bottom_x, bottom_y)

def varrow(x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                         linewidth=1.2, color="#333333", shrinkA=0, shrinkB=0)
    ax.add_patch(a)

# --- boxes, laid out top to bottom, one branch point ---
b1 = box(5.5, 10.1, 5.0, 1.0, "All accounts\n(X or Instagram)",
         fc="#f2f2f2", ec="#666666")
b2 = box(5.5, 8.6, 7.4, 1.2, "Layer 1: FraudScore\n(6 behavioral signals,\nvalidation-fit weights)",
         fc="#dfe9f5", ec="#2b5b84")
b3 = box(5.5, 7.1, 2.6, 1.0, "Gate $\\tau$", fc="#eef6e9", ec="#3f7d34")

b3a = box(2.4, 5.5, 4.4, 1.2, "Below $\\tau$:\npredicted genuine\n(no further processing)",
          fc="#f7f2df", ec="#8a6d1f", fs=8.6)
b3b = box(8.6, 5.5, 4.4, 1.2, "At/above $\\tau$: gated accounts\nenter behavioral-fingerprint graph",
          fc="#f7dedf", ec="#8a6d1f", fs=8.6)

b4 = box(8.6, 3.9, 5.2, 1.4, "Layer 2: 2-layer GCN classification\n+ Louvain over the same graph\n$\\rightarrow$ ring recovery (Table IV)",
         fc="#f0e3f5", ec="#6a3f84", fs=8.6)
b5 = box(8.6, 2.4, 5.2, 1.0, "Per-account label +\ncoordinated-group attribution",
         fc="#eaeaea", ec="#555555", fs=8.8)

# --- arrows, anchored to box edges only ---
varrow(5.5, b1[3], 5.5, b2[1])            # b1 -> b2
varrow(5.5, b2[3], 5.5, b3[1])            # b2 -> b3
varrow(4.4, b3[3], 2.4, b3a[1])           # b3 -> b3a (left branch)
varrow(6.6, b3[3], 8.6, b3b[1])           # b3 -> b3b (right branch)
varrow(8.6, b3b[3], 8.6, b4[1])           # b3b -> b4
varrow(8.6, b4[3], 8.6, b5[1])            # b4 -> b5

plt.tight_layout()
plt.savefig(r"F:\christ\sem 5\research paper\fig1_architecture_new.png", dpi=220)
print("saved")
