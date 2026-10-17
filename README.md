# Behavioral-Fingerprint GCN for Coordinated Bot Detection

Code for the paper *"Behavioral-Fingerprint Graph Convolutional Networks for Coordinated Bot
Detection: A Two-Stage Framework Validated on Real Cross-Platform Campaigns"*
(A. James, Logeshwaran J, Sudhakar T — IC-ISHCT 2026).

## Pipeline

| Script | What it does | Main output |
|---|---|---|
| `cresci/01_load_eda.py` | Loads the five Cresci-2017 user files (genuine + 4 campaigns) | `merged_raw.pkl` |
| `cresci/02_features_eda.py` | Raw per-account fields | `merged_features_raw.pkl` |
| `cresci/03_build_features.py` | Six behavioral signals in [0,1] | `features_final.pkl`, `thresholds.json` |
| `cresci/04_split_and_layer1.py` | 70/10/20 split stratified by label+campaign; validation-fitted FraudScore weights; gate sweep, τ = 0.32 | `features_split.pkl`, `layer1_results.json` |
| `cresci/05_build_graph.py` | Fingerprint graph over gated accounts (run with `EDGE_THETA=0.04`) | `edges_global.npy`, `graph_stats.json` |
| `cresci/06_models.py` | RF, XGBoost, MLP, Isolation Forest, One-Class SVM, FraudScore baseline | `baseline_results.json`, `models.pkl` |
| `cresci/07_gcn_louvain.py` | 2-layer GCN + Louvain on the same graph | `layer2_results.json`, `comm_of.npy`, `gcn_preds.npy` |
| `cresci/08_campaign_recovery.py` | Campaign (ring) recovery, NMI | `ring_recovery_results.json` |
| `cresci/09_leave_one_campaign_out.py` | Leave-one-campaign-out generalization | `leave_one_out_results.json` |
| `cresci/10_make_figures.py`, `11_architecture_fig.py` | Paper figures | `combo_*.png`, `fig1_architecture_new.png` |
| `instafake/eda.py`, `instafake/build_and_eval.py` | Cross-platform run on InstaFake (60/20/20 stratified) | `instagram_results.json` |
| `experiments/12_edge_threshold_sweep.py` | Edge-threshold sensitivity (revision) | `edge_sweep_results.json` |
| `experiments/13_isolated_node_remedies.py` | Isolated-node analysis and RF fallback (revision) | `isolated_remedies_results.json` |
| `experiments/14_gcn_edge_sensitivity.py <theta> <cutoff>` | GCN retrained on an alternative edge rule (revision) | `gcn_sensitivity_<theta>_<cutoff>.json` |

Run the Cresci scripts in numeric order, then the InstaFake scripts. Every script reads and
writes in its own folder.

## Data (not redistributed)

* **Cresci-2017**: download `datasets_full.csv` from the Bot Repository
  (https://botometer.osome.iu.edu/bot-repository/datasets.html). Extract `users.csv` from
  `genuine_accounts`, `fake_followers`, `social_spambots_1/2/3` into `cresci/users/` as
  `<name>_users.csv`.
* **InstaFake**: download from https://github.com/fcakyon/instafake-dataset and place
  `automatedAccountData.json` and `nonautomatedAccountData.json` in `instafake/`.

## Reproducibility

* Random seed **42** everywhere (`train_test_split`, all scikit-learn/XGBoost models,
  `torch.manual_seed`, Louvain `seed`).
* Environment used for the paper: Windows 11, CPU only (20 logical cores, 16 GB RAM),
  Python 3.13.5, scikit-learn 1.8.0, XGBoost 3.2.0, PyTorch 2.10.0 (CPU), PyTorch Geometric
  2.7.0, NetworkX 3.6.1, NumPy 2.2.0, pandas 3.0.1, matplotlib 3.10.8. See `requirements.txt`.
* Run times on that machine: graph construction ≈ 2 s, Louvain ≈ 78 s, GCN training ≈ 60 min
  (32 epochs, early stopping, 25.6 M directed edges).
