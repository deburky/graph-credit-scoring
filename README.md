# Credit Scoring on Graphs

Code for a working paper on graph-based credit-fraud scoring (Denis Burakov and Alessandro Fogli,
in preparation). The experiments compare three graphs over the same borrowers, with the same models
and features:

- **no graph**: node features only;
- **a similarity graph**: exact Euclidean *k*-nearest neighbors in feature space, the construction
  used by earlier GNN credit-scoring work such as NIFTY;
- **an observed graph**: the emergency-contact network of DGraph-Fin, where an edge means that one
  user named another as an emergency contact.

On each graph we train tabular learners on graph-derived features (LightGBM, CatBoost, an MLP), a
GraphSAGE network, a relational GraphSAGE and a graph transformer that see edge direction, contact
type and time, and label-free Deep Graph Infomax embeddings with a logistic-regression head. The graph
transformer's attention weights show which contacts drive a user's score. A global-attention model,
in which every user attends to every other user from node features alone, serves as a learned
similarity graph: a stronger version of the *k*-NN control.

## Data

The experiments use [DGraph-Fin](https://dgraph.xinye.com) (Huang et al., 2022), a graph of
3,700,550 users of the Finvolution lending platform and 4,300,999 emergency-contact edges. Its
positive class, which we call **credit fraud**, is defined by the dataset's authors as users who "do
not repay the loans a long time after the due date and ignore the platform's repeated reminders".

DGraph-Fin is released under a user license that requires registration, so this repository does not
include it. Download `DGraphFin.zip` from the official site, accept the license, and place
`dgraph.npz` in `data/dgraphfin/raw/`.

The repository ships a **synthetic sample** in the same format instead (`data/sample/raw/dgraph.npz`,
6,000 users), written by `scripts/make_sample.py`. Its structure is planted to resemble the real data
(fraudsters name fewer contacts and tend to name each other), so every script runs on it and has
some signal to find. Results on the sample are only a check that the code runs; with about 15
fraudsters in its test split, the metrics carry no information about the methods.

The `.npz` file holds these arrays:

| Key | Shape | Content |
|---|---|---|
| `x` | (n, 17) | node features, `-1` for a missing value |
| `y` | (n,) | `0` normal, `1` credit fraud, `2`/`3` background users with no target |
| `edge_index` | (m, 2) | (source, target): the source named the target as an emergency contact |
| `edge_type` | (m,) | contact type, 1 to 11 |
| `edge_timestamp` | (m,) | desensitized time, 1 to 821 |
| `train_mask`, `valid_mask`, `test_mask` | index arrays | split of the labeled users |

`dgraph_data.py` keeps the official training set and splits the official validation set 50/50 into
validation and test, because the competition release of DGraph-Fin withholds the official test labels.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
./scripts/run_sample.sh        # the full pipeline on the synthetic sample, a few minutes on CPU
```

Results are written as JSON lines to `outputs/sample/`. To run on the real data, leave `DGRAPH_DIR`
and `GCS_OUTPUTS` unset (they default to `data/dgraphfin` and `outputs`) and use the same commands
with more epochs; the full-graph GNNs take one to three hours each on a laptop CPU.

## Scripts

| Script | What it does |
|---|---|
| `paths.py` | Data and output locations (`DGRAPH_DIR`, `GCS_OUTPUTS`) |
| `make_sample.py` | Writes the synthetic sample |
| `dgraph_data.py` | Loads the `.npz`, drops columns beyond the 17 published features, adds missing-value indicators, standardizes, splits and caches |
| `graph_features.py` | Label-free feature blocks: structure (degree, contact-type counts, edge-time statistics) and neighbor means over the real or *k*-NN graph |
| `build_graphs.py` | Exact *k*-NN graph with FAISS |
| `train_tabular.py` | LightGBM, CatBoost or an MLP on any combination of feature blocks; `--tuned` uses the values from `tune_tabular.py` |
| `tune_tabular.py` | Syne Tune (BOHB) search per learner and feature set, by validation AUC (test users untouched) |
| `tune_trial.py` | One tuning trial; reports validation AUC as it trains so weak trials stop early |
| `ablate_struct.py` | Which structure features carry the lift |
| `train_gnn.py` | GraphSAGE, GCN or GAT on node features over a chosen graph |
| `train_embed.py` | Relational GraphSAGE, graph transformer (with attention weights), global linear attention, and Deep Graph Infomax, each with a logistic-regression head |
| `metrics.py` | AUC, Gini, Brier score, Brier Index and the base-rate-adjusted Brier Index |
| `calibration.py` | Scores every saved prediction file on the test users with `metrics.py` |
| `modal_gpu.py` | Runs the GPU jobs of `train_embed.py` on Modal, with the data in a private volume |
| `gpu_jobs.py` | The same GPU jobs by name, for the Google Cloud runs |
| `gcp_gpu.sh` | Runs those jobs over several seeds on one L4 in Google Cloud, with the data in a private bucket; the VM deletes itself when the jobs end |
| `gcp_startup.sh` | Startup script of that VM: runs each job and seed in turn and skips finished ones, so a reclaimed spot VM resumes where it stopped |
| `bootstrap.py` | Paired bootstrap intervals for test-AUC differences between saved predictions |
| `attention_share.py` | How much attention the transformer on the contact + k-NN graph gives the k-NN edges |
| `run_sample.sh` | Runs all of the above on the sample |

Two practical notes. Message passing on the full graph runs on the CPU: at 3.7 million nodes, sparse
aggregation is slower on Apple's MPS backend than on the CPU, and sparse CSR matrices are not
supported there. FAISS and LightGBM each bundle an OpenMP runtime that conflicts with PyTorch's, so
the scripts load them in separate processes or before torch, as noted in the code.

## Checks

```bash
uv run ruff check scripts/ && uv run ruff format --check scripts/
uv run mypy scripts/
uv run ty check --exit-zero-on-warning scripts/
```

ty reports two library declarations as warnings at every PyTorch and PyG call site; see the comment
in `pyproject.toml`.

## References

Falkner, S., Klein, A., and Hutter, F. (2018). BOHB: robust and efficient hyperparameter
optimization at scale. *ICML 2018*, PMLR 80, 1437–1446.

Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F. (2020). Transformers are RNNs: fast
autoregressive transformers with linear attention. *ICML 2020*. arXiv:2006.16236.

Huang, X., Yang, Y., Wang, Y., Wang, C., Zhang, Z., Xu, J., Chen, L., and Vazirgiannis, M. (2022).
DGraph: a large-scale financial dataset for graph anomaly detection. *NeurIPS 2022, Datasets and
Benchmarks Track*. arXiv:2207.03579.

Kučinskas, S., Bastani, H., and Reynolds, M. (2026). Making forecasting scores easier to
interpret: introducing the Brier Index. Forecasting Research Institute, 4 March 2026.
https://forecastingresearch.substack.com/p/introducing-the-brier-index

Salinas, D., Seeger, M., Klein, A., Perrone, V., Wistuba, M., and Archambeau, C. (2022). Syne
Tune: a library for large scale hyperparameter tuning and reproducible research. *AutoML 2022*,
PMLR 188.
