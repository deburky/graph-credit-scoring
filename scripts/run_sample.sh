#!/bin/zsh
# End-to-end run on the synthetic sample (a few minutes on a laptop CPU).
# The same commands run on the real DGraph-Fin with DGRAPH_DIR=data/dgraphfin and more epochs.
set -e
cd "$(dirname "$0")/.."
export DGRAPH_DIR=data/sample GCS_OUTPUTS=outputs/sample
mkdir -p "$GCS_OUTPUTS"
py() { uv run python "$@"; }

[ -f data/sample/raw/dgraph.npz ] || py scripts/make_sample.py
py scripts/dgraph_data.py                      # preprocess, split, cache
py scripts/graph_features.py                   # node, structure and neighbor-mean feature blocks

# exact k-NN similarity graph; FAISS runs in its own process (OpenMP clash with torch)
py scripts/build_graphs.py --prepare
KMP_DUPLICATE_LIB_OK=TRUE py scripts/build_graphs.py --k 2 6
py -c "import sys; sys.path.insert(0, 'scripts'); import dgraph_data, graph_features; g = dgraph_data.load(); [graph_features.build(b, g) for b in ('knn2_agg', 'knn6_agg')]"

# tabular learners on graph features
for learner in lgbm catboost mlp; do
  py scripts/train_tabular.py --learner $learner --blocks self
  py scripts/train_tabular.py --learner $learner --blocks self knn6_agg
  py scripts/train_tabular.py --learner $learner --blocks self real_struct real_agg
done
py scripts/ablate_struct.py

# graph neural networks (full batch on CPU)
py scripts/train_gnn.py --model sage --graph real --device cpu --epochs 60
py scripts/train_gnn.py --model sage --graph knn2 --device cpu --epochs 60
py scripts/train_embed.py --model relsage --epochs 60
py scripts/train_embed.py --model transformer --epochs 60
py scripts/train_embed.py --model dgi --graph real --epochs 60 --patience 20
py scripts/train_embed.py --model dgi --graph knn2 --epochs 60 --patience 20

echo "results in $GCS_OUTPUTS/*.jsonl"
