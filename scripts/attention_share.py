"""How much attention the graph transformer on the contact + k-NN graph pays to the k-NN edges.

Each user's attention weights sum to 1 over the messages it receives. If attention ignored
the similarity edges, the weight on k-NN messages would be far below their share of the
messages; if it treated all edges alike, the two would be equal. Writes
outputs/results_attention.json.

Run: uv run python scripts/attention_share.py   (after the transformer_knn2 job of modal_gpu.py)
"""

import json

import numpy as np

import dgraph_data
import paths

ATTN = paths.OUTPUTS / "modal" / "transformer_knn2" / "preds" / "transformer_attention_real+knn2.npy"


def main() -> None:
    g = dgraph_data.load()
    n = g.num_nodes
    real = g.edge_index.numpy()
    knn = np.load(paths.GRAPHS / "knn2.npy")
    src = np.concatenate([real[0], knn[0]])
    dst = np.concatenate([real[1], knn[1]])
    # message order of train_embed.transformer_graph: stored edges (dst hears from src), then reversed
    target = np.concatenate([dst, src])
    is_knn = np.tile(np.concatenate([np.zeros(real.shape[1]), np.ones(knn.shape[1])]), 2)
    att = np.load(ATTN).astype(np.float64)
    assert len(att) == len(target)
    mass = np.bincount(target, weights=att, minlength=n)
    mass_knn = np.bincount(target, weights=att * is_knn, minlength=n)
    count = np.bincount(target, minlength=n).astype(np.float64)
    count_knn = np.bincount(target, weights=is_knn, minlength=n)
    y, te = g.y.numpy(), g.test_idx.numpy()
    out: dict[str, dict[str, float]] = {}
    for name, idx in [("test", te), ("test_fraud", te[y[te] == 1]), ("test_normal", te[y[te] == 0])]:
        share = count_knn[idx] / count[idx]
        attention = mass_knn[idx] / mass[idx]
        out[name] = {
            "users": int(len(idx)),
            "knn_message_share": float(share.mean()),
            "knn_attention_share": float(attention.mean()),
            "below_uniform": float((attention < share).mean()),
        }
        print(name, out[name])
    (paths.OUTPUTS / "results_attention.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
