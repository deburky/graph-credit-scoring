"""Full-batch node classification on DGraph-Fin with a choice of graph.

The same model and features are trained on different edge sets so the only
thing that changes is where the edges come from:

    none     no edges (MLP)
    real     emergency-contact edges, made undirected
    knn      k-NN in preprocessed feature space (the Fogli / NIFTY style graph)
    real+knn union of the two

Graphs other than "real" are read from <DGRAPH_DIR>/graphs/<name>.npy
(built by scripts/build_graphs.py, or build_rf_graph.py for rfprox{k}).

Run: uv run python scripts/train_gnn.py --model sage --graph real
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.utils import coalesce, to_undirected

import dgraph_data
import paths

GRAPHS = paths.GRAPHS
RESULTS = paths.OUTPUTS / "results.jsonl"
PREDS = paths.PREDS


# -------------------------------------------------------------------------------
# Models
# -------------------------------------------------------------------------------
class GNN(nn.Module):
    def __init__(self, kind: str, d_in: int, hidden: int, layers: int, dropout: float, heads: int = 4):
        super().__init__()
        self.kind = kind
        self.dropout = dropout
        self.inp = nn.Linear(d_in, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            if kind == "mlp":
                conv = nn.Linear(hidden, hidden)
            elif kind == "gcn":
                conv = GCNConv(hidden, hidden, cached=True)
            elif kind == "sage":
                conv = SAGEConv(hidden, hidden)
            elif kind == "gat":
                conv = GATConv(hidden, hidden // heads, heads=heads)
            else:
                raise ValueError(kind)
            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(hidden))
        self.out = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.inp(x))
        for conv, norm in zip(self.convs, self.norms, strict=True):
            z = conv(h) if self.kind == "mlp" else conv(h, edge_index)
            z = F.relu(norm(z))
            h = h + F.dropout(z, self.dropout, self.training)
        return self.out(h)


# -------------------------------------------------------------------------------
# Graphs
# -------------------------------------------------------------------------------
def load_graph(name: str, g: dgraph_data.DGraph) -> torch.Tensor:
    parts = []
    for part in name.split("+"):
        if part == "none":
            parts.append(torch.empty(2, 0, dtype=torch.long))
        elif part == "real":
            parts.append(g.edge_index)
        else:
            parts.append(torch.from_numpy(np.load(GRAPHS / f"{part}.npy")))
    ei = torch.cat(parts, dim=1)
    ei = ei[:, ei[0] != ei[1]]
    return coalesce(to_undirected(ei, num_nodes=g.num_nodes), num_nodes=g.num_nodes)


# -------------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------------
@torch.no_grad()
def auc(model: GNN, x: torch.Tensor, ei: torch.Tensor, y: torch.Tensor, idxs: list[torch.Tensor]) -> list[float]:
    model.eval()
    p = model(x, ei).softmax(-1)[:, 1].float().cpu()
    return [roc_auc_score(y[i].numpy(), p[i].numpy()) for i in idxs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sage", choices=["mlp", "gcn", "sage", "gat"])
    ap.add_argument("--graph", default="real")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--save-preds", action="store_true", help="save default probabilities for all nodes")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    g = dgraph_data.load()
    graph = "none" if args.model == "mlp" else args.graph
    ei = load_graph(graph, g)
    dev = torch.device(args.device)
    x, ei_dev, y_dev = g.x.to(dev), ei.to(dev), g.y.to(dev)
    train_dev = g.train_idx.to(dev)

    model = GNN(args.model, x.shape[1], args.hidden, args.layers, args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    print(f"model={args.model} graph={graph} edges={ei.shape[1]:,} device={dev}")

    best_val, best_state, best_epoch, t0 = 0.0, None, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(x, ei_dev)
        loss = F.cross_entropy(logits[train_dev], y_dev[train_dev])
        loss.backward()
        opt.step()
        if epoch % 5 == 0:
            (val_auc,) = auc(model, x, ei_dev, g.y, [g.val_idx])
            if val_auc > best_val:
                best_val, best_epoch = val_auc, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if epoch % 25 == 0:
                print(
                    f"epoch {epoch} loss={loss.item():.4f} val_auc={val_auc:.4f} "
                    f"best={best_val:.4f}@{best_epoch} {time.time() - t0:.0f}s"
                )
            if epoch - best_epoch >= args.patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    train_auc, val_auc, test_auc = auc(model, x, ei_dev, g.y, [g.train_idx, g.val_idx, g.test_idx])
    row = dict(
        model=args.model,
        graph=graph,
        edges=ei.shape[1],
        seed=args.seed,
        best_epoch=best_epoch,
        train_auc=train_auc,
        val_auc=val_auc,
        test_auc=test_auc,
        test_gini=2 * test_auc - 1,
        seconds=round(time.time() - t0, 1),
        hidden=args.hidden,
        layers=args.layers,
    )
    print(json.dumps(row))
    if args.save_preds:
        model.eval()
        with torch.no_grad():
            prob = model(x, ei_dev).softmax(-1)[:, 1].float().cpu().numpy()
        PREDS.mkdir(parents=True, exist_ok=True)
        np.save(PREDS / f"{args.model}_{graph}.npy", prob)
    RESULTS.parent.mkdir(exist_ok=True)
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
