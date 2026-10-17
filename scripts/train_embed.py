"""Graph embeddings with a classification head on DGraph-Fin (no GBDT).

Two ways to get node embeddings, each followed by a logistic-regression head
(a linear probe: the scorecard-shaped model on top of the graph):

    relsage  supervised, relational GraphSAGE that sees what the tabular
             model sees: node features + real-graph structure features as
             inputs, separate aggregation over out-edges (contacts I named)
             and in-edges (people who named me), and a learned embedding of
             contact type and edge time added to every message. Trained end
             to end with a linear head; the probe refits the head on frozen
             embeddings.
    transformer  supervised graph transformer with the same inputs as relsage:
             TransformerConv layers attend over each user's contacts in both
             directions, with contact type, edge time and direction as edge
             features, so it learns which contacts matter instead of averaging.
    dgi      label-free Deep Graph Infomax embeddings (2-layer GraphSAGE
             encoder on node features only), on the contact graph or a k-NN
             graph, so the graph is the only thing that differs.

Message passing runs on CPU (index_add_ is faster there than on MPS at this
scale).

Run: uv run python scripts/train_embed.py --model relsage
     uv run python scripts/train_embed.py --model transformer
     uv run python scripts/train_embed.py --model dgi --graph real
     uv run python scripts/train_embed.py --model dgi --graph knn2
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import DeepGraphInfomax, SAGEConv, TransformerConv
from torch_geometric.utils import coalesce, to_undirected

import dgraph_data
import graph_features
import paths

GRAPHS = paths.GRAPHS
RESULTS = paths.OUTPUTS / "results_embed.jsonl"
PREDS = paths.PREDS


# -------------------------------------------------------------------------------
# Relational GraphSAGE
# -------------------------------------------------------------------------------
class RelConv(nn.Module):
    """h_i' = W_s h_i + W_o mean_{i->j} relu(h_j + e_ij) + W_i mean_{j->i} relu(h_j + e_ij)."""

    def __init__(self, d: int, n_types: int):
        super().__init__()
        self.type_out = nn.Embedding(n_types, d)
        self.type_in = nn.Embedding(n_types, d)
        self.time = nn.Linear(1, d)
        self.w_self = nn.Linear(d, d)
        self.w_out = nn.Linear(d, d, bias=False)
        self.w_in = nn.Linear(d, d, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        etype: torch.Tensor,
        etime: torch.Tensor,
        inv_out: torch.Tensor,
        inv_in: torch.Tensor,
    ) -> torch.Tensor:
        t = self.time(etime)
        m_out = F.relu(h[dst] + self.type_out(etype) + t)
        agg_out = torch.zeros_like(h).index_add_(0, src, m_out) * inv_out
        m_in = F.relu(h[src] + self.type_in(etype) + t)
        agg_in = torch.zeros_like(h).index_add_(0, dst, m_in) * inv_in
        return self.w_self(h) + self.w_out(agg_out) + self.w_in(agg_in)


class RelSAGE(nn.Module):
    def __init__(self, d_in: int, hidden: int, layers: int, n_types: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.inp = nn.Linear(d_in, hidden)
        self.convs = nn.ModuleList([RelConv(hidden, n_types) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
        self.head = nn.Linear(hidden, 2)

    def embed(self, x: torch.Tensor, graph: dict[str, torch.Tensor]) -> torch.Tensor:
        h = F.relu(self.inp(x))
        for conv, norm in zip(self.convs, self.norms, strict=True):
            z = F.relu(norm(conv(h, **graph)))
            h = h + F.dropout(z, self.dropout, self.training)
        return h

    def forward(self, x: torch.Tensor, graph: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.embed(x, graph))


def relational_graph(g: dgraph_data.DGraph) -> dict[str, torch.Tensor]:
    src, dst = g.edge_index
    n = g.num_nodes
    inv_out = 1.0 / torch.bincount(src, minlength=n).clamp(min=1).float()
    inv_in = 1.0 / torch.bincount(dst, minlength=n).clamp(min=1).float()
    return dict(
        src=src,
        dst=dst,
        etype=g.edge_type,
        etime=(g.edge_time.float() / g.edge_time.max()).unsqueeze(1),
        inv_out=inv_out.unsqueeze(1),
        inv_in=inv_in.unsqueeze(1),
    )


# -------------------------------------------------------------------------------
# Graph transformer
# -------------------------------------------------------------------------------
class GraphTransformer(nn.Module):
    """TransformerConv layers over contacts in both directions, residual, linear head."""

    def __init__(self, d_in: int, hidden: int, layers: int, heads: int, edge_dim: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.inp = nn.Linear(d_in, hidden)
        self.convs = nn.ModuleList(
            [TransformerConv(hidden, hidden // heads, heads=heads, edge_dim=edge_dim) for _ in range(layers)]
        )
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
        self.head = nn.Linear(hidden, 2)

    def embed(self, x: torch.Tensor, graph: dict[str, torch.Tensor]) -> torch.Tensor:
        h = F.relu(self.inp(x))
        for conv, norm in zip(self.convs, self.norms, strict=True):
            z = F.relu(norm(conv(h, graph["edge_index"], graph["edge_attr"])))
            h = h + F.dropout(z, self.dropout, self.training)
        return h

    def forward(self, x: torch.Tensor, graph: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.embed(x, graph))


def transformer_graph(g: dgraph_data.DGraph) -> dict[str, torch.Tensor]:
    """Messages flow j -> i; user i hears from the users who named it (edge as stored)
    and from the contacts it named (edge reversed), told apart by a direction flag."""
    src, dst = g.edge_index
    n_types = int(g.edge_type.max()) + 1
    one_hot = F.one_hot(g.edge_type, n_types).float()
    t = (g.edge_time.float() / g.edge_time.max()).unsqueeze(1)
    attr = torch.cat([one_hot, t], 1)
    edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], 1)
    flag = torch.cat([torch.zeros(len(src), 1), torch.ones(len(src), 1)])
    return dict(edge_index=edge_index, edge_attr=torch.cat([attr.repeat(2, 1), flag], 1))


def train_supervised(g: dgraph_data.DGraph, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, int]:
    struct = graph_features.build("real_struct", g)
    struct = (struct - struct.mean(0)) / (struct.std(0) + 1e-6)
    x = torch.cat([g.x, torch.from_numpy(struct)], 1)
    model: RelSAGE | GraphTransformer
    if args.model == "relsage":
        graph = relational_graph(g)
        model = RelSAGE(x.shape[1], args.hidden, args.layers, int(g.edge_type.max()) + 1, args.dropout)
    else:
        graph = transformer_graph(g)
        model = GraphTransformer(
            x.shape[1], args.hidden, args.layers, args.heads, graph["edge_attr"].shape[1], args.dropout
        )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    print(
        f"{args.model}: inputs={x.shape[1]} edges={g.edge_index.shape[1]:,} "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )

    best_val, best_state, best_epoch, t0 = 0.0, None, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x, graph)[g.train_idx], g.y[g.train_idx])
        loss.backward()
        opt.step()
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                p = model(x, graph).softmax(-1)[:, 1]
            val_auc = roc_auc_score(g.y[g.val_idx].numpy(), p[g.val_idx].numpy())
            if val_auc > best_val:
                best_val, best_epoch = val_auc, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if epoch % 25 == 0:
                print(
                    f"epoch {epoch} loss={loss.item():.4f} val_auc={val_auc:.4f} "
                    f"best={best_val:.4f}@{best_epoch} {time.time() - t0:.0f}s",
                    flush=True,
                )
            if epoch - best_epoch >= args.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z = model.embed(x, graph)
        p = model.head(z).softmax(-1)[:, 1]
    return z, p, best_epoch


# -------------------------------------------------------------------------------
# Deep Graph Infomax (label-free)
# -------------------------------------------------------------------------------
class SAGEEncoder(nn.Module):
    def __init__(self, d_in: int, hidden: int):
        super().__init__()
        self.conv1 = SAGEConv(d_in, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.act1 = nn.PReLU(hidden)
        self.act2 = nn.PReLU(hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.conv1(x, edge_index))
        return self.act2(self.conv2(h, edge_index))


def corruption(x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x[torch.randperm(x.size(0))], edge_index


def summary(z: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
    return torch.sigmoid(z.mean(dim=0))


def load_edges(name: str, g: dgraph_data.DGraph) -> torch.Tensor:
    ei = g.edge_index if name == "real" else torch.from_numpy(np.load(GRAPHS / f"{name}.npy"))
    ei = ei[:, ei[0] != ei[1]]
    return coalesce(to_undirected(ei, num_nodes=g.num_nodes), num_nodes=g.num_nodes)


def train_dgi(g: dgraph_data.DGraph, args: argparse.Namespace) -> tuple[torch.Tensor, int]:
    ei = load_edges(args.graph, g)
    model = DeepGraphInfomax(args.hidden, SAGEEncoder(g.x.shape[1], args.hidden), summary, corruption)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"dgi: graph={args.graph} edges={ei.shape[1]:,}")
    best_loss, best_state, best_epoch, t0 = float("inf"), None, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        pos, neg, s = model(g.x, ei)
        loss = model.loss(pos, neg, s)
        loss.backward()
        opt.step()
        if loss.item() < best_loss:
            best_loss, best_epoch = loss.item(), epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(
                f"epoch {epoch} loss={loss.item():.4f} best={best_loss:.4f}@{best_epoch} {time.time() - t0:.0f}s",
                flush=True,
            )
        if epoch - best_epoch >= args.patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z = model.encoder(g.x, ei)
    return z, best_epoch


# -------------------------------------------------------------------------------
# Logistic-regression head (linear probe)
# -------------------------------------------------------------------------------
def probe(feats: np.ndarray, g: dgraph_data.DGraph) -> dict[str, float]:
    y = g.y.numpy()
    tr, va, te = (i.numpy() for i in (g.train_idx, g.val_idx, g.test_idx))
    scaler = StandardScaler().fit(feats[tr])
    best: tuple[float, float, LogisticRegression] | None = None
    for c in (0.01, 0.1, 1.0):
        lr = LogisticRegression(C=c, max_iter=2000).fit(scaler.transform(feats[tr]), y[tr])
        val = roc_auc_score(y[va], lr.predict_proba(scaler.transform(feats[va]))[:, 1])
        if best is None or val > best[0]:
            best = (val, c, lr)
    assert best is not None
    val, c, lr = best
    test = roc_auc_score(y[te], lr.predict_proba(scaler.transform(feats[te]))[:, 1])
    return dict(val_auc=val, test_auc=test, test_gini=2 * test - 1, C=c)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["relsage", "transformer", "dgi"], default="relsage")
    ap.add_argument("--graph", default="real", help="dgi only: real or a k-NN graph name (knn2, knn6)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4, help="transformer only")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    g = dgraph_data.load()
    y = g.y.numpy()
    te = g.test_idx.numpy()
    t0 = time.time()
    rows = []

    if args.model in ("relsage", "transformer"):
        z, p, best_epoch = train_supervised(g, args)
        test_auc = roc_auc_score(y[te], p[te].numpy())
        rows.append(
            dict(
                model=args.model,
                head="end-to-end linear",
                graph="real",
                best_epoch=best_epoch,
                test_auc=test_auc,
                test_gini=2 * test_auc - 1,
            )
        )
        rows.append(dict(model=args.model, head="logistic probe", graph="real", **probe(z.numpy(), g)))
        PREDS.mkdir(parents=True, exist_ok=True)
        np.save(PREDS / f"{args.model}_real.npy", p.numpy().astype(np.float32))
    else:
        if args.graph == "real":
            # the label-free baseline: the same logistic head on node features alone
            rows.append(dict(model="none", head="logistic", graph="none", **probe(g.x.numpy(), g)))
        z, best_epoch = train_dgi(g, args)
        rows.append(
            dict(model="dgi", head="logistic probe", graph=args.graph, best_epoch=best_epoch, **probe(z.numpy(), g))
        )
        rows.append(
            dict(
                model="dgi+x", head="logistic probe", graph=args.graph, **probe(np.hstack([g.x.numpy(), z.numpy()]), g)
            )
        )

    RESULTS.parent.mkdir(exist_ok=True)
    with RESULTS.open("a") as f:
        for row in rows:
            row.update(seed=args.seed, seconds=round(time.time() - t0, 1))
            print(json.dumps(row))
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
