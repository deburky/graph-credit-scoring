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
    global   linear global attention (Katharopoulos et al., 2020) on node features
             only: every user attends to every user, so the model learns its own
             similarity graph (the learned-similarity control; no edges are used).
    dgi      label-free Deep Graph Infomax embeddings (2-layer GraphSAGE
             encoder on node features only), on the contact graph or a k-NN
             graph, so the graph is the only thing that differs.

Use --device cuda on a GPU (scripts/modal_gpu.py runs these on Modal). On Apple MPS, sparse
aggregation is slower than on the CPU at this scale, and the global-attention reductions over all
3.7M users come out wrong, so use the CPU there.

Run: uv run python scripts/train_embed.py --model relsage
     uv run python scripts/train_embed.py --model transformer
     uv run python scripts/train_embed.py --model transformer --add-knn knn2
     uv run python scripts/train_embed.py --model global --device cuda
     uv run python scripts/train_embed.py --model dgi --graph real
     uv run python scripts/train_embed.py --model dgi --graph knn2
"""

import argparse
import dataclasses
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.checkpoint import checkpoint
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
    def __init__(self, d_in: int, hidden: int, layers: int, n_types: int, dropout: float, ckpt: bool = False):
        super().__init__()
        self.dropout = dropout
        self.ckpt = ckpt
        self.inp = nn.Linear(d_in, hidden)
        self.convs = nn.ModuleList([RelConv(hidden, n_types) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
        self.head = nn.Linear(hidden, 2)

    def embed(self, x: torch.Tensor, graph: dict[str, torch.Tensor]) -> torch.Tensor:
        h = F.relu(self.inp(x))
        for conv, norm in zip(self.convs, self.norms, strict=True):

            def layer(h_in: torch.Tensor, conv: nn.Module = conv) -> torch.Tensor:
                return conv(h_in, **graph)

            out = checkpoint(layer, h, use_reentrant=False) if self.ckpt and self.training else layer(h)
            z = F.relu(norm(out))
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

    def __init__(
        self, d_in: int, hidden: int, layers: int, heads: int, edge_dim: int, dropout: float, ckpt: bool = False
    ):
        super().__init__()
        self.dropout = dropout
        self.ckpt = ckpt
        self.inp = nn.Linear(d_in, hidden)
        self.convs = nn.ModuleList(
            [TransformerConv(hidden, hidden // heads, heads=heads, edge_dim=edge_dim) for _ in range(layers)]
        )
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
        self.head = nn.Linear(hidden, 2)

    def embed(self, x: torch.Tensor, graph: dict[str, torch.Tensor]) -> torch.Tensor:
        h = F.relu(self.inp(x))
        for conv, norm in zip(self.convs, self.norms, strict=True):
            args = (h, graph["edge_index"], graph["edge_attr"])
            out = checkpoint(conv, *args, use_reentrant=False) if self.ckpt and self.training else conv(*args)
            z = F.relu(norm(out))
            h = h + F.dropout(z, self.dropout, self.training)
        return h

    def forward(self, x: torch.Tensor, graph: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.embed(x, graph))

    @torch.no_grad()
    def attention(self, x: torch.Tensor, graph: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Per-layer attention weights, (messages, heads); they sum to one over each user's incoming messages."""
        weights = []
        h = F.relu(self.inp(x))
        for conv, norm in zip(self.convs, self.norms, strict=True):
            out, (_, alpha) = conv(h, graph["edge_index"], graph["edge_attr"], return_attention_weights=True)
            weights.append(alpha)
            h = h + F.relu(norm(out))
        return weights


# -------------------------------------------------------------------------------
# Global linear attention (learned similarity, no graph)
# -------------------------------------------------------------------------------
class GlobalAttention(nn.Module):
    """Kernelized linear attention over all users (Katharopoulos et al., 2020), O(n d^2).

    With phi(z) = elu(z) + 1, user i receives sum_j w_ij v_j where
    w_ij = phi(q_i) . phi(k_j) / sum_l phi(q_i) . phi(k_l): a normalized average over every
    user, weighted by learned feature similarity, computed per head without the n x n matrix.
    """

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.out = nn.Linear(d, d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        n, d = h.shape
        shape = (n, self.heads, d // self.heads)
        q = F.elu(self.wq(h).view(shape)) + 1
        k = F.elu(self.wk(h).view(shape)) + 1
        v = self.wv(h).view(shape)
        kv = torch.einsum("nhm,nhd->hmd", k, v)
        num = torch.einsum("nhm,hmd->nhd", q, kv)
        den = torch.einsum("nhm,hm->nh", q, k.sum(0)).unsqueeze(-1)
        return self.out((num / den).reshape(n, d))


class GlobalFormer(nn.Module):
    """Node features, pre-norm global attention blocks with residuals, linear head. Uses no edges.

    At initialization the attention output is close to the same global mean for every user, so
    it is added through the residual at its own scale. A batch norm after the attention would
    divide that near-constant signal by its tiny spread across users and blow up the noise.
    """

    def __init__(self, d_in: int, hidden: int, layers: int, heads: int, dropout: float, ckpt: bool = False):
        super().__init__()
        self.dropout = dropout
        self.ckpt = ckpt
        self.inp = nn.Linear(d_in, hidden)
        self.attns = nn.ModuleList([GlobalAttention(hidden, heads) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.ffns = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.ReLU()) for _ in range(layers)]
        )
        self.head = nn.Linear(hidden, 2)

    def embed(self, x: torch.Tensor, graph: dict[str, torch.Tensor]) -> torch.Tensor:
        h = F.relu(self.inp(x))
        for attn, norm, ffn in zip(self.attns, self.norms, self.ffns, strict=True):

            def mix(h_in: torch.Tensor, attn: nn.Module = attn, norm: nn.Module = norm) -> torch.Tensor:
                return attn(norm(h_in))

            ckpt = self.ckpt and self.training
            h = h + F.dropout(checkpoint(mix, h, use_reentrant=False) if ckpt else mix(h), self.dropout, self.training)
            h = h + F.dropout(checkpoint(ffn, h, use_reentrant=False) if ckpt else ffn(h), self.dropout, self.training)
        return h

    def forward(self, x: torch.Tensor, graph: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.embed(x, graph))


def add_similarity_edges(g: dgraph_data.DGraph, name: str) -> dgraph_data.DGraph:
    """The contact graph plus a k-NN graph's edges as one extra contact type, with edge time 0.

    Inputs and structure features stay those of the contact graph, so only the messages change:
    a model that can tell the two kinds of edge apart can learn to ignore the similarity edges.
    """
    knn = torch.from_numpy(np.load(paths.GRAPHS / f"{name}.npy"))
    knn_type = int(g.edge_type.max()) + 1
    return dataclasses.replace(
        g,
        edge_index=torch.cat([g.edge_index, knn], 1),
        edge_type=torch.cat([g.edge_type, torch.full((knn.shape[1],), knn_type, dtype=g.edge_type.dtype)]),
        edge_time=torch.cat([g.edge_time, torch.zeros(knn.shape[1], dtype=g.edge_time.dtype)]),
    )


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


def train_supervised(
    g: dgraph_data.DGraph, args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor | None]:
    """Train relsage, transformer or global end to end; returns embeddings, probabilities,
    best epoch, and (transformer only) last-layer attention averaged over heads."""
    dev = torch.device(args.device)
    model: RelSAGE | GraphTransformer | GlobalFormer
    graph: dict[str, torch.Tensor]
    if args.model == "global":
        # the learned-similarity arm: node features only, no edges and no graph-derived features
        x = g.x
        graph = {}
        model = GlobalFormer(x.shape[1], args.hidden, args.layers, args.heads, args.dropout, args.checkpoint)
    else:
        struct = graph_features.build("real_struct", g)
        struct = (struct - struct.mean(0)) / (struct.std(0) + 1e-6)
        x = torch.cat([g.x, torch.from_numpy(struct)], 1)
        if args.model == "relsage":
            graph = relational_graph(g)
            n_types = int(g.edge_type.max()) + 1
            model = RelSAGE(x.shape[1], args.hidden, args.layers, n_types, args.dropout, args.checkpoint)
        else:
            graph = transformer_graph(g)
            model = GraphTransformer(
                x.shape[1],
                args.hidden,
                args.layers,
                args.heads,
                graph["edge_attr"].shape[1],
                args.dropout,
                args.checkpoint,
            )
    x = x.to(dev)
    graph = {k: v.to(dev) for k, v in graph.items()}
    model = model.to(dev)
    train_idx, y_train = g.train_idx.to(dev), g.y[g.train_idx].to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # bfloat16 activations halve the per-edge message tensors; weights and the loss stay float32
    amp = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=args.bf16)
    print(
        f"{args.model}: inputs={x.shape[1]} edges={g.edge_index.shape[1]:,} "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )

    best_val, best_state, best_epoch, t0 = 0.0, None, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        with amp:
            loss = F.cross_entropy(model(x, graph)[train_idx], y_train)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad(), amp:
                p = model(x, graph).float().softmax(-1)[:, 1].cpu()
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
    with torch.no_grad(), amp:
        z = model.embed(x, graph)
        p = model.head(z).float().softmax(-1)[:, 1]
        attn = model.attention(x, graph)[-1].mean(1).cpu() if isinstance(model, GraphTransformer) else None
    z = z.float()
    attn = attn.float() if attn is not None else None
    return z.cpu(), p.cpu(), best_epoch, attn


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
    ap.add_argument("--model", choices=["relsage", "transformer", "global", "dgi"], default="relsage")
    ap.add_argument("--graph", default="real", help="dgi only: real or a k-NN graph name (knn2, knn6)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4, help="transformer and global")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--clip", type=float, default=0.0, help="max gradient norm, 0 = no clipping")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu, cuda, or mps (mps is inaccurate for global attention)")
    ap.add_argument(
        "--checkpoint",
        action="store_true",
        help="recompute each layer's messages (or attention) in the backward pass to save GPU memory",
    )
    ap.add_argument("--bf16", action="store_true", help="train in bfloat16 mixed precision (cuda) to save GPU memory")
    ap.add_argument("--add-knn", default=None, help="relsage/transformer: add this k-NN graph's edges (e.g. knn2)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    g = dgraph_data.load()
    # structure features are built (or read from cache) on the contact graph alone
    graph_features.build("real_struct", g)
    if args.add_knn:
        g = add_similarity_edges(g, args.add_knn)
    y = g.y.numpy()
    te = g.test_idx.numpy()
    t0 = time.time()
    rows = []

    if args.model in ("relsage", "transformer", "global"):
        z, p, best_epoch, attn = train_supervised(g, args)
        graph_name = "none" if args.model == "global" else "real" + (f"+{args.add_knn}" if args.add_knn else "")
        test_auc = roc_auc_score(y[te], p[te].numpy())
        rows.append(
            dict(
                model=args.model,
                head="end-to-end linear",
                graph=graph_name,
                best_epoch=best_epoch,
                test_auc=test_auc,
                test_gini=2 * test_auc - 1,
            )
        )
        rows.append(dict(model=args.model, head="logistic probe", graph=graph_name, **probe(z.numpy(), g)))
        PREDS.mkdir(parents=True, exist_ok=True)
        np.save(PREDS / f"{args.model}_{graph_name}.npy", p.numpy().astype(np.float32))
        if attn is not None:
            # message order as in transformer_graph: stored edges (target hears from source), then reversed
            suffix = f"_{graph_name}" if args.add_knn else ""
            np.save(PREDS / f"transformer_attention{suffix}.npy", attn.numpy().astype(np.float16))
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
