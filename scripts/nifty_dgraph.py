"""NIFTY on DGraph-Fin: its training recipe on the contact graph, the k-NN graph and no graph.

NIFTY (Agarwal, Lakkaraju and Zitnik, 2021) trains a GNN encoder on two perturbed views of the
graph: each view drops edges at random and shifts a random subset of feature columns by one draw
of N(0, 1), and the second view also flips the sensitive attribute. A Siamese loss (negative
cosine similarity between one view's prediction head and the other's projection, with a stop
gradient) makes the embeddings agree across views, and a classifier head on the embeddings is
trained on the labels. DGraph-Fin's features are anonymized and include no sensitive attribute,
so the flip is left out and the two views differ by edge and feature perturbations only; the rest
follows models/ssf.py and nifty_sota_gnn.py in the NIFTY repository: a one-layer GCN encoder with
spectral normalization, the same projection, prediction and classifier heads, the two optimizers,
the settings of NIFTY's README (hidden 16, learning rate 1e-3, 1000 epochs, sim_coeff 0.6, edge
drop 0.001, feature drop 0.1), and selection on validation loss.

    --model nifty   the NIFTY recipe above
    --model gcn     NIFTY's plain GCN baseline (one GCN layer, linear head), same selection
    --select auc    keep the NIFTY checkpoint with the highest validation AUC instead

Both use the 33 node features only, so they compare with the node-only rows of the paper's
tables. Results go to results_nifty_dgraph.jsonl and test-user scores to preds/.

Run: uv run python scripts/nifty_dgraph.py --model nifty --graph real --seed 0
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn.utils import spectral_norm
from torch_geometric.nn import GCNConv
from torch_geometric.utils import coalesce, dropout_edge, to_undirected

import dgraph_data
import paths

RESULTS = paths.OUTPUTS / "results_nifty_dgraph.jsonl"


# -------------------------------------------------------------------------------
# Graphs
# -------------------------------------------------------------------------------
def load_edges(name: str, g: dgraph_data.DGraph) -> torch.Tensor:
    """Undirected edges without self-loops; GCNConv adds the self-loops."""
    if name == "none":
        return torch.empty(2, 0, dtype=torch.long)
    ei = g.edge_index if name == "real" else torch.from_numpy(np.load(paths.GRAPHS / f"{name}.npy"))
    ei = ei[:, ei[0] != ei[1]]
    return coalesce(to_undirected(ei, num_nodes=g.num_nodes), num_nodes=g.num_nodes)


# -------------------------------------------------------------------------------
# Models, as in NIFTY's models/ssf.py and models/gcn.py
# -------------------------------------------------------------------------------
def init_linear(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class SpectralGCN(nn.Module):
    """One GCN layer whose weight is spectrally normalized (NIFTY wraps GCNConv in spectral_norm)."""

    def __init__(self, d_in: int, hidden: int):
        super().__init__()
        self.conv = GCNConv(d_in, hidden)
        # PyG's own Linear saves its weight under a fixed name, so swap in a torch Linear that can carry the norm
        self.conv.lin = spectral_norm(nn.Linear(d_in, hidden, bias=False))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv(x, edge_index)


class NIFTY(nn.Module):
    def __init__(self, d_in: int, hidden: int, proj_hidden: int):
        super().__init__()
        self.encoder = SpectralGCN(d_in, hidden)
        self.fc1 = nn.Sequential(
            spectral_norm(nn.Linear(hidden, proj_hidden)), nn.BatchNorm1d(proj_hidden), nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(spectral_norm(nn.Linear(proj_hidden, hidden)), nn.BatchNorm1d(hidden))
        self.fc3 = nn.Sequential(
            spectral_norm(nn.Linear(hidden, hidden)), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True)
        )
        self.fc4 = spectral_norm(nn.Linear(hidden, hidden))
        self.c1 = spectral_norm(nn.Linear(hidden, 1))
        self.apply(init_linear)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(z))

    def prediction(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc4(self.fc3(z))


class GCN(nn.Module):
    def __init__(self, d_in: int, hidden: int):
        super().__init__()
        self.conv = GCNConv(d_in, hidden)
        self.fc = nn.Linear(hidden, 1)
        self.apply(init_linear)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x, edge_index))


def neg_cosine(h: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return -F.cosine_similarity(h, p.detach(), dim=-1).mean()


def drop_feature(x: torch.Tensor, p: float) -> torch.Tensor:
    """NIFTY's feature perturbation: columns chosen with probability p are shifted by one N(0, 1) draw."""
    mask = torch.rand(x.shape[1], device=x.device) < p
    out = x.clone()
    out[:, mask] += torch.randn(1, device=x.device)
    return out


# -------------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------------
def train_nifty(x: torch.Tensor, ei: torch.Tensor, y: torch.Tensor, idx: dict, args: argparse.Namespace) -> tuple:
    model = NIFTY(x.shape[1], args.hidden, args.proj_hidden).to(x.device)
    enc = list(model.encoder.parameters())
    heads = [p for m in (model.fc1, model.fc2, model.fc3, model.fc4) for p in m.parameters()]
    opt_sim = torch.optim.Adam(enc + heads, lr=args.lr, weight_decay=args.weight_decay)
    opt_cls = torch.optim.Adam(list(model.c1.parameters()) + enc, lr=args.lr, weight_decay=args.weight_decay)
    tr, va = idx["train"], idx["val"]
    # fixed validation views, as in nifty_sota_gnn.py
    val_views = [(drop_feature(x, args.drop_feature), dropout_edge(ei, p=args.drop_edge)[0]) for _ in range(2)]

    def sim_loss(z1: torch.Tensor, z2: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        p1, p2 = model.projection(z1), model.projection(z2)
        h1, h2 = model.prediction(p1), model.prediction(p2)
        return args.sim_coeff * (neg_cosine(h1[rows], p2[rows]) + neg_cosine(h2[rows], p1[rows])) / 2

    best_loss, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(args.epochs + 1):
        model.train()
        views = [(drop_feature(x, args.drop_feature), dropout_edge(ei, p=args.drop_edge)[0]) for _ in range(2)]
        opt_sim.zero_grad()
        opt_cls.zero_grad()
        z1, z2 = (model(xv, ev) for xv, ev in views)
        sim_loss(z1, z2, tr).backward()
        opt_sim.step()
        # the classifier step recomputes the embeddings with the updated encoder
        opt_cls.zero_grad()
        z1, z2 = (model(xv, ev) for xv, ev in views)
        cls = sum(F.binary_cross_entropy_with_logits(model.c1(z)[tr].squeeze(1), y[tr]) / 2 for z in (z1, z2))
        ((1 - args.sim_coeff) * cls).backward()
        opt_cls.step()

        model.eval()
        with torch.no_grad():
            z1, z2 = (model(xv, ev) for xv, ev in val_views)
            val_cls = sum(F.binary_cross_entropy_with_logits(model.c1(z)[va].squeeze(1), y[va]) / 2 for z in (z1, z2))
            val_loss = float(sim_loss(z1, z2, va) + val_cls)
            if args.select == "auc":
                # rank checkpoints by the classifier alone, which the fraud rate drowns in the summed loss
                logits = sum(model.c1(z)[va].squeeze(1) for z in (z1, z2))
                val_loss = -float(roc_auc_score(y[va].cpu().numpy(), logits.cpu().numpy()))
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 100 == 0:
            print(f"epoch {epoch} val_loss={val_loss:.4f} best@{best_epoch}", flush=True)
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model.c1(model(x, ei)).squeeze(1).cpu().numpy(), best_epoch


def train_gcn(x: torch.Tensor, ei: torch.Tensor, y: torch.Tensor, idx: dict, args: argparse.Namespace) -> tuple:
    model = GCN(x.shape[1], args.hidden).to(x.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tr, va = idx["train"], idx["val"]
    best_loss, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(args.epochs + 1):
        model.train()
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(model(x, ei)[tr].squeeze(1), y[tr]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = F.binary_cross_entropy_with_logits(model(x, ei)[va].squeeze(1), y[va]).item()
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 100 == 0:
            print(f"epoch {epoch} val_loss={val_loss:.4f} best@{best_epoch}", flush=True)
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(x, ei).squeeze(1).cpu().numpy(), best_epoch


# -------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["nifty", "gcn"], default="nifty")
    ap.add_argument("--graph", default="real", help="real, none, or a k-NN graph name (knn2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    # NIFTY's settings from its README
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--proj-hidden", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--sim-coeff", type=float, default=0.6)
    ap.add_argument("--drop-edge", type=float, default=0.001)
    ap.add_argument("--drop-feature", type=float, default=0.1)
    ap.add_argument(
        "--select",
        choices=["loss", "auc"],
        default="loss",
        help="keep the checkpoint with the lowest validation loss (NIFTY) or the highest validation AUC",
    )
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    g = dgraph_data.load()
    dev = torch.device(args.device)
    x, ei = g.x.to(dev), load_edges(args.graph, g).to(dev)
    y = g.y.clamp(min=0).float().to(dev)
    idx = {"train": g.train_idx.to(dev), "val": g.val_idx.to(dev)}
    print(f"{args.model} on {args.graph}: {g.num_nodes:,} users, {ei.shape[1] // 2:,} undirected edges", flush=True)

    t0 = time.time()
    train = train_nifty if args.model == "nifty" else train_gcn
    p, best_epoch = train(x, ei, y, idx, args)
    yy, va, te = g.y.numpy(), g.val_idx.numpy(), g.test_idx.numpy()
    row = {
        "model": args.model,
        "graph": args.graph,
        "seed": args.seed,
        "select": args.select,
        "best_epoch": best_epoch,
        "val_auc": float(roc_auc_score(yy[va], p[va])),
        "test_auc": float(roc_auc_score(yy[te], p[te])),
        "seconds": round(time.time() - t0, 1),
    }
    row["test_gini"] = 2 * row["test_auc"] - 1
    print(json.dumps(row))
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    paths.PREDS.mkdir(parents=True, exist_ok=True)
    np.save(paths.PREDS / f"nifty_{args.model}_{args.graph}_{args.select}_s{args.seed}.npy", p.astype(np.float32))


if __name__ == "__main__":
    main()
