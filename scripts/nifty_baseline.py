"""NIFTY's credit graphs with and without the edges: does message passing beat the node features?

NIFTY (Agarwal, Lakkaraju and Zitnik, 2021) turns the German credit and Taiwan credit-default data
into graphs by linking borrowers whose feature vectors are close. On NIFTY's own data, split and
settings (loaded by nifty_tabular.py), this script trains, with several seeds:

    gcn, sage            NIFTY's GCN and GraphSAGE, same layers, optimizer, epochs and selection
                         on validation loss
    gcn_noedge, ...      the same networks with every edge removed (self-loops only)
    gcn_shuffled, ...    the same networks on the graph with node ids permuted at random, which
                         keeps the degree sequence but breaks the feature similarity

and first runs nifty_tabular.py in a child process for the graph-free baselines on the same test
users. Only the training labels are used for fitting: 100 borrowers for German and 6,000 for
credit, as in NIFTY. Results go to results_nifty.jsonl.

Run: uv run python scripts/nifty_baseline.py --dataset german --seeds 0 1 2 3 4
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv

from nifty_tabular import RESULTS, SPECS, load, row, write


def shuffled(edge_index: np.ndarray, n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n)[edge_index]


# -------------------------------------------------------------------------------
# NIFTY's GCN and GraphSAGE (models/gcn.py, models/sage.py)
# -------------------------------------------------------------------------------
class GCN(nn.Module):
    def __init__(self, d_in: int, hidden: int):
        super().__init__()
        self.conv = GCNConv(d_in, hidden)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x, edge_index))


class SAGE(nn.Module):
    def __init__(self, d_in: int, hidden: int, dropout: float):
        super().__init__()
        self.conv1 = SAGEConv(d_in, hidden, normalize=True)
        self.transition = nn.Sequential(nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout))
        self.conv2 = SAGEConv(hidden, hidden, normalize=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv2(self.transition(self.conv1(x, edge_index)), edge_index))


def init_linear(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def train_gnn(arch: str, d: dict, edges: np.ndarray, seed: int, args: argparse.Namespace) -> np.ndarray:
    """Train as nifty_sota_gnn.py does: full batch, Adam, keep the weights with the lowest validation loss."""
    torch.manual_seed(seed)
    x, edge_index = torch.from_numpy(d["x"]), torch.from_numpy(edges)
    model: nn.Module = (
        GCN(d["x"].shape[1], args.hidden) if arch == "gcn" else SAGE(d["x"].shape[1], args.hidden, args.dropout)
    )
    model.apply(init_linear)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    y = torch.tensor(d["y"], dtype=torch.float32).unsqueeze(1)
    train, val = torch.from_numpy(d["train"]), torch.from_numpy(d["val"])
    best_loss, best_state = float("inf"), None
    for _ in range(args.epochs + 1):
        model.train()
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(x, edge_index)[train], y[train])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = F.binary_cross_entropy_with_logits(model(x, edge_index)[val], y[val]).item()
        if val_loss < best_loss:
            best_loss, best_state = (
                val_loss,
                {k: v.clone() for k, v in model.state_dict().items()},
            )
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(x, edge_index).squeeze(1).numpy()


# -------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(SPECS), default="german")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--labels", type=int, default=None, help="cap on training labels (default: NIFTY's)")
    ap.add_argument("--no-tabular", action="store_true", help="skip the graph-free baselines")
    # NIFTY's settings from its README
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=1000)
    args = ap.parse_args()

    d = load(args.dataset, args.labels)
    n = len(d["y"])
    n_edges = int((d["edge_index"][0] != d["edge_index"][1]).sum()) // 2
    print(
        f"{args.dataset}: {n} borrowers, {n_edges:,} similarity edges, {d['x'].shape[1]} features, "
        f"train/val/test {len(d['train'])}/{len(d['val'])}/{len(d['test'])}"
    )
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_tabular:
        tabular = Path(__file__).with_name("nifty_tabular.py")
        cmd = [sys.executable, str(tabular), "--dataset", args.dataset, "--seeds", *[str(s) for s in args.seeds]]
        if args.labels:
            cmd += ["--labels", str(args.labels)]
        subprocess.run(cmd, check=True)

    self_loops = np.tile(np.arange(n), (2, 1))
    for seed in args.seeds:
        t0 = time.time()
        rows = []
        for arch in ["gcn", "sage"]:
            rows.append(row(args.dataset, arch, seed, train_gnn(arch, d, d["edge_index"], seed, args), d))
            rows.append(row(args.dataset, f"{arch}_noedge", seed, train_gnn(arch, d, self_loops, seed, args), d))
            edges = shuffled(d["edge_index"], n, seed)
            rows.append(row(args.dataset, f"{arch}_shuffled", seed, train_gnn(arch, d, edges, seed, args), d))
        write(rows)
        print(f"seed {seed}: GNNs done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
