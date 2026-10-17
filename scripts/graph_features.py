"""Graph-derived node features for tabular learners (SIGN-style propagation).

Feature blocks:
    self        preprocessed node features (33)
    real_struct in/out degree, contact-type counts (11 x 2), edge-time stats
    real_agg    mean neighbor features over out-edges (contacts I listed),
                in-edges (people who listed me), and 2-hop undirected
    knn{k}_agg  mean neighbor features over the k-NN graph, 1 and 2 hops
    rf_agg      same over the RF-proximity graph (when built)

No labels are used in any block, so nothing leaks from val/test.

Run: uv run python scripts/graph_features.py
"""

import numpy as np
import torch
from torch_geometric.utils import to_undirected

import dgraph_data
import paths

GRAPHS = paths.GRAPHS
FEATS = paths.FEATS


# -------------------------------------------------------------------------------
# Propagation helpers
# -------------------------------------------------------------------------------
def mean_adj(src: torch.Tensor, dst: torch.Tensor, n: int) -> torch.Tensor:
    """Row-normalized adjacency: row i averages over j for edges (i -> j)."""
    deg = torch.bincount(src, minlength=n).clamp(min=1).float()
    vals = 1.0 / deg[src]
    return torch.sparse_coo_tensor(torch.stack([src, dst]), vals, (n, n)).coalesce().to_sparse_csr()


def propagate(a: torch.Tensor, x: torch.Tensor, hops: int) -> list[torch.Tensor]:
    out, h = [], x
    for _ in range(hops):
        h = a @ h
        out.append(h)
    return out


# -------------------------------------------------------------------------------
# Blocks
# -------------------------------------------------------------------------------
def real_struct(g: dgraph_data.DGraph) -> torch.Tensor:
    n = g.num_nodes
    src, dst = g.edge_index
    n_types = int(g.edge_type.max()) + 1
    cols = [torch.bincount(src, minlength=n).float(), torch.bincount(dst, minlength=n).float()]
    for etype in range(1, n_types):
        m = g.edge_type == etype
        cols.append(torch.bincount(src[m], minlength=n).float())
        cols.append(torch.bincount(dst[m], minlength=n).float())
    t = g.edge_time.float()
    both = torch.cat([src, dst])
    tt = torch.cat([t, t])
    cnt = torch.bincount(both, minlength=n).clamp(min=1).float()
    t_mean = torch.zeros(n).index_add_(0, both, tt) / cnt
    t_min = torch.full((n,), float("inf")).scatter_reduce_(0, both, tt, "amin")
    t_max = torch.zeros(n).scatter_reduce_(0, both, tt, "amax")
    t_min = torch.where(torch.isinf(t_min), torch.zeros_like(t_min), t_min)
    cols += [t_mean, t_min, t_max, t_max - t_min]
    return torch.log1p(torch.stack(cols, 1))


def real_agg(g: dgraph_data.DGraph) -> torch.Tensor:
    n = g.num_nodes
    src, dst = g.edge_index
    out_mean = mean_adj(src, dst, n) @ g.x
    in_mean = mean_adj(dst, src, n) @ g.x
    und = to_undirected(g.edge_index, num_nodes=n)
    two_hop = propagate(mean_adj(und[0], und[1], n), g.x, 2)[1]
    return torch.cat([out_mean, in_mean, two_hop], 1)


def sim_agg(g: dgraph_data.DGraph, name: str) -> torch.Tensor:
    ei = torch.from_numpy(np.load(GRAPHS / f"{name}.npy"))
    und = to_undirected(ei, num_nodes=g.num_nodes)
    return torch.cat(propagate(mean_adj(und[0], und[1], g.num_nodes), g.x, 2), 1)


def build(name: str, g: dgraph_data.DGraph) -> np.ndarray:
    path = FEATS / f"{name}.npy"
    if path.exists():
        return np.load(path)
    if name == "self":
        f = g.x
    elif name == "real_struct":
        f = real_struct(g)
    elif name == "real_agg":
        f = real_agg(g)
    elif name.endswith("_agg"):
        f = sim_agg(g, name.removesuffix("_agg"))
    else:
        raise ValueError(name)
    arr = f.numpy().astype(np.float32)
    FEATS.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


def matrix(blocks: list[str], g: dgraph_data.DGraph) -> np.ndarray:
    return np.hstack([build(b, g) for b in blocks])


if __name__ == "__main__":
    g = dgraph_data.load()
    for b in ["self", "real_struct", "real_agg"]:
        print(b, build(b, g).shape)
