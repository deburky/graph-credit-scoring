"""2-D maps of relational GraphSAGE's hidden state before and after each message-passing layer.

Reads relsage_layers.npz from train_embed.py --save-layers: h0 is the input projection of the node
features and the 28 structure features, h1 and h2 the state after the first and second layer. A
sample of test users (fraudsters oversampled) and their contacts are placed by t-SNE in each
state, each map starting from the previous one's layout so that users can be followed across the
panels. Under the points, the map is shaded by the share of fraudsters among nearby test users
(a Gaussian-kernel average); over them, only the edges from sampled fraudsters to their contacts
are drawn.

Each panel also gets the test AUC of a logistic probe on that state, fit on a sample of training
users.

The layouts and probe AUCs are cached in layer_map.npz; pass --recompute to rebuild them. Writes a
preview (outputs/layer_map.png) and, for a LaTeX figure, one image per panel (layer_{0,1,2}_map.png) and
layer_numbers.tex in LAYER_OUT (default outputs/layer_map); pgfplots draws the titles and the colorbar.

Run: uv run python scripts/train_embed.py --model relsage --save-layers
     uv run python scripts/make_layer_map.py
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import dgraph_data
import paths

OUT = Path(os.environ.get("LAYER_OUT", paths.OUTPUTS / "layer_map"))
CACHE = paths.OUTPUTS / "layer_map.npz"
INK = "#2B2B2B"
FRAUD = "#F33254"
NORMAL = "#8C8C8C"
BACKGROUND = "#C9CDD2"
# the paper's credit-fraud colormap: low teal, mid amber, high crimson
SHARE_MAP = LinearSegmentedColormap.from_list("pdmap", [(0, "#01CABF"), (0.55, "#FDAF03"), (1, "#F33254")])
SHARE_MAX = 0.6
GRID = 240
LIMIT = 1.12
TITLES = ["Before message passing", "After layer 1", "After layer 2"]


# -------------------------------------------------------------------------------
# Sampling and layouts
# -------------------------------------------------------------------------------
def adjacency(edge_index: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """CSR over the undirected graph (row i lists neighbors of i)."""
    src = np.concatenate([edge_index[0], edge_index[1]])
    dst = np.concatenate([edge_index[1], edge_index[0]])
    order = np.argsort(src, kind="stable")
    indptr = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=n))])
    return indptr, dst[order]


def contacts(indptr: np.ndarray, indices: np.ndarray, v: int, cap: int, rng) -> np.ndarray:
    """Contacts of v, at most cap of them."""
    nb = np.unique(indices[indptr[v] : indptr[v + 1]])
    return rng.choice(nb, size=cap, replace=False) if len(nb) > cap else nb


def probe_auc(h: np.ndarray, g: dgraph_data.DGraph, rng) -> float:
    """Test AUC of a logistic regression on h, fit on 200k training users."""
    tr = rng.choice(g.train_idx.numpy(), size=200_000, replace=False)
    te = g.test_idx.numpy()
    y = g.y.numpy()
    scaler = StandardScaler().fit(h[tr])
    clf = LogisticRegression(max_iter=1000).fit(scaler.transform(h[tr]), y[tr])
    return float(roc_auc_score(y[te], clf.predict_proba(scaler.transform(h[te]))[:, 1]))


def compute(args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Sample users, place them in every hidden state, and fit the probes."""
    rng = np.random.default_rng(args.seed)
    g = dgraph_data.load()
    n = g.num_nodes
    y = g.y.numpy()
    states = np.load(paths.PREDS / "relsage_layers.npy", mmap_mode="r")
    layers = [np.asarray(s, dtype=np.float32) for s in states]

    test = g.test_idx.numpy()
    fraud = rng.choice(test[y[test] == 1], size=args.fraud, replace=False)
    normal = rng.choice(test[y[test] == 0], size=args.normal, replace=False)
    seeds = np.concatenate([fraud, normal])
    indptr, indices = adjacency(g.edge_index.numpy(), n)
    links = [(v, u) for v in seeds for u in contacts(indptr, indices, v, args.cap, rng)]
    nodes = np.unique(np.concatenate([seeds, np.array([u for _, u in links])]))
    # an edge is drawn only from a sampled fraudster; the other links just bring contacts onto the map
    fraud_links = np.array([(v, u) for v, u in links if y[v] == 1]).T
    print(f"{len(nodes):,} users on the map, {fraud_links.shape[1]:,} fraudster-contact edges")

    xys, aucs = [], []
    init: np.ndarray | str = "pca"
    for i, h in enumerate(layers):
        aucs.append(probe_auc(h, g, rng))
        z = StandardScaler().fit_transform(h[nodes])
        xy = TSNE(n_components=2, perplexity=30, init=init, random_state=args.seed).fit_transform(z)
        # the next map starts from this layout, scaled as t-SNE expects of an initialization
        init = (xy - xy.mean(0)) / xy[:, 0].std() * 1e-4
        # each panel spans [-1, 1] on its longer side
        xy = xy - (xy.max(0) + xy.min(0)) / 2
        xys.append(xy / np.abs(xy).max())
        print(f"h{i}: probe test AUC {aucs[-1]:.3f}")

    # how close contacts are in each standardized state, relative to random pairs of users
    real = g.edge_index.numpy()
    a, b = real[:, rng.choice(real.shape[1], size=200_000, replace=False)]
    r = rng.choice(n, size=(2, 200_000))
    touches_fraud = (y[a] == 1) | (y[b] == 1)
    ratios = []
    for i, h in enumerate(layers):
        h = (h - h.mean(0)) / (h.std(0) + 1e-6)
        base = np.median(np.linalg.norm(h[r[0]] - h[r[1]], axis=1))
        d = np.linalg.norm(h[a] - h[b], axis=1)
        ratios.append((np.median(d) / base, np.median(d[touches_fraud]) / base))
        print(f"h{i}: contact / random distance {ratios[-1][0]:.3f}, with a fraudster {ratios[-1][1]:.3f}")
    return {
        "nodes": nodes,
        "label": y[nodes],
        "seed": np.isin(nodes, seeds),
        "edges": np.searchsorted(nodes, fraud_links),
        "xy": np.stack(xys),
        "auc": np.array(aucs),
        "ratio": np.array(ratios),
    }


# -------------------------------------------------------------------------------
# Fraud share surface
# -------------------------------------------------------------------------------
def share_surface(xy: np.ndarray, fraud: np.ndarray, bandwidth: float) -> tuple[np.ndarray, np.ndarray]:
    """Kernel-weighted fraud share of the sampled test users on a grid, and the kernel mass."""
    axis = np.linspace(-LIMIT, LIMIT, GRID)
    gx, gy = np.meshgrid(axis, axis)
    cells = np.column_stack([gx.ravel(), gy.ravel()])
    w = np.exp(-((cells[:, None, :] - xy[None, :, :]) ** 2).sum(-1) / (2 * bandwidth**2))
    mass = w.sum(1)
    share = (w * fraud).sum(1) / np.maximum(mass, 1e-12)
    return share.reshape(GRID, GRID), mass.reshape(GRID, GRID)


def share_image(share: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """RGBA image of the share, fading out where few test users are nearby."""
    rgba = SHARE_MAP(np.clip(share / SHARE_MAX, 0, 1))
    rgba[..., 3] = 0.55 * np.clip(mass / np.quantile(mass, 0.6), 0, 1)
    return rgba


# -------------------------------------------------------------------------------
# Outputs
# -------------------------------------------------------------------------------
def draw_panel(ax, xy: np.ndarray, m: dict[str, np.ndarray], image: np.ndarray, scale: float) -> None:
    """Shading, fraudster edges and users on one panel; scale multiplies marker sizes and line widths."""
    lab = m["label"]
    ax.imshow(image, origin="lower", extent=(-LIMIT, LIMIT, -LIMIT, LIMIT), zorder=0)
    a, b = m["edges"]
    segs = np.stack([xy[a], xy[b]], axis=1)
    ax.add_collection(LineCollection(list(segs), colors=INK, linewidths=0.35 * scale, alpha=0.35, zorder=1))
    ax.scatter(*xy[lab < 0].T, s=3 * scale**2, c=BACKGROUND, lw=0, zorder=2, label="unlabeled contact")
    ax.scatter(*xy[lab == 0].T, s=5 * scale**2, c=NORMAL, lw=0, zorder=3, label="normal")
    fraud = xy[lab == 1].T
    ax.scatter(
        *fraud, s=11 * scale**2, c=FRAUD, edgecolors="white", linewidths=0.3 * scale, zorder=4, label="fraudster"
    )
    ax.set_xlim(-LIMIT, LIMIT)
    ax.set_ylim(-LIMIT, LIMIT)
    ax.axis("off")


def export(m: dict[str, np.ndarray], images: list[np.ndarray]) -> None:
    """One raster image per panel and the macros for the paper; pgfplots adds titles and the colorbar."""
    OUT.mkdir(parents=True, exist_ok=True)
    side = 4.6 / 2.54
    for i, xy in enumerate(m["xy"]):
        fig = plt.figure(figsize=(side, side))
        ax = fig.add_axes((0, 0, 1, 1))
        draw_panel(ax, xy, m, images[i], scale=side / 5)
        fig.savefig(OUT / f"layer_{i}_map.png", dpi=600, transparent=True)
        plt.close(fig)
    names = ["Zero", "One", "Two"]
    macros = [f"\\providecommand{{\\LayerAUC{names[i]}}}{{{auc:.3f}}}" for i, auc in enumerate(m["auc"])]
    for i, (all_edges, fraud_edges) in enumerate(m["ratio"]):
        macros.append(f"\\providecommand{{\\LayerDist{names[i]}}}{{{all_edges:.2f}}}")
        macros.append(f"\\providecommand{{\\LayerDistFraud{names[i]}}}{{{fraud_edges:.2f}}}")
    n_fraud = int(((m["label"] == 1) & m["seed"]).sum())
    n_seed = int(m["seed"].sum())
    macros += [
        f"\\providecommand{{\\LayerMapFraud}}{{{n_fraud:,}}}".replace(",", "{,}"),
        f"\\providecommand{{\\LayerMapSeeds}}{{{n_seed:,}}}".replace(",", "{,}"),
        f"\\providecommand{{\\LayerMapUsers}}{{{len(m['nodes']):,}}}".replace(",", "{,}"),
        f"\\providecommand{{\\LayerMapShareMax}}{{{SHARE_MAX:.1f}}}",
    ]
    (OUT / "layer_numbers.tex").write_text("\n".join(macros) + "\n")
    print(f"wrote panels and macros to {OUT}")


def preview(m: dict[str, np.ndarray], images: list[np.ndarray]) -> None:
    """The same figure in matplotlib, for checking before a LaTeX build."""
    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5.6))
    for i, ax in enumerate(axes):
        draw_panel(ax, m["xy"][i], m, images[i], scale=1.0)
        ax.set_title(f"{TITLES[i]}\nprobe test AUC {m['auc'][i]:.3f}", fontsize=11)
    axes[0].legend(loc="lower left", frameon=False, fontsize=9, markerscale=2)
    fig.savefig(paths.OUTPUTS / "layer_map.png", dpi=200, bbox_inches="tight")
    print(f"wrote {paths.OUTPUTS / 'layer_map.png'}")


# -------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fraud", type=int, default=300, help="test fraudsters on the map")
    ap.add_argument("--normal", type=int, default=900, help="test normal users on the map")
    ap.add_argument("--cap", type=int, default=6, help="contacts kept per sampled user")
    ap.add_argument("--bandwidth", type=float, default=0.08, help="kernel width for the fraud share, in map units")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recompute", action="store_true", help="rebuild the cached layouts and probes")
    args = ap.parse_args()

    if args.recompute or not CACHE.exists():
        c = compute(args)
        np.savez(
            CACHE,
            nodes=c["nodes"],
            label=c["label"],
            seed=c["seed"],
            edges=c["edges"],
            xy=c["xy"],
            auc=c["auc"],
            ratio=c["ratio"],
        )
    m = dict(np.load(CACHE))
    images = []
    for xy in m["xy"]:
        tested = m["seed"]
        share, mass = share_surface(xy[tested], (m["label"][tested] == 1).astype(float), args.bandwidth)
        images.append(share_image(share, mass))
    preview(m, images)
    export(m, images)


if __name__ == "__main__":
    main()
