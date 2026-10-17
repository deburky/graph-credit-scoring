"""Data and output locations, shared by every script.

    DGRAPH_DIR   dataset directory holding raw/dgraph.npz (default: data/dgraphfin);
                 caches (processed.pt, features/, graphs/) are written next to it
    GCS_OUTPUTS  where results, logs and predictions go (default: outputs)

To run everything on the bundled synthetic sample:

    export DGRAPH_DIR=data/sample GCS_OUTPUTS=outputs/sample
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / os.environ.get("DGRAPH_DIR", "data/dgraphfin")
OUTPUTS = ROOT / os.environ.get("GCS_OUTPUTS", "outputs")
RAW = DATA / "raw" / "dgraph.npz"
GRAPHS = DATA / "graphs"
FEATS = DATA / "features"
PREDS = OUTPUTS / "preds"
