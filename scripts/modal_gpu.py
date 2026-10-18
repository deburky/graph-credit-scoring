"""Run the GPU experiments of train_embed.py on Modal.

The data stays in a private Modal volume (DGraph-Fin's license does not allow redistribution).
Upload the caches once; they are what train_embed.py reads, so the raw .npz is not needed:

    modal volume create graph-credit-dgraph
    modal volume put graph-credit-dgraph data/dgraphfin/processed.pt data/processed.pt
    modal volume put graph-credit-dgraph data/dgraphfin/features/real_struct.npy data/features/real_struct.npy
    modal volume put graph-credit-dgraph data/dgraphfin/graphs/knn2.npy data/graphs/knn2.npy

Then run jobs (in parallel; each writes to its own output folder in the volume):

    modal run scripts/modal_gpu.py --jobs global,transformer_knn2,relsage_knn2
    modal volume get graph-credit-dgraph outputs/<job> outputs/modal/

Run: modal run scripts/modal_gpu.py --jobs global
     modal run scripts/modal_gpu.py --jobs global --epochs 5   # short trial
"""

import os
import subprocess
from pathlib import Path

import modal

SCRIPTS = Path(__file__).resolve().parent
VOLUME = "graph-credit-dgraph"

JOBS: dict[str, list[str]] = {
    "global": ["--model", "global", "--clip", "1.0"],
    "transformer_knn2": ["--model", "transformer", "--add-knn", "knn2", "--hidden", "32", "--heads", "2"],
    "relsage_knn2": ["--model", "relsage", "--add-knn", "knn2"],
    # the transformer at the relational GraphSAGE's width, to compare the two at equal size
    "transformer_h64": ["--model", "transformer", "--hidden", "64", "--heads", "4"],
    # the contact-graph runs repeated on the GPU, so the +knn2 comparisons share a device
    "relsage_real": ["--model", "relsage"],
    "transformer_real": ["--model", "transformer", "--hidden", "32", "--heads", "2"],
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.1", "torch-geometric==2.8.1", "scikit-learn==1.9.1", "numpy==2.5.4", "scipy==1.18.1")
    .add_local_dir(SCRIPTS, "/root/gcs/scripts", ignore=["__pycache__"])
)
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
app = modal.App("graph-credit-scoring", image=image)


@app.function(gpu="A100-80GB", volumes={"/vol": volume}, timeout=6 * 3600)
def train(job: str, epochs: int) -> str:
    out_dir = f"/vol/outputs/{job}" if epochs == 0 else f"/vol/outputs/smoke/{job}"
    env = {"DGRAPH_DIR": "/vol/data", "GCS_OUTPUTS": out_dir, "PYTHONUNBUFFERED": "1"}
    cmd = ["python", "/root/gcs/scripts/train_embed.py", *JOBS[job], "--device", "cuda"]
    if epochs:
        cmd += ["--epochs", str(epochs)]
    out = subprocess.run(cmd, env={**os.environ, **env}, capture_output=True, text=True)
    log = out.stdout + out.stderr
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, "train.log").write_text(log)
    volume.commit()
    if out.returncode != 0:
        raise RuntimeError(f"{job} failed:\n{log[-3000:]}")
    return log


@app.local_entrypoint()
def main(jobs: str = "global", epochs: int = 0) -> None:
    """epochs > 0 runs a short trial into outputs/smoke/ instead of the full job."""
    names = [j.strip() for j in jobs.split(",") if j.strip()]
    unknown = [j for j in names if j not in JOBS]
    if unknown:
        raise SystemExit(f"unknown jobs {unknown}; choose from {sorted(JOBS)}")
    for job, log in zip(names, train.starmap([(j, epochs) for j in names], return_exceptions=True), strict=True):
        print(f"--- {job}")
        print(log if isinstance(log, str) else f"failed: {log}")
