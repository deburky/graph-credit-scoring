"""The GPU jobs of train_embed.py by name, for the Google Cloud runs of gcp_gpu.sh.

The same jobs as modal_gpu.py. Prints a job's train_embed.py arguments, so the VM's startup
script can run each job with several seeds:

    python scripts/train_embed.py $(python scripts/gpu_jobs.py relsage_real) --device cuda --seed 1

Run: python scripts/gpu_jobs.py <job>
"""

import sys

JOBS: dict[str, list[str]] = {
    "relsage_real": ["--model", "relsage"],
    # these three do not fit a 24 GB L4 without recomputing each layer in the backward pass,
    # and the transformer on real + k-NN edges also needs bfloat16 activations
    "relsage_knn2": ["--model", "relsage", "--add-knn", "knn2", "--checkpoint"],
    "transformer_real": ["--model", "transformer", "--hidden", "32", "--heads", "2"],
    "transformer_knn2": [
        "--model",
        "transformer",
        "--add-knn",
        "knn2",
        "--hidden",
        "32",
        "--heads",
        "2",
        "--checkpoint",
        "--bf16",
    ],
    "transformer_h64": ["--model", "transformer", "--hidden", "64", "--heads", "4"],
    "global": ["--model", "global", "--clip", "1.0", "--checkpoint"],
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in JOBS:
        raise SystemExit(f"usage: gpu_jobs.py <job>, one of {sorted(JOBS)}")
    print(" ".join(JOBS[sys.argv[1]]))


if __name__ == "__main__":
    main()
