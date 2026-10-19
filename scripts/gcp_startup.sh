#!/usr/bin/env bash
# Startup script of the VM that gcp_gpu.sh creates: fetch code and data from the bucket, run every
# job x seed in turn, mirror results to the bucket, then delete this VM (also on failure).
#
# Each job writes to runs/<run>/outputs/<job>_s<seed>/; a job with a done marker there is skipped,
# so restarting a reclaimed spot run continues where it stopped. A job that fails (out of memory,
# say) is logged and the queue moves on. GPU memory is sampled every 30 s into gpu_mem.csv.

set -uo pipefail

md() { curl -sf -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
BUCKET="$(md attributes/bucket)"
RUN="$(md attributes/run)"
JOBS="$(md attributes/jobs)"
SEEDS="$(md attributes/seeds)"
NAME="$(md name)"
ZONE="$(md zone | awk -F/ '{print $NF}')"
REMOTE="gs://$BUCKET/runs/$RUN"
WORK=/opt/gcs
mkdir -p "$WORK/out"
LOG="$WORK/run.log"
exec > >(tee -a "$LOG") 2>&1

sync_logs() {
    gcloud storage cp "$LOG" "$REMOTE/run.log" --quiet || true
    gcloud storage cp "$WORK/gpu_mem.csv" "$REMOTE/gpu_mem.csv" --quiet || true
}
finish() {
    echo "$(date -u +%FT%TZ) finished, deleting $NAME"
    sync_logs
    gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet
}
trap finish EXIT

# -------------------------------------------------------------------------------
# Environment, code and data
# -------------------------------------------------------------------------------
# on the Deep Learning image gcloud is a snap that is not ready this early in boot
for _ in $(seq 60); do gcloud --version >/dev/null 2>&1 && break; sleep 10; done
for _ in $(seq 60); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 30 >"$WORK/gpu_mem.csv" &
(while true; do sleep 60; sync_logs; done) &

curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
uv venv "$WORK/.venv" --python 3.12
uv pip install --python "$WORK/.venv" torch==2.14.1 torch-geometric==2.8.1 scikit-learn==1.9.1 numpy==2.5.4 scipy==1.18.1
PY="$WORK/.venv/bin/python"

# without code, data or a GPU there is nothing to run: stop, and the trap deletes the VM
gcloud storage cp "$REMOTE/code.tar.gz" "$WORK/code.tar.gz" || exit 1
tar -xzf "$WORK/code.tar.gz" -C "$WORK" || exit 1
gcloud storage rsync --recursive "gs://$BUCKET/data" "$WORK/data" || exit 1
"$PY" -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))" || exit 1

# -------------------------------------------------------------------------------
# Jobs
# -------------------------------------------------------------------------------
for seed in $SEEDS; do
    for job in $JOBS; do
        name="${job}_s${seed}"
        if gcloud storage ls "$REMOTE/outputs/$name/done" >/dev/null 2>&1; then
            echo "skip $name (done)"
            continue
        fi
        echo "$(date -u +%FT%TZ) start $name"
        out="$WORK/out/$name"
        mkdir -p "$out"
        read -ra args <<<"$("$PY" "$WORK/scripts/gpu_jobs.py" "$job")"
        # expandable segments let the allocator reuse fragmented cache for the k-NN jobs' 2.8 GB blocks
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DGRAPH_DIR="$WORK/data" GCS_OUTPUTS="$out" PYTHONUNBUFFERED=1 \
            "$PY" "$WORK/scripts/train_embed.py" "${args[@]}" --device cuda --seed "$seed" >"$out/train.log" 2>&1
        status=$?
        tail -n 3 "$out/train.log"
        echo "$(date -u +%FT%TZ) end $name exit=$status"
        [[ "$status" == 0 ]] && date -u +%FT%TZ >"$out/done"
        gcloud storage rsync --recursive "$out" "$REMOTE/outputs/$name" --quiet
    done
done
