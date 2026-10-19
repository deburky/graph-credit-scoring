#!/usr/bin/env bash
# Run the GPU jobs of train_embed.py on one L4 in Google Cloud, with nothing left running afterwards.
#
#   scripts/gcp_gpu.sh init         service account, private bucket, data upload (once)
#   scripts/gcp_gpu.sh up <run>     push the code and start a VM that runs every job x seed in turn,
#                                   mirrors results to the bucket and deletes itself; rerunning a run
#                                   name skips the jobs already finished (spot VMs can be reclaimed)
#   scripts/gcp_gpu.sh logs <run>   the run's log, synced to the bucket every minute
#   scripts/gcp_gpu.sh fetch <run>  copy the run's results to outputs/gcp/<run>/
#   scripts/gcp_gpu.sh status       VMs carrying the graph-credit label
#   scripts/gcp_gpu.sh down         delete every VM carrying the label, and this script's NAT gateway
#
# Jobs and seeds come from JOBS and SEEDS (job names as in gpu_jobs.py), for example
#   JOBS="relsage_real global" SEEDS="1 2" scripts/gcp_gpu.sh up seeds
#
# The DGraph-Fin license does not allow redistribution, so the data goes only to a private bucket
# (uniform access, public access prevention on) that only this script's service account can read.
# Three things stop a VM from outliving its run: the startup script deletes the VM when the jobs end
# or fail, --max-run-duration makes Google delete it after MAX_HOURS whatever happens, and `down`
# deletes anything labelled project=graph-credit. The project forbids external IPs on VMs, so the VM
# reaches PyPI through a Cloud NAT: an existing gateway in the region is reused, otherwise this
# script creates its own and `down` deletes it.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-europe-west4}"
# European zones with L4s, tried in order until one has capacity
ZONES="${GCP_ZONES:-europe-west4-b europe-west4-c europe-west4-a europe-west1-b europe-west1-c europe-west3-a europe-west3-b}"
BUCKET="${GCP_BUCKET:-graph-credit-${PROJECT}}"
DATA="${DGRAPH_DIR:-$ROOT/data/dgraphfin}"
SA_NAME="graph-credit-runner"
SA="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
ROLE="graphCreditSelfDelete"
ROUTER="graph-credit-router"
NAT="graph-credit-nat"
MACHINE="${GCP_MACHINE:-g2-standard-16}"
IMAGE_FAMILY="common-cu129-ubuntu-2204-nvidia-580"
MAX_HOURS="${MAX_HOURS:-14}"
SPOT="${SPOT:-1}"
LABEL="project=graph-credit"
# the global-attention jobs are the slowest, so they go last
JOBS="${JOBS:-relsage_real relsage_knn2 transformer_real transformer_knn2 global}"
SEEDS="${SEEDS:-1 2 3 4}"

# -------------------------------------------------------------------------------
# One-time setup
# -------------------------------------------------------------------------------
ensure_nat() {
    # one NAT gateway per subnet: reuse any in the region (event-gpt's, say), else create ours
    local region="$1"
    if [[ -n "$(gcloud compute routers list --regions="$region" --format='value(name)' 2>/dev/null)" ]]; then
        return 0
    fi
    gcloud compute routers create "$ROUTER" --network=default --region="$region" >/dev/null
    gcloud compute routers nats create "$NAT" --router="$ROUTER" --region="$region" \
        --auto-allocate-nat-external-ips --nat-all-subnet-ip-ranges >/dev/null
}

init() {
    gcloud services enable compute.googleapis.com storage.googleapis.com iam.googleapis.com
    if ! gcloud iam service-accounts describe "$SA" >/dev/null 2>&1; then
        gcloud iam service-accounts create "$SA_NAME" --display-name="graph-credit-scoring GPU VM"
    fi
    if ! gcloud iam roles describe "$ROLE" --project="$PROJECT" >/dev/null 2>&1; then
        gcloud iam roles create "$ROLE" --project="$PROJECT" --title="Delete own VM (graph-credit)" \
            --permissions=compute.instances.delete,compute.instances.get,compute.zoneOperations.get
    fi
    gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$SA" \
        --role="projects/$PROJECT/roles/$ROLE" --condition=None >/dev/null
    if ! gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1; then
        gcloud storage buckets create "gs://$BUCKET" --location="$REGION" --uniform-bucket-level-access \
            --public-access-prevention
    fi
    gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --member="serviceAccount:$SA" \
        --role=roles/storage.objectAdmin >/dev/null
    # the caches train_embed.py reads; the raw .npz is not needed
    gcloud storage cp "$DATA/processed.pt" "gs://$BUCKET/data/processed.pt"
    gcloud storage cp "$DATA/features/real_struct.npy" "gs://$BUCKET/data/features/real_struct.npy"
    gcloud storage cp "$DATA/graphs/knn2.npy" "gs://$BUCKET/data/graphs/knn2.npy"
    echo "ready: gs://$BUCKET, service account $SA"
}

# -------------------------------------------------------------------------------
# Runs
# -------------------------------------------------------------------------------
up() {
    local run="$1"
    if [[ ! "$run" =~ ^[a-z0-9-]+$ ]]; then
        echo "run name must be lowercase letters, digits and dashes (it becomes a VM name and a label)" >&2
        exit 1
    fi
    local tarball
    tarball="$(mktemp -d)/code.tar.gz"
    tar -czf "$tarball" -C "$ROOT" --exclude=__pycache__ scripts
    gcloud storage cp "$tarball" "gs://$BUCKET/runs/$run/code.tar.gz"
    local provisioning=(--provisioning-model=STANDARD)
    if [[ "$SPOT" == "1" ]]; then
        provisioning=(--provisioning-model=SPOT)
    fi
    local err
    err="$(mktemp)"
    for zone in $ZONES; do
        echo "trying $zone"
        ensure_nat "${zone%-*}"
        if gcloud compute instances create "gcs-$run" \
            --zone="$zone" \
            --machine-type="$MACHINE" \
            --accelerator=type=nvidia-l4,count=1 \
            --maintenance-policy=TERMINATE \
            "${provisioning[@]}" \
            --max-run-duration="${MAX_HOURS}h" \
            --instance-termination-action=DELETE \
            --image-project=deeplearning-platform-release \
            --image-family="$IMAGE_FAMILY" \
            --boot-disk-size=100GB \
            --no-address \
            --service-account="$SA" \
            --scopes=cloud-platform \
            --labels="$LABEL,run=$run" \
            --metadata="install-nvidia-driver=True,bucket=$BUCKET,run=$run,jobs=$JOBS,seeds=$SEEDS" \
            --metadata-from-file=startup-script="$ROOT/scripts/gcp_startup.sh" 2>"$err"; then
            echo "started gcs-$run in $zone (deleted after the jobs, or after ${MAX_HOURS}h at most)"
            echo "log: scripts/gcp_gpu.sh logs $run"
            return 0
        fi
        cat "$err" >&2
        # only a stockout is worth another zone; anything else fails the same way everywhere
        if ! grep -qE "ZONE_RESOURCE_POOL_EXHAUSTED|does not have enough resources" "$err"; then
            drop_nat
            exit 1
        fi
    done
    echo "no zone in [$ZONES] had an L4 free" >&2
    drop_nat
    exit 1
}

logs() {
    gcloud storage cat "gs://$BUCKET/runs/$1/run.log"
}

fetch() {
    mkdir -p "$ROOT/outputs/gcp/$1"
    gcloud storage rsync --recursive "gs://$BUCKET/runs/$1/outputs" "$ROOT/outputs/gcp/$1"
    gcloud storage cp "gs://$BUCKET/runs/$1/run.log" "gs://$BUCKET/runs/$1/gpu_mem.csv" "$ROOT/outputs/gcp/$1/"
}

status() {
    gcloud compute instances list --filter="labels.project=graph-credit" \
        --format="table(name,zone.basename(),status,labels.run,creationTimestamp.date('%Y-%m-%d %H:%M'))"
}

drop_nat() {
    # only our own gateway; a NAT holds an external IP that is billed even with no VM behind it
    gcloud compute routers list --filter="name=$ROUTER" --format="value(name,region.basename())" |
        while read -r name region; do
            gcloud compute routers delete "$name" --region="$region" --quiet
        done
}

down() {
    local rows
    rows="$(gcloud compute instances list --filter="labels.project=graph-credit" --format="value(name,zone.basename())")"
    while read -r name zone; do
        [[ -n "$name" ]] && gcloud compute instances delete "$name" --zone="$zone" --quiet
    done <<<"$rows"
    drop_nat
}

cmd="${1:-}"
shift || true
case "$cmd" in
    init) init ;;
    up) up "$@" ;;
    logs) logs "$@" ;;
    fetch) fetch "$@" ;;
    status) status ;;
    down) down ;;
    *) sed -n '2,13p' "$0" ;;
esac
