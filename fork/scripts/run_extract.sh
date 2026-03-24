#!/usr/bin/env bash
# Build and run RoomFormer polygon extraction in Docker.
#
# Usage:
#   bash fork/scripts/run_extract.sh
#
# Expects data/ to already contain the datasets and checkpoints
# (run `python fork/scripts/download_data.py` first).

set -euo pipefail

FORK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${FORK_ROOT}/.." && pwd)"
IMAGE_NAME="roomformer-infer"

echo "Building Docker image '${IMAGE_NAME}'..."
docker build -t "${IMAGE_NAME}" -f "${FORK_ROOT}/Dockerfile.infer" "${REPO_ROOT}"

echo "Running extraction for all datasets..."
docker run --rm --gpus all \
    -v "${FORK_ROOT}/input:/app/fork/input" \
    -v "${FORK_ROOT}/output:/app/fork/output" \
    "${IMAGE_NAME}"

echo "Done. Results in ${FORK_ROOT}/output/"
