#!/bin/bash

set -euo pipefail

echo "====================================="
echo "Starting ChartRetrieval release backend"
echo "====================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE_ROOT="$(dirname "$REPO_ROOT")"

echo "Repo root: $REPO_ROOT"
echo "Workspace root: $WORKSPACE_ROOT"
echo "Python environment: $(which python)"
echo "Python version: $(python --version)"

export HF_HOME="${HF_HOME:-/mnt/share/xujing/hf}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CHARTRETRIEVAL_DATA_ROOT="${CHARTRETRIEVAL_DATA_ROOT:-/mnt/share/public/converted/converted}"
export RETRIEVAL_CKPT="${RETRIEVAL_CKPT:-$REPO_ROOT/retrieval_training/output_4types/best_model.pt}"
export RETRIEVAL_SPLIT_FILE="${RETRIEVAL_SPLIT_FILE:-$REPO_ROOT/retrieval_training/data/train_filtered_thresh_0.92.json}"
export RETRIEVAL_EMBEDDINGS_CACHE_DIR="${RETRIEVAL_EMBEDDINGS_CACHE_DIR:-$REPO_ROOT/retrieval_training/data/embeddings_cache}"
export CHART_TYPES_HIERARCHY_FILE="${CHART_TYPES_HIERARCHY_FILE:-$REPO_ROOT/data_processing/chart_types_hierarchy.json}"
export CHART_METADATA_FILE="${CHART_METADATA_FILE:-$WORKSPACE_ROOT/data/samples_info_200k_new.json}"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

echo ""
echo "Checking required files..."

CHECKPOINT="$RETRIEVAL_CKPT"
METADATA="$CHART_METADATA_FILE"
HIERARCHY="$CHART_TYPES_HIERARCHY_FILE"
SPLIT="$RETRIEVAL_SPLIT_FILE"
DATA_ROOT="$CHARTRETRIEVAL_DATA_ROOT"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi
echo "✓ Checkpoint: $CHECKPOINT"

if [ ! -f "$METADATA" ]; then
    echo "ERROR: Metadata not found: $METADATA"
    exit 1
fi
echo "✓ Metadata: $METADATA"

if [ ! -f "$HIERARCHY" ]; then
    echo "ERROR: Hierarchy file not found: $HIERARCHY"
    exit 1
fi
echo "✓ Hierarchy: $HIERARCHY"

if [ ! -f "$SPLIT" ]; then
    echo "ERROR: Split file not found: $SPLIT"
    exit 1
fi
echo "✓ Split file: $SPLIT"

if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Data root not found: $DATA_ROOT"
    exit 1
fi
echo "✓ Data root: $DATA_ROOT"

echo ""
echo "====================================="
echo "Starting server on port 8005..."
echo "====================================="

uvicorn api_server:app --host 0.0.0.0 --port 8005
