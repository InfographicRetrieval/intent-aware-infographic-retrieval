#!/bin/bash

set -euo pipefail

echo "====================================="
echo "Starting ChartRetrieval release backend"
echo "====================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TOY_ROOT="$REPO_ROOT/data/toy_demo"

echo "Repo root: $REPO_ROOT"
echo "Python environment: $(which python)"
echo "Python version: $(python --version)"

export RETRIEVAL_MODE="${RETRIEVAL_MODE:-toy}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export CHARTRETRIEVAL_DATA_ROOT="${CHARTRETRIEVAL_DATA_ROOT:-$TOY_ROOT/gallery}"
export RETRIEVAL_CKPT="${RETRIEVAL_CKPT:-$REPO_ROOT/retrieval_training/output_4types/best_model.pt}"
export RETRIEVAL_SPLIT_FILE="${RETRIEVAL_SPLIT_FILE:-$TOY_ROOT/toy_split.json}"
export RETRIEVAL_EMBEDDINGS_CACHE_DIR="${RETRIEVAL_EMBEDDINGS_CACHE_DIR:-$REPO_ROOT/.cache/embeddings}"
export CHART_TYPES_HIERARCHY_FILE="${CHART_TYPES_HIERARCHY_FILE:-$TOY_ROOT/chart_types_hierarchy.json}"
export CHART_METADATA_FILE="${CHART_METADATA_FILE:-$TOY_ROOT/samples_info_toy_demo.json}"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

echo ""
echo "Retrieval mode: $RETRIEVAL_MODE"
echo "Checking retrieval assets..."

CHECKPOINT="$RETRIEVAL_CKPT"
METADATA="$CHART_METADATA_FILE"
HIERARCHY="$CHART_TYPES_HIERARCHY_FILE"
SPLIT="$RETRIEVAL_SPLIT_FILE"
DATA_ROOT="$CHARTRETRIEVAL_DATA_ROOT"

if [ "$RETRIEVAL_MODE" = "paper" ]; then
    if [ ! -f "$CHECKPOINT" ]; then
        echo "ERROR: Paper-model checkpoint not found: $CHECKPOINT"
        exit 1
    fi
    if [ ! -f "$REPO_ROOT/retrieval_training/evaluate_5types_human.py" ]; then
        echo "ERROR: Paper-model retrieval source not found under $REPO_ROOT/retrieval_training"
        exit 1
    fi
    echo "Checkpoint: $CHECKPOINT"
elif [ "$RETRIEVAL_MODE" != "toy" ]; then
    echo "ERROR: RETRIEVAL_MODE must be 'toy' or 'paper'"
    exit 1
fi

if [ ! -f "$METADATA" ]; then
    echo "ERROR: Metadata not found: $METADATA"
    exit 1
fi
echo "Metadata: $METADATA"

if [ ! -f "$HIERARCHY" ]; then
    echo "ERROR: Hierarchy file not found: $HIERARCHY"
    exit 1
fi
echo "Hierarchy: $HIERARCHY"

if [ ! -f "$SPLIT" ]; then
    echo "ERROR: Split file not found: $SPLIT"
    exit 1
fi
echo "Split file: $SPLIT"

if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Data root not found: $DATA_ROOT"
    exit 1
fi
echo "Data root: $DATA_ROOT"

echo ""
echo "====================================="
echo "Starting server on port 8005..."
echo "====================================="

uvicorn api_server:app --host 0.0.0.0 --port 8005
