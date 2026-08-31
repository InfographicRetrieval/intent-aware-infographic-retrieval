# Code for "Show Me the Infographic I Imagine: Intent-Aware Infographic Retrieval for Authoring Support"

This repository contains the supplemental implementation for the paper
"Show Me the Infographic I Imagine: Intent-Aware Infographic Retrieval for
Authoring Support." It includes the interactive authoring interface, backend
services, prompt templates, SVG-processing utilities, and a self-contained
26-item gallery for verifying the retrieval workflow.

## Reproduction Modes

The release supports two modes through the same API and interface.

- `toy` is the default. It uses the bundled gallery and a deterministic
  metadata retriever, so the retrieval, filtering, selection, image serving,
  and interface data flow can be reproduced without downloading model weights.
- `paper` loads the facet-aware BGE-VL retriever used by the full system. Use
  this mode with the paper checkpoint, corpus metadata, split, hierarchy, and
  gallery paths described below.

The toy retriever preserves the structured five-facet query and result API, but
it is a lightweight verification backend rather than the learned model reported
in the quantitative tables.

## Repository Contents

- `interface/src/`: React frontend
- `interface/public/`: frontend static assets
- `interface/backend/`: FastAPI backend, retrieval adapters, prompt files,
  SVG processing, and session management
- `data/toy_demo/`: 26 examples spanning all 13 coarse chart types
- `interface/package-lock.json`: pinned frontend dependency graph
- `interface/backend/requirements.txt`: backend dependencies

## Requirements

- Python 3.10 or later
- Node.js 18 or later
- An OpenAI-compatible API key for conversational generation
- A CUDA-capable GPU for practical use of the paper retriever

Retrieval-only endpoints and the baseline page work in toy mode without an API
key. The chat-based authoring workflow requires one of the model configurations
listed under `Model Configuration`.

## Quick Reproduction

### 1. Install the backend

```bash
cd interface/backend
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Start the bundled retrieval backend

```bash
bash start_server.sh
```

The script defaults to `RETRIEVAL_MODE=toy` and resolves all data paths relative
to this repository. Verify it from another terminal:

```bash
curl http://localhost:8005/api/health
curl -X POST http://localhost:8005/api/baseline/search \
  -F 'query=horizontal bar chart about food consumption' \
  -F 'top_k=5'
```

### 3. Start the interface

```bash
cd interface
npm ci
npm start
```

Open `http://localhost:3000`. The frontend proxies API requests to
`http://localhost:8005`.

## Model Configuration

For chat generation, configure either an OpenAI endpoint:

```bash
export OPENAI_API_KEY=your_key_here
export OPENAI_BASE_URL=https://api.openai.com/v1
```

or the Qwen-compatible DashScope endpoint:

```bash
export DASHSCOPE_API_KEY=your_key_here
export DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

The compatibility aliases `CLOSEAI_API_KEY`, `CLOSEAI_BASE_URL`,
`SILICONFLOW_API_KEY`, and `SILICONFLOW_BASE_URL` remain supported.

## Paper Retriever

Set `RETRIEVAL_MODE=paper` and point the backend to the paper-scale artifacts:

```bash
cd interface/backend
python -m pip install -r requirements-paper.txt
export RETRIEVAL_MODE=paper
export RETRIEVAL_CKPT=/path/to/best_model.pt
export CHARTRETRIEVAL_DATA_ROOT=/path/to/gallery
export CHART_METADATA_FILE=/path/to/samples_info.json
export CHART_TYPES_HIERARCHY_FILE=/path/to/chart_types_hierarchy.json
export RETRIEVAL_SPLIT_FILE=/path/to/evaluation_split.json
export RETRIEVAL_EMBEDDINGS_CACHE_DIR=/path/to/embeddings_cache
bash start_server.sh
```

The gallery layout and metadata schema are documented in
[`data/README.md`](data/README.md). The first run computes and caches gallery
embeddings; subsequent runs reuse that cache.

## Interface Routes

- `#/`: intent-aware retrieval and authoring workflow
- `#/plain-chat`: plain-chat comparison condition
- `#/baseline`: direct text-retrieval view

## Generated Data

- Main-chat sessions: `interface/backend/data/sessions/users/`
- Plain-chat sessions: `interface/backend/data/plainchat/users/`
- Uploaded images: `interface/backend/user_images/`
- Default local caches: `.cache/`

These paths are ignored by Git and can be removed between runs.
