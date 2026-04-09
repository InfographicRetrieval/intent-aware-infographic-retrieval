# Supplemental Code

Paper: "Show Me the Infographic I Imagine: Intent-Aware Infographic Retrieval for Authoring Support"

This package contains a cleaned supplemental-code snapshot of the interactive interface used in the paper
"Show Me the Infographic I Imagine: Intent-Aware Infographic Retrieval for Authoring Support."

## Included

- `interface/src/`: React frontend source
- `interface/public/`: frontend static assets
- `interface/backend/`: FastAPI backend, retrieval glue code, prompt files, and session helpers
- `interface/package.json` and `interface/package-lock.json`: frontend dependencies
- `interface/backend/requirements.txt`: backend dependencies

## Intentionally Excluded

- `node_modules/`
- `build/`
- cached Python bytecode
- local session data under `backend/data/`
- uploaded user files under `backend/user_images/`
- local test files and lock files
- local-only `.env.development.local`

## Before You Start

This package contains the interface code, but the retrieval backend still expects a few retrieval assets in addition to the frontend and backend source:

- retrieval checkpoint
- retrieval split file
- chart type hierarchy
- metadata file
- gallery image root

The backend supports environment-variable overrides for those paths, so readers do not have to match the original author's machine exactly.

For a lightweight example dataset and the expected gallery format, see `data/README.md`.

## Required Environment Variables

At least one OpenAI API configuration is needed for chat generation:

- `OPENAI_API_KEY`
- optional: `OPENAI_BASE_URL` (defaults to `https://api.openai.com/v1`)

If you use the Qwen route, set:

- `DASHSCOPE_API_KEY`
- optional: `DASHSCOPE_BASE_URL` (defaults to `https://dashscope.aliyuncs.com/compatible-mode/v1`)

Backward compatibility:

- the backend still accepts `CLOSEAI_API_KEY` / `CLOSEAI_BASE_URL`
- it also still accepts `SILICONFLOW_API_KEY` / `SILICONFLOW_BASE_URL`

Useful path overrides:

- `CHARTRETRIEVAL_DATA_ROOT`
- `RETRIEVAL_CKPT`
- `RETRIEVAL_SPLIT_FILE`
- `RETRIEVAL_EMBEDDINGS_CACHE_DIR`
- `CHART_TYPES_HIERARCHY_FILE`
- `CHART_METADATA_FILE`
- optional: `CUDA_VISIBLE_DEVICES`

## Quick Start

### 1. Install frontend dependencies

```bash
cd interface
npm install
```

### 2. Install backend dependencies

```bash
cd interface/backend
pip install -r requirements.txt
```

### 3. Export API credentials

Example:

```bash
export OPENAI_API_KEY=your_key_here
export OPENAI_BASE_URL=https://api.openai.com/v1
```

If you want to use Qwen via DashScope, also export:

```bash
export DASHSCOPE_API_KEY=your_dashscope_key_here
export DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

To try the included toy demo, also export:

```bash
export CHARTRETRIEVAL_DATA_ROOT=./data/toy_demo/gallery
export CHART_METADATA_FILE=./data/toy_demo/samples_info_toy_demo.json
export CHART_TYPES_HIERARCHY_FILE=./data/toy_demo/chart_types_hierarchy.json
export RETRIEVAL_SPLIT_FILE=./data/toy_demo/toy_split.json
```

### 4. Start backend

```bash
cd interface/backend
bash start_server.sh
```

Backend starts on `http://localhost:8005`.

### 5. Start frontend

In a new terminal:

```bash
cd interface
npm start
```

The frontend proxy is configured for `http://localhost:8005`.

## Available Pages

- `#/` main retrieval workflow
- `#/plain-chat` plain chat page
- `#/baseline` baseline retrieval page

## Notes

- Main-chat sessions are written to `interface/backend/data/sessions/users/`
- Plain-chat sessions are written to `interface/backend/data/plainchat/users/`
- Uploaded user images are written to `interface/backend/user_images/`
- The backend startup script checks the main retrieval assets before launching
- This package removes hard-coded API keys and uses environment variables instead
- For broader portability, the default API configuration targets official OpenAI and official Qwen DashScope endpoints
