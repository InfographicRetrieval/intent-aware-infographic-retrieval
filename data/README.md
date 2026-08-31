# Toy Demo Data

This directory contains the self-contained verification gallery used by the
default retrieval mode.

## Included

- `toy_demo/gallery/`: 26 example chart folders
- `toy_demo/samples_info_toy_demo.json`: toy metadata in the same list-based format as the full dataset
- `toy_demo/toy_split.json`: a minimal split file
- `toy_demo/chart_types_hierarchy.json`: chart-type hierarchy used by the interface
- `toy_demo/toy_manifest.json`: a compact manifest of the selected examples

The toy subset covers all 13 root chart categories in the interface, with two examples per root category.

## Default Use

`interface/backend/start_server.sh` selects these paths automatically when
`RETRIEVAL_MODE=toy`. The equivalent explicit configuration is:

```bash
export CHARTRETRIEVAL_DATA_ROOT=./data/toy_demo/gallery
export CHART_METADATA_FILE=./data/toy_demo/samples_info_toy_demo.json
export CHART_TYPES_HIERARCHY_FILE=./data/toy_demo/chart_types_hierarchy.json
export RETRIEVAL_SPLIT_FILE=./data/toy_demo/toy_split.json
```

## Verification Coverage

- structured five-facet query submission
- deterministic retrieval and result ranking
- chart-type metadata and hierarchy loading
- gallery image and SVG serving
- exemplar selection and frontend/backend data flow

The learned paper retriever uses the same file contracts at corpus scale. The
bundled mode keeps verification fast and deterministic while the paper mode
substitutes learned embeddings and the released checkpoint.

## Expected Gallery Layout

Each sample lives in its own folder under `toy_demo/gallery/`:

```text
toy_demo/gallery/<sample_id>/
  chart.png
  chart.svg
  info.json
```

The metadata file is a JSON list. Each entry follows the same structure as the full dataset, including fields such as:

- `folder_name`
- `chart_type`
- `chart_variation`
- `layout`
- `info`
- `info_json_path`

The split file is a simple JSON object:

```json
{
  "train": ["00000000", "00000001"],
  "val": [],
  "test": []
}
```

## Using A Full Gallery

Mirror the bundled format when configuring the paper retriever:

1. Export each chart example into a folder containing `chart.png`, optionally `chart.svg`, and `info.json`.
2. Build a metadata JSON list with one item per folder.
3. Build a split JSON file listing the sample IDs you want the retriever to index.
4. Point the backend to those files with `CHARTRETRIEVAL_DATA_ROOT`, `CHART_METADATA_FILE`, and `RETRIEVAL_SPLIT_FILE`.

The bundled gallery is the reference implementation of this contract.
