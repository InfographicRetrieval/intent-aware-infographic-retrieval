# Toy Demo Data

This directory contains a small toy demo dataset for the interface.

## Included

- `toy_demo/gallery/`: 26 example chart folders
- `toy_demo/samples_info_toy_demo.json`: toy metadata in the same list-based format as the full dataset
- `toy_demo/toy_split.json`: a minimal split file
- `toy_demo/chart_types_hierarchy.json`: chart-type hierarchy used by the interface
- `toy_demo/toy_manifest.json`: a compact manifest of the selected examples

The toy subset covers all 13 root chart categories in the interface, with two examples per root category.

## Quick Use

From the package root, point the backend to the toy demo with:

```bash
export CHARTRETRIEVAL_DATA_ROOT=./data/toy_demo/gallery
export CHART_METADATA_FILE=./data/toy_demo/samples_info_toy_demo.json
export CHART_TYPES_HIERARCHY_FILE=./data/toy_demo/chart_types_hierarchy.json
export RETRIEVAL_SPLIT_FILE=./data/toy_demo/toy_split.json
```

If you also have the retrieval code and weights available, you can keep the rest of the startup flow unchanged.

## What The Toy Demo Is For

- sanity-checking the interface data flow
- understanding the expected gallery layout
- testing metadata and split-file wiring
- providing a concrete example for adapting external data

This toy demo is intentionally small. It is not meant to reproduce the full retrieval results reported in the paper.

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

## Adapting Your Own ChartGalaxy-Derived Data

If you want to use your own data derived from ChartGalaxy, the simplest path is to mirror the toy demo format:

1. Export each chart example into a folder containing `chart.png`, optionally `chart.svg`, and `info.json`.
2. Build a metadata JSON list with one item per folder.
3. Build a split JSON file listing the sample IDs you want the retriever to index.
4. Point the backend to those files with `CHARTRETRIEVAL_DATA_ROOT`, `CHART_METADATA_FILE`, and `RETRIEVAL_SPLIT_FILE`.

The toy demo is intended to serve as the reference format for that conversion.
