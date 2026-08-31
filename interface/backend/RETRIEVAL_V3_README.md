# Paper Retriever Adapter

`retrieval_v3.py` exposes the facet-aware paper retriever through the same
`search()` contract used by the bundled toy mode.

Install the paper dependencies from this directory:

```bash
python -m pip install -r requirements-paper.txt
```

Then configure the checkpoint, gallery, metadata, hierarchy, split, and cache
paths documented in the repository-level `README.md`, set
`RETRIEVAL_MODE=paper`, and run `bash start_server.sh`.

The model definition and gallery-loading helpers imported by the adapter are in
`../../retrieval_training/evaluate_5types_human.py`.
