"""Deterministic retriever for the bundled 26-item demonstration gallery."""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(value: Any) -> Set[str]:
    return set(TOKEN_RE.findall(str(value).lower().replace("_", " ")))


def _query_parts(query: Any) -> Iterable[tuple[str, float]]:
    if isinstance(query, str):
        yield query, 1.0
        return

    if not isinstance(query, dict):
        yield str(query), 1.0
        return

    for facet in ("content", "style", "layout", "illustration", "chart_type"):
        value = query.get(facet, {})
        if isinstance(value, dict):
            text = str(value.get("query", ""))
            weight = float(value.get("weight", 0.0) or 0.0)
        else:
            text = str(value)
            weight = 1.0
        if text.strip():
            yield text, max(weight, 0.05)


class ToyRetriever:
    """Rank the bundled examples using weighted metadata-token overlap.

    This lightweight implementation exercises the same structured-query and
    result contracts as the paper retriever without downloading model weights.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir).resolve()
        manifest_path = self.data_dir / "toy_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            self.items: List[Dict[str, Any]] = json.load(handle)

        for item in self.items:
            sample_id = item["folder_name"]
            info_path = self.data_dir / "gallery" / sample_id / "info.json"
            with info_path.open("r", encoding="utf-8") as handle:
                info = json.load(handle)
            item["info"] = info
            searchable = " ".join(
                str(value)
                for value in (
                    item.get("root_type", ""),
                    item.get("chart_type", ""),
                    item.get("chart_variation", ""),
                    item.get("layout", ""),
                    info.get("data_source", ""),
                    info.get("image_mode", ""),
                    info.get("background_color", ""),
                )
            )
            item["search_tokens"] = _tokens(searchable)

    @staticmethod
    def _score(item: Dict[str, Any], query: Any) -> float:
        score = 0.0
        total_weight = 0.0
        item_tokens = item["search_tokens"]
        root_tokens = _tokens(item.get("root_type", ""))

        for text, weight in _query_parts(query):
            query_tokens = _tokens(text)
            if not query_tokens:
                continue
            overlap = len(query_tokens & item_tokens) / math.sqrt(
                len(query_tokens) * max(len(item_tokens), 1)
            )
            score += weight * overlap
            total_weight += weight

            # Give an exact coarse chart-type request a stable, visible effect.
            if query_tokens == root_tokens:
                score += weight

        return score / total_weight if total_weight else 0.0

    def retrieve(self, query: Any, k: int = 10) -> List[Dict[str, Any]]:
        ranked = sorted(
            self.items,
            key=lambda item: (-self._score(item, query), item["folder_name"]),
        )[: min(k, len(self.items))]

        return [
            {
                "rank": rank,
                "similarity_score": self._score(item, query),
                "folder_name": item["folder_name"],
                "chart_path": f"{item['folder_name']}/chart.png",
                "chart_type_parent": item.get("root_type", "unknown"),
                "chart_type": item.get("chart_type", "unknown"),
            }
            for rank, item in enumerate(ranked, start=1)
        ]

    def search(self, query: Any, top_k: int = 10, **kwargs: Any) -> Dict[str, Any]:
        started_at = time.time()
        results = self.retrieve(query, k=top_k)
        return {
            "query_info": {"type": "toy_metadata", "timestamp": started_at},
            "results": results,
            "search_stats": {
                "total_found": len(results),
                "search_time": round(time.time() - started_at, 4),
                "filters_applied": {
                    "chart_type_filter": kwargs.get("chart_type_filter"),
                    "layout_filter": kwargs.get("layout_filter"),
                },
            },
        }
