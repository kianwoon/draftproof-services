"""Shared runtime state primitives for rewrite pipeline orchestration."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class RewritePipelineState:
    """Mutable state shared by extracted rewrite orchestration phases.

    The legacy pipeline still owns the full orchestration flow. This state
    object is intentionally small so phases can move out of the monolith
    without passing dozens of loosely related parameters.
    """

    source_text: str
    current_text: str
    original_report: dict | None = None
    current_report: dict | None = None
    summary: dict = field(default_factory=dict)
    stage_timings: list[dict] = field(default_factory=list)


@dataclass
class RewriteScanGateway:
    """Full-scan gateway with copy-safe text cache and visible stats."""

    run_full_scan: Callable[[str], dict]
    enabled: bool = True
    _cache: dict[str, dict] = field(default_factory=dict)
    stats: dict = field(default_factory=lambda: {
        "enabled": True,
        "hits": 0,
        "misses": 0,
        "cached_texts": 0,
    })

    def scan(self, text: str) -> dict:
        cache_key = str(text or "")
        if self.enabled and cache_key in self._cache:
            self.stats["hits"] += 1
            return copy.deepcopy(self._cache[cache_key])
        self.stats["misses"] += 1
        report_dict = self.run_full_scan(cache_key)
        if self.enabled:
            self._cache[cache_key] = copy.deepcopy(report_dict)
            self.stats["cached_texts"] = len(self._cache)
        return report_dict
