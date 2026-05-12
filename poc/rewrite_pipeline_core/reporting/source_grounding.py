"""Source-grounding search report layer."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import requests


@dataclass(frozen=True)
class SourceGroundingSearchDeps:
    source_search_enabled: Callable[[], bool]
    source_search_max_calls_per_run: Callable[[], int]
    source_search_calls_used: Callable[[], int]
    source_search_remaining_calls: Callable[[], int]
    float_env: Callable[[str, float], float]
    blocker_operation_plan: Callable[..., dict]
    source_grounding_targets_from_block_decisions: Callable[..., list[dict]]
    citation_reference_search_targets: Callable[..., list[dict]]
    source_grounding_claim_targets: Callable[..., list[dict]]
    source_search_depth_status: Callable[[dict | None, int], dict]
    source_search_domain_list: Callable[..., list[str]]
    source_search_default_exclude_domains: set[str]
    tavily_search: Callable[..., dict]
    normalize_tavily_results: Callable[..., list[dict]]
    source_result_confidence: Callable[[list[dict]], str]


def build_source_grounding_search_layer(
    text: str,
    report_dict: dict | None,
    *,
    max_queries: int | None = None,
    max_results: int | None = None,
    deps: SourceGroundingSearchDeps,
) -> dict:
    if not deps.source_search_enabled():
        return {}
    max_calls_per_run = deps.source_search_max_calls_per_run()
    calls_used_before = deps.source_search_calls_used()
    remaining_calls = deps.source_search_remaining_calls()
    if max_calls_per_run <= 0 or remaining_calls <= 0:
        return {
            "enabled": True,
            "kind": "source_grounding_search",
            "provider": os.environ.get("DRAFTPROOF_SOURCE_SEARCH_PROVIDER", "tavily"),
            "status": "budget_exhausted",
            "auto_apply": False,
            "budget": {
                "max_calls_per_run": 0,
                "hard_max_calls_per_run": 5,
            },
            "claim_targets": [],
            "results": [],
            "policy": [
                "Search results may support public/source-grounded claims after user review.",
                "Search results must not be converted into author-owned observations or lived experience.",
            ],
            "call_accounting": {
                "calls_used_before": calls_used_before,
                "calls_used_after": deps.source_search_calls_used(),
                "remaining_before": remaining_calls,
                "remaining_after": deps.source_search_remaining_calls(),
            },
        }
    requested_queries = max(1, int(max_queries or deps.float_env("DRAFTPROOF_SOURCE_SEARCH_MAX_QUERIES", 2.0)))
    max_queries = min(requested_queries, remaining_calls)
    max_results = max(1, int(max_results or deps.float_env("DRAFTPROOF_SOURCE_SEARCH_MAX_RESULTS", 3.0)))
    block_plan = deps.blocker_operation_plan(text, report_dict or {}, limit=max_queries)
    block_decision_targets = deps.source_grounding_targets_from_block_decisions(
        text,
        report_dict,
        block_plan.get("block_decisions") or [],
        limit=max_queries,
    )
    citation_reference_targets = deps.citation_reference_search_targets(text, report_dict, limit=max_queries)
    claim_targets = deps.source_grounding_claim_targets(text, report_dict, limit=max_queries)
    targets = []
    for target in [*block_decision_targets, *citation_reference_targets, *claim_targets]:
        if not isinstance(target, dict):
            continue
        key = (target.get("citation_label") or target.get("claim") or target.get("query") or "").lower()
        if key and any(
            key == (existing.get("citation_label") or existing.get("claim") or existing.get("query") or "").lower()
            for existing in targets
        ):
            continue
        targets.append(target)
        if len(targets) >= max_queries:
            break
    depth_status = deps.source_search_depth_status(report_dict, len(targets))
    layer = {
        "enabled": True,
        "kind": "source_grounding_search",
        "provider": os.environ.get("DRAFTPROOF_SOURCE_SEARCH_PROVIDER", "tavily"),
        "status": "ready",
        "auto_apply": False,
        "claim_targets": targets,
        "results": [],
        "target_source": (
            "block_decisions"
            if block_decision_targets
            else "citation_references"
            if citation_reference_targets
            else "claim_targets"
        ),
        "citation_reference_targets": citation_reference_targets,
        "block_decision_plan": {
            "enabled": block_plan.get("enabled"),
            "active_blockers": block_plan.get("active_blockers"),
            "block_decisions": block_plan.get("block_decisions"),
        },
        "budget": {
            "requested_queries": requested_queries,
            "max_queries": max_queries,
            "max_calls_per_run": max_calls_per_run,
            "hard_max_calls_per_run": 5,
            "calls_used_before": calls_used_before,
            "remaining_calls_before": remaining_calls,
            "search_depth": depth_status.get("search_depth"),
            "search_depth_source": depth_status.get("source"),
            "chunks_per_source": depth_status.get("chunks_per_source"),
            "include_answer": depth_status.get("include_answer"),
        },
        "policy": [
            "Search results may support public/source-grounded claims after user review.",
            "Search results must not be converted into author-owned observations or lived experience.",
            "Search results must not directly change AI Authorship scoring; grounding remains a separate quality dimension.",
            "Generation may cite or bridge to a source only after the source is relevant and acceptable to the user.",
        ],
        "rewrite_handoff": {
            "allowed": [
                "source-to-claim bridge",
                "narrow an unsupported claim using an identified public source",
                "prepare citation/evidence candidates for user review",
            ],
            "forbidden": [
                "invent author-owned context",
                "invent statistics, dates, names, or source claims not present in the result",
                "auto-apply searched evidence without review",
            ],
        },
    }
    if not targets:
        layer["status"] = "no_claim_targets"
        return layer
    provider = str(layer["provider"] or "tavily").lower()
    if provider != "tavily":
        layer["status"] = "unsupported_provider"
        return layer
    api_key = os.environ.get("TAVILY_API_KEY") or os.environ.get("DRAFTPROOF_TAVILY_API_KEY")
    if not api_key:
        layer["status"] = "missing_api_key"
        return layer
    timeout = deps.float_env("DRAFTPROOF_SOURCE_SEARCH_TIMEOUT", 20.0)
    excluded_domains = set(deps.source_search_domain_list(
        "DRAFTPROOF_SOURCE_SEARCH_EXCLUDE_DOMAINS",
        deps.source_search_default_exclude_domains,
    ))
    included_domains = deps.source_search_domain_list("DRAFTPROOF_SOURCE_SEARCH_INCLUDE_DOMAINS")
    layer["domain_filters"] = {
        "exclude_domains": sorted(excluded_domains),
        "include_domains": included_domains,
    }
    errors = []
    for target in targets:
        try:
            payload = deps.tavily_search(
                target.get("query") or target.get("claim") or "",
                api_key=api_key,
                max_results=max_results,
                timeout=timeout,
                exclude_domains=sorted(excluded_domains),
                include_domains=included_domains,
                search_depth=depth_status.get("search_depth"),
                chunks_per_source=depth_status.get("chunks_per_source"),
                include_answer=bool(depth_status.get("include_answer")),
            )
            sources = deps.normalize_tavily_results(
                payload,
                target.get("claim") or "",
                limit=max_results,
                excluded_domains=excluded_domains,
            )
            layer["results"].append({
                "claim_id": target.get("id"),
                "paragraph_index": target.get("paragraph_index"),
                "query": target.get("query"),
                "search_depth": depth_status.get("search_depth"),
                "source_confidence": deps.source_result_confidence(sources),
                "sources": sources,
            })
        except requests.RequestException as exc:
            errors.append({
                "claim_id": target.get("id"),
                "error": exc.__class__.__name__,
                "message": str(exc)[:240],
            })
    layer["errors"] = errors
    layer["call_accounting"] = {
        "calls_used_before": calls_used_before,
        "calls_used_after": deps.source_search_calls_used(),
        "remaining_before": remaining_calls,
        "remaining_after": deps.source_search_remaining_calls(),
        "calls_this_layer": max(0, deps.source_search_calls_used() - calls_used_before),
    }
    layer["status"] = "completed" if layer["results"] else "search_error"
    return layer
