"""DOI/Crossref/URL resolver + source retrieval (Phase-2 N2, Track A).

This is N2 in the Phase-2 entailment pipeline
(``docs/plans/phase2_entailment_grounding_scope.md`` §4). It takes the
``external_citation`` ``EvidenceNode``s that N1 (``citations.py``) emits and, for
each, RESOLVES the locator and RETRIEVES retrievable source text — nothing more.
It mints NO ``verified`` state and never touches ``verification_status``; the
entail verdict is N3/N4. Claims stay ``unverified`` after N2.

Status taxonomy (CRITICAL — feeds the §3 verdict asymmetry). The three terminal
states are DISTINCT and must never be conflated:

  * ``resolved``   — got retrievable source text (Crossref abstract present, or a
                     URL body fetched).
  * ``paywalled``  — the work EXISTS but no retrievable full text/abstract
                     (Crossref 200 with no ``abstract``, or an HTTP 401/403/451).
                     → downstream ``unverified`` + ``paywalled`` limitation.
                     NEVER treated as fabricated.
  * ``unresolved`` — the DOI/URL 404s / does not exist, or an ``in_text`` marker
                     with no bibliography locator. → downstream ``unverified``.
                     ``unresolved`` ≠ ``contradicted``; it does NOT imply
                     fabrication and is NEVER penalised.
  * ``error``      — a transient failure (network/parse). Fail-open, NOT cached,
                     retried on the next run. Treated like ``unresolved`` for
                     limitation purposes.

How ``paywalled`` vs ``unresolved`` is decided:
  Crossref — HTTP 200 + a JATS ``abstract`` → ``resolved``; HTTP 200 with the
    work present but NO abstract → ``paywalled`` (found-but-no-full-text); HTTP
    404 → ``unresolved``; HTTP 401/403 → ``paywalled``; other 4xx → ``unresolved``;
    5xx / exception → ``error``.
  URL — HTTP 200 + non-empty body → ``resolved``; 401/403/451 → ``paywalled``;
    404/410/other 4xx → ``unresolved``; 5xx / exception → ``error``.
  ``in_text`` — not resolvable without a bibliography → ``unresolved``
    (``locator_missing``). A keyword Tavily fallback exists but is OFF by default
    (``DRAFTPROOF_CLAIM_GRAPH_TAVILY_FALLBACK``) so we do not burn a search per
    marker.

Determinism, caching, snapshot:
  * cache  — RUNTIME, read/write on-disk, keyed by NORMALIZED locator. A cache
             hit performs NO network. Only terminal states (resolved/paywalled/
             unresolved) are cached; ``error``/rate-capped are not.
  * snapshot — a FROZEN eval fixture (separate from the live cache). When
             provided it takes priority over cache AND network, so the N6 eval
             harness replays WITHOUT touching the network (risk R4). Build the
             snapshot with ``write_snapshot(snapshot_from_nodes(nodes), path)``.

Safety: rate-capped (env-overridable Crossref/URL cap + the shared Tavily
advanced cap), fail-open (any error → node keeps ``error``/``unresolved`` and the
build never fails). Gating: the caller (``build.py``) only invokes this when
``entailment_enabled()`` is on; the module itself is flag-agnostic like
``citations.link_citations``.

Tavily reuse note: the repo's Tavily client is ``rewrite_pipeline._tavily_search``
(+ its module-global call-budget helpers). It lives in the heavy
``poc/rewrite_pipeline.py`` (imports the whole detection stack), so importing it
eagerly would violate this package's import-light discipline. It is therefore
lazily imported ONLY when the (default-OFF) Tavily fallback fires, and the shared
budget helpers are reused when present. Crossref and plain URL fetch use
``requests`` directly — that is the standard HTTP client, not a re-implemented
Tavily client.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlsplit, urlunsplit

from .schema import EvidenceNode

# ── Status taxonomy ──────────────────────────────────────────────────────────
STATUS_RESOLVED = "resolved"
STATUS_PAYWALLED = "paywalled"
STATUS_UNRESOLVED = "unresolved"
STATUS_ERROR = "error"

# ── §C limitation codes ──────────────────────────────────────────────────────
LIM_PAYWALLED = "paywalled"
LIM_NO_FULL_TEXT = "no_full_text"
LIM_UNRESOLVED = "unresolved_citation"
LIM_OUTSIDE = "outside_source_access"

# ── §C coverage channels ─────────────────────────────────────────────────────
CHANNEL_CROSSREF = "doi_crossref"
CHANNEL_WEB = "public_web"
CHANNEL_TAVILY = "tavily"

_CROSSREF_BASE = "https://api.crossref.org/works/"
_DEFAULT_MAILTO = "support@draftproof.app"  # real product contact (polite pool)
_FALSEY = {"0", "false", "no", "off", ""}

# Cap on retrievable source-text length we keep (keeps records + cache small).
_SOURCE_TEXT_MAX = 4000


# ── Deps (injectable so the whole suite is offline) ──────────────────────────
@dataclass
class ResolverDeps:
    """Network seams. ``http_get(url, *, headers, timeout) -> resp`` where resp
    exposes ``.status_code`` (int), ``.text`` (str) and ``.json()`` (dict). The
    default uses ``requests``; tests inject a fake so nothing hits the wire."""

    http_get: Callable[..., Any]
    tavily_search: Optional[Callable[..., Any]] = None


def _default_http_get(url, *, headers=None, timeout=20.0):
    import requests  # lazy — keeps module import-light

    return requests.get(url, headers=headers or {}, timeout=max(3.0, float(timeout or 20.0)))


def default_deps() -> ResolverDeps:
    return ResolverDeps(http_get=_default_http_get, tavily_search=None)


# ── env helpers ──────────────────────────────────────────────────────────────
def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSEY


def _int_env(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _source_max_calls() -> int:
    """Crossref + URL fetch cap for one resolver run (env-overridable)."""
    return max(0, _int_env("DRAFTPROOF_CLAIM_GRAPH_SOURCE_MAX_CALLS", 25))


def _tavily_max_calls() -> int:
    """Reuse the shared Tavily advanced cap (mirrors rewrite_pipeline)."""
    return max(0, _int_env("DRAFTPROOF_TAVILY_ADVANCED_MAX_CALLS", 3))


def _tavily_fallback_enabled() -> bool:
    return _flag("DRAFTPROOF_CLAIM_GRAPH_TAVILY_FALLBACK", False)


def _crossref_mailto() -> str:
    return _env("DRAFTPROOF_CROSSREF_MAILTO", _DEFAULT_MAILTO).strip() or _DEFAULT_MAILTO


def _crossref_headers() -> dict[str, str]:
    # Crossref polite-pool etiquette: identify with a product contact mailto.
    return {"User-Agent": f"DraftProof/1.0 (+https://draftproof.app; mailto:{_crossref_mailto()})"}


# ── locator normalisation (cache-key stability + determinism) ────────────────
_DOI_PREFIXES = (
    "https://doi.org/", "http://doi.org/",
    "https://dx.doi.org/", "http://dx.doi.org/", "doi:",
)


def normalize_locator(raw: Any, locator_type: str) -> str:
    """Return a canonical cache key for a DOI/URL locator.

    DOI → lowercased, resolver-prefix + trailing punctuation stripped.
    URL → lowercased scheme+host (host only; path case preserved), trailing
    slash + punctuation stripped, fragment dropped.
    """
    s = str(raw or "").strip()
    if locator_type == "doi":
        low = s.lower()
        for pre in _DOI_PREFIXES:
            if low.startswith(pre):
                s = s[len(pre):]
                break
        return s.lower().rstrip("/.,;:)]}\"'")
    if locator_type == "url":
        try:
            parts = urlsplit(s)
            scheme = (parts.scheme or "http").lower()
            netloc = parts.netloc.lower()
            path = parts.path.rstrip("/")
            norm = urlunsplit((scheme, netloc, path, parts.query, ""))
            return norm.rstrip(".,;:)]}\"'")
        except Exception:
            return s.lower()
    return s


# ── record construction ──────────────────────────────────────────────────────
def _record(locator: str, locator_type: str, status: str, *,
            title: Optional[str] = None, source_text: Optional[str] = None,
            container_title: Optional[str] = None, published: Optional[str] = None,
            provider: str = "none", retrieved_via: str = "none",
            http_status: Optional[int] = None, note: Optional[str] = None) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "locator": locator,
        "locator_type": locator_type,
        "status": status,
        "provider": provider,
        "retrieved_via": retrieved_via,
        "http_status": http_status,
    }
    if title:
        rec["title"] = title
    if source_text:
        rec["source_text"] = source_text[:_SOURCE_TEXT_MAX]
    if container_title:
        rec["container_title"] = container_title
    if published:
        rec["published"] = published
    if note:
        rec["note"] = note
    return rec


_JATS_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean_text(value: Any) -> str:
    return _WS_RE.sub(" ", _JATS_TAG_RE.sub(" ", str(value or ""))).strip()


def _first(value: Any) -> Optional[str]:
    if isinstance(value, list):
        for item in value:
            if item:
                return str(item)
        return None
    return str(value) if value else None


def _published(value: Any) -> Optional[str]:
    try:
        parts = (value or {}).get("date-parts") or []
        if parts and isinstance(parts[0], list) and parts[0]:
            return "-".join(str(p) for p in parts[0])
    except Exception:
        pass
    return None


# ── per-locator network resolution ───────────────────────────────────────────
def _resolve_doi(locator: str, deps: ResolverDeps) -> dict[str, Any]:
    from urllib.parse import quote

    url = _CROSSREF_BASE + quote(locator, safe="/")
    resp = deps.http_get(url, headers=_crossref_headers(), timeout=20.0)
    code = int(getattr(resp, "status_code", 0) or 0)
    if code == 404:
        return _record(locator, "doi", STATUS_UNRESOLVED, provider="crossref",
                       retrieved_via=CHANNEL_CROSSREF, http_status=404)
    if code in (401, 403):
        return _record(locator, "doi", STATUS_PAYWALLED, provider="crossref",
                       retrieved_via=CHANNEL_CROSSREF, http_status=code)
    if code >= 500:
        return _record(locator, "doi", STATUS_ERROR, provider="crossref",
                       retrieved_via=CHANNEL_CROSSREF, http_status=code)
    if code >= 400:
        return _record(locator, "doi", STATUS_UNRESOLVED, provider="crossref",
                       retrieved_via=CHANNEL_CROSSREF, http_status=code)
    # 2xx — inspect the message body.
    data = resp.json() or {}
    msg = data.get("message") if isinstance(data, dict) else None
    msg = msg if isinstance(msg, dict) else {}
    title = _first(msg.get("title"))
    container = _first(msg.get("container-title"))
    published = _published(msg.get("published") or msg.get("issued"))
    abstract = _clean_text(msg.get("abstract")) if msg.get("abstract") else ""
    if abstract:
        return _record(locator, "doi", STATUS_RESOLVED, title=title,
                       source_text=abstract, container_title=container,
                       published=published, provider="crossref",
                       retrieved_via=CHANNEL_CROSSREF, http_status=code)
    # Work exists (Crossref knows it) but no retrievable abstract → paywalled.
    return _record(locator, "doi", STATUS_PAYWALLED, title=title,
                   container_title=container, published=published,
                   provider="crossref", retrieved_via=CHANNEL_CROSSREF,
                   http_status=code, note="no_abstract")


def _resolve_url(locator: str, deps: ResolverDeps) -> dict[str, Any]:
    resp = deps.http_get(locator, headers={"User-Agent": "DraftProof/1.0 (+https://draftproof.app)"},
                         timeout=20.0)
    code = int(getattr(resp, "status_code", 0) or 0)
    if code in (401, 403, 451):
        return _record(locator, "url", STATUS_PAYWALLED, provider="web",
                       retrieved_via=CHANNEL_WEB, http_status=code)
    if code >= 500:
        return _record(locator, "url", STATUS_ERROR, provider="web",
                       retrieved_via=CHANNEL_WEB, http_status=code)
    if code >= 400:
        return _record(locator, "url", STATUS_UNRESOLVED, provider="web",
                       retrieved_via=CHANNEL_WEB, http_status=code)
    body = _clean_text(getattr(resp, "text", ""))
    if body:
        return _record(locator, "url", STATUS_RESOLVED, source_text=body,
                       provider="web", retrieved_via=CHANNEL_WEB, http_status=code)
    # 200 but empty body → nothing retrievable.
    return _record(locator, "url", STATUS_PAYWALLED, provider="web",
                   retrieved_via=CHANNEL_WEB, http_status=code, note="empty_body")


# ── on-disk runtime cache ────────────────────────────────────────────────────
def _default_cache_dir() -> Path:
    override = _env("DRAFTPROOF_CLAIM_GRAPH_SOURCE_CACHE_DIR")
    if override:
        return Path(override)
    # ``poc/.cache/claim_graph/sources`` — sibling of the package dir.
    return Path(__file__).resolve().parent.parent / ".cache" / "claim_graph" / "sources"


class SourceCache:
    """Deterministic on-disk cache keyed by NORMALIZED locator. Fail-open."""

    def __init__(self, cache_dir: Any = None):
        self.dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()

    def _path(self, locator: str) -> Path:
        h = hashlib.sha1(locator.encode("utf-8")).hexdigest()
        return self.dir / f"{h}.json"

    def get(self, locator: str) -> Optional[dict[str, Any]]:
        try:
            p = self._path(locator)
            if p.is_file():
                data = json.loads(p.read_text("utf-8"))
                return data if isinstance(data, dict) else None
        except Exception:
            return None
        return None

    def set(self, locator: str, record: dict[str, Any]) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._path(locator).write_text(
                json.dumps(record, ensure_ascii=False, sort_keys=True), "utf-8")
        except Exception:
            pass


# ── frozen eval snapshot (separate from the runtime cache) ───────────────────
_SNAPSHOT_SCHEMA = "claim_graph_source_snapshot_v1"


class SourceSnapshot:
    """A read-only frozen fixture: ``{normalized_locator -> record}``.

    Takes priority over the runtime cache AND the network so the N6 eval harness
    replays deterministically offline (risk R4). Distinct from ``SourceCache``:
    the snapshot is never written during a resolve run."""

    def __init__(self, path: Any = None, records: Optional[dict[str, Any]] = None):
        self.path = path
        self._records = records if records is not None else self._load(path)

    @staticmethod
    def _load(path: Any) -> dict[str, Any]:
        try:
            if path and Path(path).is_file():
                data = json.loads(Path(path).read_text("utf-8"))
                if isinstance(data, dict):
                    # Accept both the wrapped ``{"records": {...}}`` fixture and a
                    # bare ``{locator: record}`` mapping.
                    recs = data.get("records") if "records" in data else data
                    return recs if isinstance(recs, dict) else {}
        except Exception:
            return {}
        return {}

    def get(self, locator: str) -> Optional[dict[str, Any]]:
        rec = self._records.get(locator)
        return dict(rec) if isinstance(rec, dict) else None


def snapshot_from_nodes(nodes: list) -> dict[str, Any]:
    """Collect ``{normalized_locator -> record}`` from resolved nodes for freezing."""
    out: dict[str, Any] = {}
    for node in nodes or []:
        detail = _peek_detail(node)
        if not isinstance(detail, dict):
            continue
        rec = detail.get("resolution")
        if not isinstance(rec, dict):
            continue
        loc = rec.get("locator")
        ltype = rec.get("locator_type")
        if loc and ltype in ("doi", "url"):
            out[normalize_locator(loc, ltype)] = dict(rec)
    return out


def write_snapshot(records_by_locator: dict[str, Any], path: Any) -> None:
    """Persist a frozen snapshot fixture. Deterministic (sorted keys)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": _SNAPSHOT_SCHEMA, "records": dict(records_by_locator or {})}
    p.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), "utf-8")


# ── node detail access (EvidenceNode OR serialised dict) ─────────────────────
def _peek_detail(node: Any) -> Optional[dict[str, Any]]:
    if isinstance(node, EvidenceNode):
        return node.detail
    if isinstance(node, dict):
        return node.get("detail")
    return None


def _writable_detail(node: Any) -> Optional[dict[str, Any]]:
    if isinstance(node, EvidenceNode):
        if node.detail is None:
            node.detail = {}
        return node.detail
    if isinstance(node, dict):
        d = node.get("detail")
        if d is None:
            d = {}
            node["detail"] = d
        return d if isinstance(d, dict) else None
    return None


def _node_kind(node: Any) -> Optional[str]:
    if isinstance(node, EvidenceNode):
        return node.kind
    if isinstance(node, dict):
        return node.get("kind")
    return None


# ── Tavily fallback (default OFF; reuse rewrite_pipeline client + budget) ─────
def _resolve_default_tavily() -> Optional[Callable[..., Any]]:
    """Lazily borrow the repo Tavily client. Heavy import — only on fallback."""
    try:
        from rewrite_pipeline import _tavily_search  # type: ignore
    except Exception:
        return None

    api_key = _env("DRAFTPROOF_TAVILY_API_KEY") or _env("TAVILY_API_KEY")
    if not api_key:
        return None

    def _call(query: str, **kwargs: Any) -> dict[str, Any]:
        return _tavily_search(query, api_key=api_key, max_results=3)

    return _call


def _record_shared_tavily_call() -> None:
    try:
        from rewrite_pipeline import _record_source_search_call  # type: ignore

        _record_source_search_call()
    except Exception:
        pass


def _resolve_in_text_via_tavily(marker: str, tavily: Callable[..., Any]) -> dict[str, Any]:
    query = re.sub(r"[()\[\]]", " ", str(marker or "")).strip()
    payload = tavily(query) or {}
    results = payload.get("results") if isinstance(payload, dict) else None
    top = results[0] if isinstance(results, list) and results else None
    if isinstance(top, dict):
        content = _clean_text(top.get("content"))
        if content:
            return _record(marker, "in_text", STATUS_RESOLVED, source_text=content,
                           title=_first(top.get("title")), provider="tavily",
                           retrieved_via=CHANNEL_TAVILY)
    return _record(marker, "in_text", STATUS_UNRESOLVED, provider="tavily",
                   retrieved_via=CHANNEL_TAVILY, note="tavily_no_result")


# ── limitations bookkeeping ──────────────────────────────────────────────────
def _accumulate_limitations(record: dict[str, Any], limitations: set[str]) -> None:
    status = record.get("status")
    if status == STATUS_PAYWALLED:
        limitations.add(LIM_PAYWALLED)
        if not record.get("source_text"):
            limitations.add(LIM_NO_FULL_TEXT)
    elif status in (STATUS_UNRESOLVED, STATUS_ERROR):
        if record.get("note") == "locator_missing":
            limitations.add(LIM_OUTSIDE)
        else:
            limitations.add(LIM_UNRESOLVED)


# ── public entry ─────────────────────────────────────────────────────────────
def resolve_sources(
    evidence_nodes: list,
    *,
    deps: Optional[ResolverDeps] = None,
    cache: Optional[SourceCache] = None,
    snapshot: Optional[SourceSnapshot] = None,
) -> tuple[list, dict[str, Any], list[str]]:
    """Resolve + retrieve source text for ``external_citation`` evidence nodes.

    Attaches a ``resolution`` block to each node's ``detail`` (N1 keys intact) and
    returns ``(nodes, coverage, limitations)`` where ``coverage`` is the §C
    ``{"type": "source_access", "value": [channels]}`` object and ``limitations``
    is a sorted list of §C codes. Fail-open: NEVER raises."""
    try:
        deps = deps or default_deps()
        cache = cache if cache is not None else SourceCache()
        channels: set[str] = set()
        limitations: set[str] = set()
        max_calls = _source_max_calls()
        tav_max = _tavily_max_calls()
        tav_enabled = _tavily_fallback_enabled()
        tavily = deps.tavily_search
        if tav_enabled and tavily is None:
            tavily = _resolve_default_tavily()
        http_calls = 0
        tav_calls = 0

        for node in evidence_nodes or []:
            if _node_kind(node) != "external_citation":
                continue
            detail = _writable_detail(node)
            if detail is None:
                continue
            ltype = detail.get("locator_type")
            raw = detail.get("doi") or detail.get("url") or detail.get("marker")

            if ltype == "in_text":
                record = None
                if tav_enabled and tavily is not None and tav_calls < tav_max:
                    tav_calls += 1
                    _record_shared_tavily_call()
                    try:
                        record = _resolve_in_text_via_tavily(str(raw or ""), tavily)
                    except Exception as exc:
                        record = _record(str(raw or ""), "in_text", STATUS_ERROR,
                                         provider="tavily", retrieved_via=CHANNEL_TAVILY,
                                         note=str(exc)[:120])
                    channels.add(CHANNEL_TAVILY)
                if record is None:
                    record = _record(str(raw or ""), "in_text", STATUS_UNRESOLVED,
                                     note="locator_missing")
                _accumulate_limitations(record, limitations)
                detail["resolution"] = record
                continue

            if ltype not in ("doi", "url"):
                continue

            loc = normalize_locator(raw, ltype)
            record = None

            # 1) frozen snapshot wins (eval determinism, no network/cache).
            if snapshot is not None:
                hit = snapshot.get(loc)
                if hit is not None:
                    record = dict(hit)
                    record["retrieved_via"] = "snapshot"
                    channels.add(CHANNEL_CROSSREF if ltype == "doi" else CHANNEL_WEB)

            # 2) runtime cache hit → no network.
            if record is None:
                hit = cache.get(loc)
                if hit is not None:
                    record = dict(hit)
                    channels.add(CHANNEL_CROSSREF if ltype == "doi" else CHANNEL_WEB)

            # 3) network (rate-capped).
            if record is None:
                if http_calls >= max_calls:
                    record = _record(loc, ltype, STATUS_UNRESOLVED, note="rate_capped")
                    # rate-capped: no channel used, not cached.
                else:
                    http_calls += 1
                    try:
                        record = (_resolve_doi(loc, deps) if ltype == "doi"
                                  else _resolve_url(loc, deps))
                    except Exception as exc:
                        record = _record(loc, ltype, STATUS_ERROR, note=str(exc)[:120])
                    channels.add(CHANNEL_CROSSREF if ltype == "doi" else CHANNEL_WEB)
                    if record.get("status") in (STATUS_RESOLVED, STATUS_PAYWALLED, STATUS_UNRESOLVED):
                        cache.set(loc, record)

            _accumulate_limitations(record, limitations)
            detail["resolution"] = record

        coverage = {"type": "source_access", "value": sorted(channels)}
        return list(evidence_nodes or []), coverage, sorted(limitations)
    except Exception:
        # Catastrophic fail-open: return inputs untouched, empty coverage.
        return list(evidence_nodes or []), {"type": "source_access", "value": []}, []


__all__ = [
    "STATUS_RESOLVED",
    "STATUS_PAYWALLED",
    "STATUS_UNRESOLVED",
    "STATUS_ERROR",
    "ResolverDeps",
    "SourceCache",
    "SourceSnapshot",
    "default_deps",
    "normalize_locator",
    "resolve_sources",
    "snapshot_from_nodes",
    "write_snapshot",
]
