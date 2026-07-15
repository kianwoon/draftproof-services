"""N2 tests — DOI/Crossref/URL resolver + source retrieval (network MOCKED).

Governing plan: docs/plans/phase2_entailment_grounding_scope.md §3 (verdict
asymmetry — resolved vs paywalled vs unresolved), §4 (N2 = resolver + retrieval,
cached, fail-open, rate-capped, snapshot), §5 (N2 row), §8 (risk 4 snapshot
determinism).

N2 RETRIEVES source text ready for N3 entailment; it mints NO ``verified`` state
and never changes ``verification_status`` (that is N3/N4). The status taxonomy
here — ``resolved`` / ``paywalled`` / ``unresolved`` — is what feeds the §3
asymmetry, so its exactness is directly tested.

EVERY test is offline: a fake ``http_get`` is injected via ``ResolverDeps``; the
default suite never touches live Crossref/Tavily/HTTP. The one optional live
Crossref smoke is gated behind ``CLAIM_GRAPH_LIVE_SMOKE=1`` and skipped by
default.
"""
from __future__ import annotations

import json
import os

import pytest

from claim_graph import ENTAILMENT_KILL_SWITCH_ENV, KILL_SWITCH_ENV
from claim_graph import sources
from claim_graph.schema import EvidenceNode


# ── Fakes ────────────────────────────────────────────────────────────────────
class _Resp:
    """Minimal requests-like response for the injected ``http_get``."""

    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _crossref_ok(abstract="A study of 35% cost reduction in cohorts.",
                 title="Cohort cost study"):
    msg = {"title": [title], "container-title": ["J. Trials"],
           "published": {"date-parts": [[2021, 4]]}}
    if abstract is not None:
        msg["abstract"] = f"<jats:p>{abstract}</jats:p>"
    return _Resp(200, {"status": "ok", "message": msg})


class _RecordingHttp:
    """Injectable ``http_get`` that records calls and replays scripted responses."""

    def __init__(self, script):
        # script: dict keyed by substring → _Resp, or a single _Resp, or Exception
        self.script = script
        self.calls: list[str] = []

    def __call__(self, url, *, headers=None, timeout=20.0):
        self.calls.append(url)
        if isinstance(self.script, Exception):
            raise self.script
        if isinstance(self.script, _Resp):
            return self.script
        for key, resp in self.script.items():
            if key in url:
                if isinstance(resp, Exception):
                    raise resp
                return resp
        return _Resp(404, {"status": "error"})


def _doi_node(doi="10.1234/abc.def", eid="e_001"):
    return EvidenceNode(id=eid, kind="external_citation", claim_ids=["c_001"],
                        detail={"marker": doi, "locator_type": "doi", "doi": doi,
                                "offset": 0})


def _url_node(url="https://example.org/paper", eid="e_002"):
    return EvidenceNode(id=eid, kind="external_citation", claim_ids=["c_002"],
                        detail={"marker": url, "locator_type": "url", "url": url,
                                "offset": 0})


def _in_text_node(marker="(Smith, 2022)", eid="e_003"):
    return EvidenceNode(id=eid, kind="external_citation", claim_ids=["c_003"],
                        detail={"marker": marker, "locator_type": "in_text",
                                "offset": 0})


def _deps(http, tavily=None):
    return sources.ResolverDeps(http_get=http, tavily_search=tavily)


def _no_cache(tmp_path):
    return sources.SourceCache(cache_dir=tmp_path / "cache")


# ── Crossref DOI resolution taxonomy ─────────────────────────────────────────
def test_crossref_doi_with_abstract_resolves(tmp_path):
    http = _RecordingHttp({"api.crossref.org": _crossref_ok()})
    nodes, coverage, limitations = sources.resolve_sources(
        [_doi_node()], deps=_deps(http), cache=_no_cache(tmp_path))
    rec = nodes[0].detail["resolution"]
    assert rec["status"] == sources.STATUS_RESOLVED
    assert rec["locator_type"] == "doi"
    assert rec["provider"] == "crossref"
    assert "35%" in rec["source_text"]  # JATS tags stripped
    assert rec["title"] == "Cohort cost study"
    assert coverage == {"type": "source_access", "value": ["doi_crossref"]}
    assert limitations == []


def test_crossref_doi_200_without_abstract_is_paywalled(tmp_path):
    http = _RecordingHttp({"api.crossref.org": _crossref_ok(abstract=None)})
    nodes, coverage, limitations = sources.resolve_sources(
        [_doi_node()], deps=_deps(http), cache=_no_cache(tmp_path))
    rec = nodes[0].detail["resolution"]
    assert rec["status"] == sources.STATUS_PAYWALLED
    assert rec.get("source_text") in (None, "")
    # found the work (title present) but no retrievable full text.
    assert rec["title"] == "Cohort cost study"
    assert "paywalled" in limitations
    assert "no_full_text" in limitations


def test_crossref_doi_404_is_unresolved(tmp_path):
    http = _RecordingHttp({"api.crossref.org": _Resp(404, {"status": "error"})})
    nodes, _cov, limitations = sources.resolve_sources(
        [_doi_node()], deps=_deps(http), cache=_no_cache(tmp_path))
    rec = nodes[0].detail["resolution"]
    assert rec["status"] == sources.STATUS_UNRESOLVED
    assert rec["http_status"] == 404
    assert "unresolved_citation" in limitations


# ── URL resolution taxonomy ──────────────────────────────────────────────────
def test_url_body_resolves(tmp_path):
    http = _RecordingHttp({"example.org": _Resp(200, None, text="Full article body text here.")})
    nodes, coverage, limitations = sources.resolve_sources(
        [_url_node()], deps=_deps(http), cache=_no_cache(tmp_path))
    rec = nodes[0].detail["resolution"]
    assert rec["status"] == sources.STATUS_RESOLVED
    assert rec["locator_type"] == "url"
    assert rec["provider"] == "web"
    assert "article body" in rec["source_text"]
    assert coverage == {"type": "source_access", "value": ["public_web"]}
    assert limitations == []


def test_url_403_is_paywalled(tmp_path):
    http = _RecordingHttp({"example.org": _Resp(403, None, text="Forbidden")})
    nodes, _cov, limitations = sources.resolve_sources(
        [_url_node()], deps=_deps(http), cache=_no_cache(tmp_path))
    rec = nodes[0].detail["resolution"]
    assert rec["status"] == sources.STATUS_PAYWALLED
    assert rec["http_status"] == 403
    assert "paywalled" in limitations


def test_url_404_is_unresolved(tmp_path):
    http = _RecordingHttp({"example.org": _Resp(404, None, text="Not found")})
    nodes, _cov, limitations = sources.resolve_sources(
        [_url_node()], deps=_deps(http), cache=_no_cache(tmp_path))
    assert nodes[0].detail["resolution"]["status"] == sources.STATUS_UNRESOLVED
    assert "unresolved_citation" in limitations


# ── in_text is not resolvable without a bibliography ─────────────────────────
def test_in_text_is_unresolved_no_network(tmp_path):
    http = _RecordingHttp({})  # any call would be recorded
    nodes, coverage, limitations = sources.resolve_sources(
        [_in_text_node()], deps=_deps(http), cache=_no_cache(tmp_path))
    rec = nodes[0].detail["resolution"]
    assert rec["status"] == sources.STATUS_UNRESOLVED
    assert http.calls == []  # in_text burns NO network by default
    assert coverage == {"type": "source_access", "value": []}  # no channel used
    assert "outside_source_access" in limitations


def test_tavily_fallback_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_CLAIM_GRAPH_TAVILY_FALLBACK", raising=False)
    called = {"n": 0}

    def _tav(query, **kwargs):
        called["n"] += 1
        return {"results": [{"title": "t", "url": "u", "content": "c"}]}

    http = _RecordingHttp({})
    sources.resolve_sources([_in_text_node()], deps=_deps(http, tavily=_tav),
                            cache=_no_cache(tmp_path))
    assert called["n"] == 0  # fallback OFF → Tavily never called


def test_tavily_fallback_on_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_CLAIM_GRAPH_TAVILY_FALLBACK", "1")
    called = {"n": 0}

    def _tav(query, **kwargs):
        called["n"] += 1
        return {"results": [{"title": "Smith trial", "url": "https://x.org",
                             "content": "cohort cost reduction 35%"}]}

    http = _RecordingHttp({})
    nodes, coverage, _lim = sources.resolve_sources(
        [_in_text_node()], deps=_deps(http, tavily=_tav), cache=_no_cache(tmp_path))
    assert called["n"] == 1
    assert "tavily" in coverage["value"]
    assert nodes[0].detail["resolution"]["status"] == sources.STATUS_RESOLVED


# ── Caching: a hit performs NO network ───────────────────────────────────────
def test_cache_hit_performs_no_network(tmp_path):
    cache = _no_cache(tmp_path)
    http1 = _RecordingHttp({"api.crossref.org": _crossref_ok()})
    sources.resolve_sources([_doi_node()], deps=_deps(http1), cache=cache)
    assert len(http1.calls) == 1
    # Second run, same locator, fresh http mock → must NOT be called.
    http2 = _RecordingHttp({"api.crossref.org": _crossref_ok()})
    nodes, _cov, _lim = sources.resolve_sources([_doi_node()], deps=_deps(http2), cache=cache)
    assert http2.calls == []  # served from disk cache
    assert nodes[0].detail["resolution"]["status"] == sources.STATUS_RESOLVED


def test_error_not_cached(tmp_path):
    cache = _no_cache(tmp_path)
    boom = _RecordingHttp(RuntimeError("network down"))
    sources.resolve_sources([_doi_node()], deps=_deps(boom), cache=cache)
    # An error must NOT poison the cache; a later good run resolves.
    good = _RecordingHttp({"api.crossref.org": _crossref_ok()})
    nodes, _cov, _lim = sources.resolve_sources([_doi_node()], deps=_deps(good), cache=cache)
    assert len(good.calls) == 1
    assert nodes[0].detail["resolution"]["status"] == sources.STATUS_RESOLVED


# ── Snapshot round-trip (frozen fixture, no network) ─────────────────────────
def test_snapshot_round_trip_no_network(tmp_path):
    # First resolve live-ish (mock), snapshot the records.
    http = _RecordingHttp({"api.crossref.org": _crossref_ok()})
    nodes, _cov, _lim = sources.resolve_sources(
        [_doi_node()], deps=_deps(http), cache=_no_cache(tmp_path))
    records = sources.snapshot_from_nodes(nodes)
    snap_path = tmp_path / "snap.json"
    sources.write_snapshot(records, snap_path)
    assert snap_path.is_file()

    # Replay against the snapshot with a mock that would EXPLODE if called.
    snapshot = sources.SourceSnapshot(path=snap_path)
    boom = _RecordingHttp(RuntimeError("must not hit network in eval replay"))
    replay, coverage, _lim2 = sources.resolve_sources(
        [_doi_node()], deps=_deps(boom), cache=_no_cache(tmp_path / "other"),
        snapshot=snapshot)
    assert boom.calls == []
    rec = replay[0].detail["resolution"]
    assert rec["status"] == sources.STATUS_RESOLVED
    assert rec["retrieved_via"] == "snapshot"


# ── Coverage + limitations over a MIX ────────────────────────────────────────
def test_coverage_and_limitations_over_mix(tmp_path):
    http = _RecordingHttp({
        "api.crossref.org": _crossref_ok(),               # doi → resolved
        "paywall.org": _Resp(403, None, text="no"),       # url → paywalled
    })
    nodes, coverage, limitations = sources.resolve_sources(
        [_doi_node(), _url_node(url="https://paywall.org/x"), _in_text_node()],
        deps=_deps(http), cache=_no_cache(tmp_path))
    assert coverage["type"] == "source_access"
    assert coverage["value"] == ["doi_crossref", "public_web"]  # sorted, no tavily
    assert "paywalled" in limitations
    assert "outside_source_access" in limitations  # from the in_text node
    assert limitations == sorted(limitations)  # deterministic ordering


# ── Rate cap stops calls at the cap ──────────────────────────────────────────
def test_rate_cap_stops_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_CLAIM_GRAPH_SOURCE_MAX_CALLS", "2")
    http = _RecordingHttp({"api.crossref.org": _crossref_ok()})
    node_list = [_doi_node(doi=f"10.1234/n{i}", eid=f"e_{i:03d}") for i in range(5)]
    nodes, _cov, limitations = sources.resolve_sources(
        node_list, deps=_deps(http), cache=_no_cache(tmp_path))
    assert len(http.calls) == 2  # capped
    statuses = [n.detail["resolution"]["status"] for n in nodes]
    assert statuses[:2] == [sources.STATUS_RESOLVED, sources.STATUS_RESOLVED]
    # capped nodes stay unresolved, NEVER fabricated/verified.
    assert all(s == sources.STATUS_UNRESOLVED for s in statuses[2:])


# ── Fail-open on injected network exception ──────────────────────────────────
def test_fail_open_on_exception(tmp_path):
    boom = _RecordingHttp(RuntimeError("boom"))
    nodes, coverage, _lim = sources.resolve_sources(
        [_doi_node()], deps=_deps(boom), cache=_no_cache(tmp_path))
    # Build never fails; node keeps an error/unresolved status, not verified.
    rec = nodes[0].detail["resolution"]
    assert rec["status"] in (sources.STATUS_ERROR, sources.STATUS_UNRESOLVED)


def test_resolve_never_raises_on_garbage(tmp_path):
    # Catastrophic-input guard: resolve_sources must return, never raise.
    out, coverage, limitations = sources.resolve_sources(
        [None, {"kind": "external_citation"}, 42], deps=None, cache=_no_cache(tmp_path))
    assert isinstance(out, list)
    assert coverage["type"] == "source_access"


# ── Status taxonomy exactness: the three states are DISTINCT ──────────────────
def test_status_taxonomy_states_are_distinct():
    assert len({sources.STATUS_RESOLVED, sources.STATUS_PAYWALLED,
                sources.STATUS_UNRESOLVED, sources.STATUS_ERROR}) == 4
    assert sources.STATUS_RESOLVED == "resolved"
    assert sources.STATUS_PAYWALLED == "paywalled"
    assert sources.STATUS_UNRESOLVED == "unresolved"


# ── N2 NEVER mints verified / touches verification_status ────────────────────
def test_resolution_does_not_set_verification_status(tmp_path):
    http = _RecordingHttp({"api.crossref.org": _crossref_ok()})
    nodes, _cov, _lim = sources.resolve_sources(
        [_doi_node()], deps=_deps(http), cache=_no_cache(tmp_path))
    rec = nodes[0].detail["resolution"]
    assert "verified" not in json.dumps(rec)  # N2 mints no verdict
    assert "verification_status" not in rec


# ── Determinism: same inputs + same cache → identical records ────────────────
def test_determinism_same_inputs_same_records(tmp_path):
    def _run():
        http = _RecordingHttp({"api.crossref.org": _crossref_ok()})
        nodes, coverage, limitations = sources.resolve_sources(
            [_doi_node(), _url_node()], deps=_deps(http, tavily=None),
            cache=_no_cache(tmp_path / "d1"))
        # url_node has no matching script → 404 unresolved (deterministic)
        return ([n.detail["resolution"] for n in nodes], coverage, limitations)

    a = _run()
    b = _run()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_normalize_locator_doi_and_url():
    assert sources.normalize_locator("https://doi.org/10.1234/AbC.def", "doi") == "10.1234/abc.def"
    assert sources.normalize_locator("DOI:10.1/X", "doi") == "10.1/x"
    assert sources.normalize_locator("HTTPS://Example.ORG/Paper/", "url") == "https://example.org/Paper"


# ── Build-path wiring: coverage attaches; OFF stays byte-identical ───────────
def _seg(text, sid, pid, start):
    return {"sentence_id": sid, "paragraph_id": pid, "start_char": start,
            "end_char": start + len(text), "sentence": text}


class _FakeGateway:
    def __init__(self, responses):
        self.responses = list(responses)
        self.model = "fake/gpt-oss-120b"

    def chat(self, prompt, system=None, response_format=None, **kwargs):
        from types import SimpleNamespace
        item = self.responses.pop(0) if self.responses else "{}"
        return SimpleNamespace(content=item)


def test_build_source_coverage_off_by_default(monkeypatch):
    """Entailment OFF → no source_coverage attached (Phase-1 byte-identical)."""
    monkeypatch.delenv(ENTAILMENT_KILL_SWITCH_ENV, raising=False)
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    from claim_graph import extract
    from claim_graph.build import build_claim_graph_extracted
    extract.clear_cache()
    text = "The trial cut cost by 35% (Smith, 2022)."
    segs = [_seg(text, "s001", "p001", 0)]
    proposals = json.dumps({"claims": [
        {"sentence_id": "s001", "quote": "trial cut cost by 35%",
         "node_type": "CLAIM", "claim_type": "factual"}]})
    container = build_claim_graph_extracted(text, segs, _FakeGateway([proposals]))
    assert "source_coverage" not in container


def test_build_source_coverage_on_in_text_only_no_network(monkeypatch):
    """Entailment ON with only an in_text marker → resolver runs but hits NO
    network (in_text is unresolvable), so the build stays offline and safe."""
    monkeypatch.setenv(ENTAILMENT_KILL_SWITCH_ENV, "1")
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    monkeypatch.delenv("DRAFTPROOF_CLAIM_GRAPH_TAVILY_FALLBACK", raising=False)
    from claim_graph import extract
    from claim_graph.build import build_claim_graph_extracted
    extract.clear_cache()
    text = "The trial cut cost by 35% (Smith, 2022)."
    segs = [_seg(text, "s001", "p001", 0)]
    proposals = json.dumps({"claims": [
        {"sentence_id": "s001", "quote": "trial cut cost by 35%",
         "node_type": "CLAIM", "claim_type": "factual"}]})
    container = build_claim_graph_extracted(text, segs, _FakeGateway([proposals]))
    assert "source_coverage" in container
    cov = container["source_coverage"]
    assert cov["type"] == "source_access"
    # in_text only → no channel used, outside_source_access limitation.
    assert cov["value"] == []
    assert "outside_source_access" in (container.get("source_limitations") or [])
    # evidence node carries the resolution block; status is unverified (N4 mints).
    ev = container["evidence"][0]
    assert ev["detail"]["resolution"]["status"] == sources.STATUS_UNRESOLVED
    for c in container["claims"]:
        assert c["verification_status"] != "verified"


# ── Optional LIVE Crossref smoke (skipped unless CLAIM_GRAPH_LIVE_SMOKE=1) ────
@pytest.mark.skipif(
    os.environ.get("CLAIM_GRAPH_LIVE_SMOKE", "0").strip().lower() in {"", "0", "false", "no", "off"},
    reason="live Crossref smoke gated behind CLAIM_GRAPH_LIVE_SMOKE=1",
)
def test_live_crossref_smoke(tmp_path):
    node = _doi_node(doi="10.1145/3442188.3445922")
    nodes, coverage, _lim = sources.resolve_sources(
        [node], deps=None, cache=_no_cache(tmp_path))
    rec = nodes[0].detail["resolution"]
    assert rec["status"] in (sources.STATUS_RESOLVED, sources.STATUS_PAYWALLED)
    assert rec["provider"] == "crossref"
