"""N3 tests — claim↔source entailment engine (scorer + Modal MOCKED, offline).

Governing plan: docs/plans/phase2_entailment_grounding_scope.md §3 (the verdict
asymmetry — verified vs contradicted with a STRICTER contradicted threshold; the
gray zone falls to unverified, NEVER to contradicted), §4 pipeline (N3 =
entailment engine; premise = retrieved source span, hypothesis = normalised
claim), §5 N3 row, [G1] (NLI cross-encoder on Modal; the LLM may normalise
hypotheses but NEVER grants verified).

Milestone boundary (CRITICAL): N3 COMPUTES the entailment result and ATTACHES it
to the evidence node's ``detail["resolution"]["entailment"]``. N3 does NOT set
``claim.verification_status`` to verified/contradicted — that wire-back +
interrogatability re-gating is N4. These tests assert that boundary directly.

EVERY test is offline: a fake ``EntailmentScorer`` is injected; the default
suite never touches live Modal/HTTP.
"""
from __future__ import annotations

import json

import pytest

from claim_graph import ENTAILMENT_KILL_SWITCH_ENV, KILL_SWITCH_ENV
from claim_graph import entailment
from claim_graph import sources
from claim_graph.schema import EvidenceNode


# ── Fake scorer ──────────────────────────────────────────────────────────────
class _FakeScorer:
    """Injectable EntailmentScorer. Scripted per-pair softmax probs, or None to
    simulate a fail-open (endpoint unavailable) result."""

    def __init__(self, results=None, *, fail=False):
        # results: list of {"entailment","neutral","contradiction"} per pair, OR
        #   a callable(pairs) -> list, OR a constant dict applied to every pair.
        self._results = results
        self._fail = fail
        self.calls: list = []

    def score_pairs(self, pairs):
        self.calls.append(list(pairs))
        if self._fail:
            return None
        if callable(self._results):
            return self._results(pairs)
        if isinstance(self._results, dict):
            return [dict(self._results) for _ in pairs]
        return list(self._results or [])


def _probs(e, n, c):
    return {"entailment": e, "neutral": n, "contradiction": c}


def _thresholds(verified=0.65, contradicted=0.8, max_passages=12, min_passage_chars=1):
    return {"verified_threshold": verified, "contradicted_threshold": contradicted,
            "max_passages": max_passages, "min_passage_chars": min_passage_chars,
            "status": "provisional", "calibration_version": None}


# ── Threshold → status mapping (the calibratable core) ───────────────────────
def test_classify_verified_boundary():
    th = _thresholds(verified=0.65, contradicted=0.8)
    # Just below verified → not verified.
    assert entailment.classify(0.649, 0.0, th) == entailment.STATUS_UNVERIFIED
    # Exactly at verified, dominant → verified.
    assert entailment.classify(0.65, 0.10, th) == entailment.STATUS_VERIFIED
    assert entailment.classify(0.90, 0.05, th) == entailment.STATUS_VERIFIED


def test_classify_contradicted_boundary():
    th = _thresholds(verified=0.65, contradicted=0.8)
    # Just below the stricter contradicted bar → unverified (abstain).
    assert entailment.classify(0.0, 0.799, th) == entailment.STATUS_UNVERIFIED
    # At/above → contradicted.
    assert entailment.classify(0.0, 0.8, th) == entailment.STATUS_CONTRADICTED
    assert entailment.classify(0.10, 0.95, th) == entailment.STATUS_CONTRADICTED


def test_classify_requires_dominance_for_verified():
    th = _thresholds(verified=0.65, contradicted=0.8)
    # Entailment clears its bar but does NOT dominate contradiction → not verified.
    assert entailment.classify(0.66, 0.70, th) == entailment.STATUS_UNVERIFIED


def test_classify_asymmetry_mid_score_never_contradicts():
    """A mid score that would pass verified's bar must NOT trigger contradicted.
    contradicted_threshold is STRICTLY greater than verified_threshold (§3)."""
    th = _thresholds(verified=0.65, contradicted=0.8)
    mid = 0.70  # passes verified's 0.65 bar
    # As an entailment score → verified.
    assert entailment.classify(mid, 0.10, th) == entailment.STATUS_VERIFIED
    # The SAME magnitude as a contradiction score → below 0.8 → unverified, NOT contradicted.
    assert entailment.classify(0.10, mid, th) == entailment.STATUS_UNVERIFIED
    assert th["contradicted_threshold"] > th["verified_threshold"]


def test_classify_gray_zone_is_unverified():
    th = _thresholds(verified=0.65, contradicted=0.8)
    assert entailment.classify(0.5, 0.5, th) == entailment.STATUS_UNVERIFIED
    assert entailment.classify(0.6, 0.6, th) == entailment.STATUS_UNVERIFIED


def test_loaded_thresholds_enforce_asymmetry():
    th = entailment.load_thresholds()
    assert th["contradicted_threshold"] > th["verified_threshold"]
    assert th["status"] == "provisional"
    assert th["calibration_version"] is None


# ── Source windowing (deterministic, capped, evidence selection) ─────────────
def test_window_source_splits_sentences_deterministic():
    src = "First fact here. Second fact follows. Third and final fact."
    a, trunc_a = entailment.window_source(src, max_passages=12, min_passage_chars=1)
    b, trunc_b = entailment.window_source(src, max_passages=12, min_passage_chars=1)
    assert a == b  # deterministic
    assert trunc_a is False
    assert len(a) == 3
    assert a[0].startswith("First")


def test_window_source_truncates_at_cap():
    src = " ".join(f"Sentence number {i} here." for i in range(30))
    passages, truncated = entailment.window_source(src, max_passages=5, min_passage_chars=1)
    assert len(passages) == 5
    assert truncated is True


def test_window_source_drops_short_fragments():
    src = "Hi. This is a genuinely long enough passage to keep."
    passages, _ = entailment.window_source(src, max_passages=12, min_passage_chars=20)
    assert all(len(p) >= 20 for p in passages)
    assert any("genuinely long enough" in p for p in passages)


def test_window_source_empty_is_no_passages():
    passages, truncated = entailment.window_source("", max_passages=12, min_passage_chars=1)
    assert passages == []
    assert truncated is False


# ── Aggregation across passages (max-entailment / max-contradiction) ─────────
def test_entail_aggregates_max_entailment_passage():
    src = "Alpha unrelated. Beta the cohort cost fell 35 percent. Gamma noise."
    # passage index 1 (Beta...) carries the strong entailment.
    def _score(pairs):
        out = []
        for premise, _hyp in pairs:
            if "Beta" in premise:
                out.append(_probs(0.92, 0.05, 0.03))
            else:
                out.append(_probs(0.10, 0.85, 0.05))
        return out

    scorer = _FakeScorer(_score)
    res = entailment.entail_claim("Cohort cost fell 35%.", src, scorer,
                                  thresholds=_thresholds(min_passage_chars=1))
    assert res["status"] == entailment.STATUS_VERIFIED
    assert res["entailment_score"] == pytest.approx(0.92)
    assert res["evidence_passage_index"] == 1
    assert res["passages_scored"] == 3
    assert res["label"] == "entailment"


def test_entail_aggregates_max_contradiction_passage():
    src = "Alpha neutral bit. Beta the cohort cost ROSE sharply, not fell."
    def _score(pairs):
        out = []
        for premise, _hyp in pairs:
            if "ROSE" in premise:
                out.append(_probs(0.03, 0.05, 0.92))
            else:
                out.append(_probs(0.10, 0.85, 0.05))
        return out

    scorer = _FakeScorer(_score)
    res = entailment.entail_claim("Cohort cost fell 35%.", src, scorer,
                                  thresholds=_thresholds(min_passage_chars=1))
    assert res["status"] == entailment.STATUS_CONTRADICTED
    assert res["contradiction_score"] == pytest.approx(0.92)
    assert res["evidence_passage_index"] == 1
    assert res["label"] == "contradiction"


def test_entail_gray_zone_stays_unverified():
    scorer = _FakeScorer(_probs(0.55, 0.30, 0.15))  # entailment below 0.65 bar
    res = entailment.entail_claim("A claim.", "A single passage here to score.",
                                  scorer, thresholds=_thresholds(min_passage_chars=1))
    assert res["status"] == entailment.STATUS_UNVERIFIED


# ── Fail-open: scorer returns None → unverified, never crash/fabricate ────────
def test_entail_scorer_none_is_unverified():
    scorer = _FakeScorer(fail=True)
    res = entailment.entail_claim("A claim.", "Some source passage text here.",
                                  scorer, thresholds=_thresholds())
    assert res["status"] == entailment.STATUS_UNVERIFIED
    assert res["reason"] == "endpoint_unavailable"
    assert res["passages_scored"] == 0


def test_entail_empty_source_is_unverified():
    scorer = _FakeScorer(_probs(0.99, 0.0, 0.01))
    res = entailment.entail_claim("A claim.", "", scorer, thresholds=_thresholds())
    assert res["status"] == entailment.STATUS_UNVERIFIED
    assert res["reason"] == "no_source_text"
    assert scorer.calls == []  # nothing to score → scorer not called


def test_entail_truncation_flag_propagates():
    src = " ".join(f"Passage sentence number {i} here." for i in range(30))
    scorer = _FakeScorer(_probs(0.10, 0.85, 0.05))
    res = entailment.entail_claim("A claim.", src, scorer,
                                  thresholds=_thresholds(max_passages=5, min_passage_chars=1))
    assert res["truncated"] is True
    assert res["passages_scored"] == 5


# ── Determinism: same inputs → identical result ──────────────────────────────
def test_entail_determinism():
    src = "Alpha bit. Beta the cohort cost fell 35 percent."
    def _mk():
        return _FakeScorer(lambda pairs: [_probs(0.92, 0.05, 0.03) if "Beta" in p else
                                          _probs(0.10, 0.85, 0.05) for p, _ in pairs])
    a = entailment.entail_claim("Cohort cost fell 35%.", src, _mk(),
                                thresholds=_thresholds(min_passage_chars=1))
    b = entailment.entail_claim("Cohort cost fell 35%.", src, _mk(),
                                thresholds=_thresholds(min_passage_chars=1))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_entail_hypothesis_is_claim_premise_is_source():
    """Premise MUST be the source passage, hypothesis MUST be the claim (§4)."""
    seen = {}
    def _score(pairs):
        seen["pairs"] = list(pairs)
        return [_probs(0.9, 0.05, 0.05) for _ in pairs]

    entailment.entail_claim("THE-CLAIM", "THE-SOURCE passage long enough.",
                            _FakeScorer(_score), thresholds=_thresholds(min_passage_chars=1))
    premise, hypothesis = seen["pairs"][0]
    assert "THE-SOURCE" in premise
    assert hypothesis == "THE-CLAIM"


# ── Import-light: no modal/torch/requests pulled at engine import ────────────
def test_engine_import_light():
    import subprocess
    import sys
    code = (
        "import claim_graph.entailment\n"
        "import sys\n"
        "bad = [k for k in sys.modules if k.split('.')[0] in ('modal','torch') "
        "or k=='transformers']\n"
        "assert not bad, bad\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


# ── Modal-backed scorer maps endpoint shape + fail-opens ─────────────────────
def test_modal_scorer_fail_open_no_env(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_ENTAILMENT_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("DRAFTPROOF_ENTAILMENT_ENDPOINT_TOKEN", raising=False)
    scorer = entailment.ModalEntailmentScorer()
    assert scorer.score_pairs([("p", "h")]) is None  # unconfigured → None, no crash


def test_modal_scorer_maps_response(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_ENTAILMENT_ENDPOINT_URL", "https://x.modal.run")
    monkeypatch.setenv("DRAFTPROOF_ENTAILMENT_ENDPOINT_TOKEN", "tok")
    captured = {}

    class _Resp:
        status_code = 200
        def json(self):
            return {"results": [{"label": "entailment",
                                 "scores": {"entailment": 0.9, "neutral": 0.06,
                                            "contradiction": 0.04}}]}

    def _post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["auth"] = headers.get("Authorization")
        return _Resp()

    scorer = entailment.ModalEntailmentScorer(http_post=_post)
    out = scorer.score_pairs([("premise text", "hypothesis text")])
    assert out == [{"entailment": 0.9, "neutral": 0.06, "contradiction": 0.04}]
    assert captured["json"] == {"pairs": [{"premise": "premise text",
                                           "hypothesis": "hypothesis text"}]}
    assert captured["auth"] == "Bearer tok"


def test_modal_scorer_non_200_is_none(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_ENTAILMENT_ENDPOINT_URL", "https://x.modal.run")
    monkeypatch.setenv("DRAFTPROOF_ENTAILMENT_ENDPOINT_TOKEN", "tok")

    class _Resp:
        status_code = 500
        def json(self):
            return {}

    scorer = entailment.ModalEntailmentScorer(http_post=lambda *a, **k: _Resp())
    assert scorer.score_pairs([("p", "h")]) is None


# ── Build-path wiring (N3) ───────────────────────────────────────────────────
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


def _resolved_http():
    """A fake http_get that resolves the DOI to a Crossref abstract entailing the claim."""
    class _Resp:
        status_code = 200
        def __init__(self, payload):
            self._p = payload
        def json(self):
            return self._p
    abstract = "<jats:p>The trial cut processing cost by 35% across cohorts.</jats:p>"
    msg = {"title": ["Cohort cost study"], "abstract": abstract}

    def _get(url, *, headers=None, timeout=20.0):
        return _Resp({"status": "ok", "message": msg})

    return _get


def _text_with_doi():
    return "The trial cut cost by 35% (see 10.1234/abc.def)."


def _proposals():
    return json.dumps({"claims": [
        {"sentence_id": "s001", "quote": "trial cut cost by 35%",
         "node_type": "CLAIM", "claim_type": "factual"}]})


def _isolate_cache(monkeypatch, tmp_path):
    """Point the N2 on-disk source cache at a tmp dir so build tests don't share
    (or poison) the real ``poc/.cache`` — mirrors test_sources.py's _no_cache."""
    monkeypatch.setenv("DRAFTPROOF_CLAIM_GRAPH_SOURCE_CACHE_DIR", str(tmp_path / "cache"))


def test_build_entailment_off_byte_identical(monkeypatch, tmp_path):
    """Entailment OFF → engine not called, no entailment block attached."""
    monkeypatch.delenv(ENTAILMENT_KILL_SWITCH_ENV, raising=False)
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    _isolate_cache(monkeypatch, tmp_path)
    from claim_graph import extract
    from claim_graph.build import build_claim_graph_extracted
    extract.clear_cache()
    text = _text_with_doi()
    segs = [_seg(text, "s001", "p001", 0)]
    container = build_claim_graph_extracted(text, segs, _FakeGateway([_proposals()]))
    assert "evidence" not in container or all(
        "entailment" not in (e.get("detail", {}).get("resolution", {}) or {})
        for e in container.get("evidence", []))


def test_build_entailment_on_attaches_result(monkeypatch, tmp_path):
    """Entailment ON + resolved DOI + injected verifying scorer → entailment
    block attached to the resolved evidence node; verification_status UNTOUCHED."""
    monkeypatch.setenv(ENTAILMENT_KILL_SWITCH_ENV, "1")
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    _isolate_cache(monkeypatch, tmp_path)
    from claim_graph import extract, sources as sources_mod
    from claim_graph import build as build_mod
    extract.clear_cache()

    # Inject a resolving http_get + an entailing scorer via the build hooks.
    scorer = _FakeScorer(_probs(0.93, 0.04, 0.03))
    monkeypatch.setattr(build_mod, "_resolver_deps",
                        lambda: sources_mod.ResolverDeps(http_get=_resolved_http()))
    monkeypatch.setattr(build_mod, "_entailment_scorer", lambda: scorer)

    text = _text_with_doi()
    segs = [_seg(text, "s001", "p001", 0)]
    container = build_mod.build_claim_graph_extracted(text, segs, _FakeGateway([_proposals()]))

    ev = [e for e in container.get("evidence", [])
          if e.get("detail", {}).get("locator_type") == "doi"]
    assert ev, "expected a resolved DOI evidence node"
    resolution = ev[0]["detail"]["resolution"]
    assert resolution["status"] == sources.STATUS_RESOLVED
    ent = resolution.get("entailment")
    assert ent is not None, "N3 must attach an entailment block to the resolved node"
    # keyed by claim id → per-claim verdict computed by the engine.
    result = next(iter(ent.values()))
    assert result["status"] == entailment.STATUS_VERIFIED  # scorer entails strongly
    assert result["entailment_score"] == pytest.approx(0.93)
    assert scorer.calls, "scorer must have been called on the resolved source"


def test_build_entailment_subswitch_off_does_not_change_verification_status(
        monkeypatch, tmp_path):
    """N3 sub-switch OFF (N1+N2 only) attaches NO verdict, so N4 promotes nothing.

    (Pre-N4 this asserted N3 never flips status; N4 now DOES promote from a
    verdict — see ``test_build_n4_promotes_*``. The surviving boundary is that
    with the entailment engine disabled there is no verdict to promote from, so
    verification_status stays at its Phase-1 value.)"""
    monkeypatch.setenv(ENTAILMENT_KILL_SWITCH_ENV, "1")
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    monkeypatch.setenv("DRAFTPROOF_CLAIM_GRAPH_ENTAIL", "0")  # N3 off
    _isolate_cache(monkeypatch, tmp_path)
    from claim_graph import extract, sources as sources_mod
    from claim_graph import build as build_mod
    extract.clear_cache()

    scorer = _FakeScorer(_probs(0.99, 0.005, 0.005))  # would-be "verified" — never called
    monkeypatch.setattr(build_mod, "_resolver_deps",
                        lambda: sources_mod.ResolverDeps(http_get=_resolved_http()))
    monkeypatch.setattr(build_mod, "_entailment_scorer", lambda: scorer)

    text = _text_with_doi()
    segs = [_seg(text, "s001", "p001", 0)]
    container = build_mod.build_claim_graph_extracted(text, segs, _FakeGateway([_proposals()]))
    for c in container.get("claims", []):
        assert c["verification_status"] != "verified"
        assert c["verification_status"] != "contradicted"


def test_build_only_resolved_sources_scored(monkeypatch, tmp_path):
    """A paywalled/unresolved node is NOT entailment-scored (stays unverified)."""
    monkeypatch.setenv(ENTAILMENT_KILL_SWITCH_ENV, "1")
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    _isolate_cache(monkeypatch, tmp_path)
    from claim_graph import extract, sources as sources_mod
    from claim_graph import build as build_mod
    extract.clear_cache()

    # http_get returns 404 → DOI unresolved → must NOT be entailment-scored.
    class _R:
        status_code = 404
        def json(self):
            return {"status": "error"}
    scorer = _FakeScorer(_probs(0.99, 0.0, 0.01))
    monkeypatch.setattr(build_mod, "_resolver_deps",
                        lambda: sources_mod.ResolverDeps(http_get=lambda *a, **k: _R()))
    monkeypatch.setattr(build_mod, "_entailment_scorer", lambda: scorer)

    text = _text_with_doi()
    segs = [_seg(text, "s001", "p001", 0)]
    build_mod.build_claim_graph_extracted(text, segs, _FakeGateway([_proposals()]))
    assert scorer.calls == []  # unresolved source never scored


# ── N4: verdict → verification_status promotion + the gaming fix ─────────────
def _interrog(container):
    for s in container.get("signals", []) or []:
        if s.get("signal") == "interrogatability":
            return s
    return None


def _claim_nodes(container):
    return [c for c in container.get("claims", []) if c.get("node_type") == "CLAIM"]


def _build_on(monkeypatch, tmp_path, scorer, http=None):
    """Run the full N1–N4 build with entailment ON and mocked resolver+scorer."""
    monkeypatch.setenv(ENTAILMENT_KILL_SWITCH_ENV, "1")
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    _isolate_cache(monkeypatch, tmp_path)
    from claim_graph import extract, sources as sources_mod
    from claim_graph import build as build_mod
    extract.clear_cache()
    monkeypatch.setattr(build_mod, "_resolver_deps",
                        lambda: sources_mod.ResolverDeps(http_get=http or _resolved_http()))
    monkeypatch.setattr(build_mod, "_entailment_scorer", lambda: scorer)
    text = _text_with_doi()
    segs = [_seg(text, "s001", "p001", 0)]
    return build_mod.build_claim_graph_extracted(text, segs, _FakeGateway([_proposals()]))


def test_build_n4_promotes_verified(monkeypatch, tmp_path):
    """A resolved DOI whose source ENTAILS the claim → verification_status verified."""
    container = _build_on(monkeypatch, tmp_path, _FakeScorer(_probs(0.99, 0.005, 0.005)))
    claims = _claim_nodes(container)
    assert claims and any(c["verification_status"] == "verified" for c in claims)


def test_build_n4_promotes_contradicted(monkeypatch, tmp_path):
    """A resolved DOI whose source CONTRADICTS the claim → verification_status contradicted."""
    container = _build_on(monkeypatch, tmp_path, _FakeScorer(_probs(0.01, 0.0, 0.99)))
    claims = _claim_nodes(container)
    assert claims and any(c["verification_status"] == "contradicted" for c in claims)


def test_build_n4_unverified_verdict_leaves_status(monkeypatch, tmp_path):
    """A gray-zone verdict (neither threshold fires) → claim stays unverified."""
    container = _build_on(monkeypatch, tmp_path, _FakeScorer(_probs(0.5, 0.5, 0.0)))
    for c in _claim_nodes(container):
        assert c["verification_status"] not in ("verified", "contradicted")


def test_promote_precedence_verify_vs_internal_contradiction():
    """Unit: a claim already ``contradicted`` (internal, from edges) is NEVER
    downgraded by a ``verified`` verdict — contradicted wins + a flag note is
    appended (surface the conflict, do not silently overwrite, §I)."""
    from types import SimpleNamespace
    from claim_graph import build as build_mod, entailment
    from claim_graph.schema import ClaimNode

    contradicted = ClaimNode(id="c_001", node_type="CLAIM", text="x",
                             verification_status="contradicted")
    ok = ClaimNode(id="c_002", node_type="CLAIM", text="y",
                   verification_status="unverified")
    node = SimpleNamespace(detail={"resolution": {"entailment": {
        "c_001": {"status": entailment.STATUS_VERIFIED},
        "c_002": {"status": entailment.STATUS_VERIFIED},
    }}})
    build_mod._promote_verification_states([contradicted, ok], [node])
    # conflict: stays contradicted, gains a note
    assert contradicted.verification_status == "contradicted"
    assert any("contradicted always wins" in n for n in contradicted.limitations)
    # clean verified promotion on the non-conflicting claim
    assert ok.verification_status == "verified"


def test_promote_contradicted_overrides_verified_verdict():
    """Unit: contradicted verdict always wins over a verified verdict, any order."""
    from types import SimpleNamespace
    from claim_graph import build as build_mod, entailment
    from claim_graph.schema import ClaimNode

    c = ClaimNode(id="c_001", node_type="CLAIM", text="x", verification_status="unverified")
    node = SimpleNamespace(detail={"resolution": {"entailment": {
        "c_001": {"status": entailment.STATUS_VERIFIED}}}})
    node2 = SimpleNamespace(detail={"resolution": {"entailment": {
        "c_001": {"status": entailment.STATUS_CONTRADICTED}}}})
    build_mod._promote_verification_states([c], [node, node2])
    assert c.verification_status == "contradicted"


def test_build_n4_gaming_fix_credit_differs_by_verification(monkeypatch, tmp_path):
    """THE GAMING FIX (M4 rec (f)1, plan [G3]): identical specifics earn
    specificity credit ONLY when the cited source verifies them. Verified →
    credit > 0 and NO question; unverified → 0 credit + a teacher-probe question."""
    verified = _build_on(monkeypatch, tmp_path, _FakeScorer(_probs(0.99, 0.005, 0.005)))
    unverified = _build_on(monkeypatch, tmp_path, _FakeScorer(_probs(0.5, 0.5, 0.0)))

    v_sig, u_sig = _interrog(verified), _interrog(unverified)
    v_credit = v_sig["components"]["specificity_presence"]
    u_credit = u_sig["components"]["specificity_presence"]
    # Credit differs BY VERIFICATION — the headline assertion.
    assert v_credit > u_credit
    assert u_credit == 0.0
    # verified specific emits no question; unverified specific does.
    assert v_sig["value"]["questions_emitted"] == 0
    assert u_sig["value"]["questions_emitted"] >= 1
    # verified-only mode is recorded on both (entailment ON path).
    assert v_sig.get("specificity_verified_only") is True
    assert u_sig.get("specificity_verified_only") is True


def test_build_n4_fail_open_scorer_none(monkeypatch, tmp_path):
    """Entailment endpoint unavailable (scorer None) → build succeeds, nothing is
    verified, claims stay unverified, verified-only specificity credit is 0."""
    container = _build_on(monkeypatch, tmp_path, None)  # _entailment_scorer → None
    claims = _claim_nodes(container)
    assert claims  # build did not fail
    assert all(c["verification_status"] != "verified" for c in claims)
    assert _interrog(container)["components"]["specificity_presence"] == 0.0


def test_build_n4_determinism(monkeypatch, tmp_path):
    """Two identical N1–N4 builds → byte-identical containers."""
    import json
    a = _build_on(monkeypatch, tmp_path, _FakeScorer(_probs(0.99, 0.005, 0.005)))
    b = _build_on(monkeypatch, tmp_path, _FakeScorer(_probs(0.99, 0.005, 0.005)))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
