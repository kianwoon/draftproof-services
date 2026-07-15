"""Build the N6 real-citation eval slice from REAL Crossref metadata.

Phase-2 N6 (``docs/plans/phase2_entailment_grounding_scope.md`` §6). The M4 gaming
corpus proves only the NEGATIVE side (fabricated citations must earn no credit);
the PERSUADE humans have no bibliographies, so ``verified`` firing correctly is
unmeasurable there. This slice supplies the POSITIVE side with REAL, resolvable
DOIs whose retrieved abstracts are the ground truth.

ANTI-FABRICATION CONTRACT (the critical part — read before touching):
  * We invent NO DOIs, papers, findings, or numbers. Real DOIs come from a live
    Crossref search filtered ``has-abstract:true``; the abstract Crossref returns
    IS the ground-truth source.
  * The ENTAILED claim text is a finding sentence LIFTED VERBATIM from that same
    paper's own retrieved abstract (``extract_finding_sentence``) — never
    paraphrased, so it is provably a substring of the source. Cited with that
    paper's real DOI → pipeline resolves DOI → retrieves the same abstract →
    entails → ``verified``. Honest because the claim IS what the source says.
  * The MISATTRIBUTED claim is a verbatim finding sentence from paper X paired
    with a DIFFERENT real paper Y's DOI (a different topic). Source Y won't entail
    X's claim → ``unverified`` (or ``contradicted`` if Y actively refutes). This is
    the §I citation-stuffing / wrong-source counter.
  * Any candidate whose abstract yields no clean finding sentence is DROPPED, never
    padded with invented content.

Determinism / reproducibility:
  * ``main()`` hits Crossref LIVE once (rate-capped, polite), writes the fixtures
    AND a frozen ``SourceSnapshot`` (via ``sources.resolve_sources`` +
    ``snapshot_from_nodes``) so the eval replays offline (risk R4) without
    re-hitting Crossref.
  * The deterministic utilities (``extract_finding_sentence``, ``assemble_doc``,
    ``build_fixture``) are unit-tested; the live ``main()`` build is a one-shot
    provisioning step, not a test.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Callable, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_POC_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_POC_ROOT, ".."))
for _p in (_REPO_ROOT, _POC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CASES_DIR = os.path.join(_HERE, "real_citation_cases")
SNAPSHOT_PATH = os.path.join(_HERE, "real_citation_snapshot.json")
MANIFEST_PATH = os.path.join(_HERE, "real_citation_manifest.json")
MANIFEST_VERSION = "cg-n6-realcite-1"

CASE_ENTAILED = "entailed"
CASE_MISATTRIBUTED = "misattributed"

# Distinct topics so misattributed cross-pairs are clearly off-source.
TOPICS = (
    "physical activity cardiovascular disease",
    "sleep deprivation memory consolidation",
    "coral reef ocean warming bleaching",
    "microplastics marine food web",
    "vitamin D immune function",
    "urban green space mental health",
    "wildfire smoke respiratory health",
    "remote work productivity outcomes",
)

_MAX_FIXTURES = 15
_CROSSREF_SEARCH = "https://api.crossref.org/works"
_DEFAULT_MAILTO = "support@draftproof.app"

# ── deterministic sentence utilities (unit-tested) ───────────────────────────
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WS_RE = re.compile(r"\s+")
_JATS_TAG_RE = re.compile(r"<[^>]+>")
# JATS/abstract section labels we strip from a leading fragment. Two forms:
#  * label + separator: "Results: participants..."  → "participants..."
#  * bare label + space + Uppercase word: "Objective We aimed..." → "We aimed..."
#    (JATS <title>Objective</title> concatenates onto the sentence once tags are
#    stripped; only strip when an Uppercase word follows so "Objectives of the
#    policy" — a real sentence — is left intact).
_LABELS = (r"background|objectives?|introduction|methods?|materials?|results?"
           r"|findings?|conclusions?|discussion|aims?|purpose|summary")
_LABEL_RE = re.compile(r"^\s*(?:%s)\s*[:.\-–]\s*" % _LABELS, re.IGNORECASE)
_BARE_LABEL_RE = re.compile(r"^\s*(?:%s)\s+(?=[A-Z])" % _LABELS, re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b\d")
_RESULT_CUE_RE = re.compile(
    r"\b(increase|decrease|reduc|higher|lower|associat|significant|greater|"
    r"improv|declin|risk|effect|correlat|more likely|less likely|"
    r"compared|than)\b", re.IGNORECASE)
# Aim/method sentences read as intent, not a finding — a student citing a paper
# states its RESULT, not its protocol; and gpt-oss rarely extracts them as CLAIMs.
_AIM_METHOD_RE = re.compile(
    r"\b(we aim|aimed to|the aim|the goal|the objective|this (?:study|paper|work|"
    r"analysis|review|report|article)|to (?:determine|assess|examine|test|"
    r"investigate|evaluate|explore|estimate|quantify)\b|were (?:trained|recruited|"
    r"enrolled|randomi|sampled)|we (?:recruited|enrolled|conducted|performed|"
    r"measured|sampled|collected)|was (?:conducted|performed|carried out))",
    re.IGNORECASE)
# Contrastive/anaphoric leads are only meaningful WITH prior context — a bad
# standalone claim (drove a false `contradicted` on a conditional "However …").
_CONNECTIVE_LEAD_RE = re.compile(
    r"^(however|in contrast|by contrast|whereas|nevertheless|conversely|yet|"
    r"although|though|moreover|furthermore|additionally|in addition|thus|"
    r"therefore|hence|consequently|on the other hand)\b", re.IGNORECASE)

_MIN_CLAIM_CHARS = 60
_MAX_CLAIM_CHARS = 260


def clean_abstract(raw: Any) -> str:
    """Strip JATS tags + normalise whitespace — BYTE-IDENTICAL to
    ``sources._clean_text`` (NO html.unescape) so the lifted claim is a verbatim
    substring of the resolver's ``source_text`` (the anti-fabrication guarantee).
    Sentences still carrying HTML entities (``&lt;`` etc.) are skipped in
    ``extract_finding_sentence`` instead."""
    return _WS_RE.sub(" ", _JATS_TAG_RE.sub(" ", str(raw or ""))).strip()


def _strip_label(sentence: str) -> str:
    prev = None
    s = sentence.strip()
    while s != prev:
        prev = s
        s = _LABEL_RE.sub("", s).strip()
        s = _BARE_LABEL_RE.sub("", s).strip()
    return s


def _finding_candidates(abstract: Any) -> list[str]:
    """Ordered, label-stripped, in-window sentences that are NOT aim/method/
    connective-lead and carry no leaked HTML entity. Verbatim-preserving."""
    text = clean_abstract(abstract)
    if not text:
        return []
    out: list[str] = []
    for raw in _SENT_SPLIT_RE.split(text):
        s = _strip_label(raw).strip()
        if not (_MIN_CLAIM_CHARS <= len(s) <= _MAX_CLAIM_CHARS):
            continue
        if "&" in s and re.search(r"&(?:lt|gt|amp|#\d+);", s):
            continue  # HTML entity leaked → would diverge from source_text
        if _AIM_METHOD_RE.search(s) or _CONNECTIVE_LEAD_RE.match(s):
            continue  # protocol/intent/anaphoric — not a standalone finding
        out.append(s)
    return out


def extract_finding_sentence(abstract: Any) -> Optional[str]:
    """Return the best substantive FINDING sentence, lifted VERBATIM.

    Deterministic two-pass over ``_finding_candidates`` (already excludes
    aim/method/connective/entity sentences): prefer a sentence carrying BOTH a
    number AND a result cue (a quantified finding), else the first sentence with a
    number OR a result cue. Returns ``None`` when nothing qualifies (→ caller DROPS
    the candidate; never fabricates a claim)."""
    cands = _finding_candidates(abstract)

    def _finalize(s: str) -> str:
        return s if s[-1] in ".!?" else s + "."

    for s in cands:  # pass 1: quantified finding (number AND cue)
        if _NUMBER_RE.search(s) and _RESULT_CUE_RE.search(s):
            return _finalize(s)
    for s in cands:  # pass 2: number OR cue
        if _NUMBER_RE.search(s) or _RESULT_CUE_RE.search(s):
            return _finalize(s)
    return None


def assemble_doc(claim: str, doi: str) -> str:
    """Doc text = neutral lead + the verbatim claim + the cited DOI locator.

    The lead is generic (no specifics of its own) so it never competes as a
    claim; the DOI is appended as a BARE locator (``DOI: 10.x/y``) right after the
    claim so ``citations.link_citations`` classifies it ``doi`` (→ Crossref
    abstract) and links it to the claim's span. A ``https://doi.org/`` wrapper
    would be captured as a URL instead (URL wins the overlap dedup) and resolved
    by HTML fetch of the redirect page, NOT the abstract — so we keep it bare.

    NOTE: no generic framing lead — a competing lead sentence was extracted as its
    own CLAIM and stole the DOI link (nearest-preceding), so the finding never got
    the verdict. The finding claim stands alone as the sole extractable claim."""
    return f"{claim.strip()} (DOI: {doi.strip()})"


def build_fixture(case_id: str, case_type: str, claim: str, cite_doi: str,
                  claim_source_doi: str, claim_source_title: str,
                  cite_title: str) -> dict[str, Any]:
    """Assemble one committed fixture record (schema stable, unit-tested)."""
    return {
        "case_id": case_id,
        "case_type": case_type,             # entailed | misattributed
        "authorship": "eval_fixture_real_crossref",
        "synthetic_adversarial": case_type == CASE_MISATTRIBUTED,
        "source": "crossref_metadata",
        "cite_doi": cite_doi,               # the DOI the doc actually cites
        # Provenance of the CLAIM text (anti-fabrication audit trail):
        "claim_source_doi": claim_source_doi,   # DOI whose abstract the claim was lifted from
        "claim_source_title": claim_source_title,
        "cite_title": cite_title,           # title of the DOI actually cited
        "claim": claim,                     # VERBATIM finding sentence from claim_source_doi
        "words": len(assemble_doc(claim, cite_doi).split()),
        "text": assemble_doc(claim, cite_doi),
        "note": ("entailed: cite_doi == claim_source_doi (claim lifted from that "
                 "paper's own abstract). misattributed: cite_doi is a DIFFERENT "
                 "real paper; the claim is NOT its finding."),
    }


# ── live Crossref candidate collection (network; one-shot provisioning) ───────
def _crossref_headers(mailto: str) -> dict[str, str]:
    return {"User-Agent": f"DraftProof/1.0 (+https://draftproof.app; mailto:{mailto})"}


def _default_get(url, *, params=None, headers=None, timeout=30.0):
    import requests  # lazy
    return requests.get(url, params=params, headers=headers or {},
                        timeout=max(3.0, float(timeout)))


def fetch_candidates(topic: str, http_get: Callable[..., Any], mailto: str,
                     rows: int = 4) -> list[dict[str, Any]]:
    """Live Crossref search → [{doi,title,abstract,finding}] with a usable finding.

    Filters ``has-abstract:true,type:journal-article``; keeps only candidates whose
    abstract yields a finding sentence. Fail-open: returns [] on any error."""
    params = {
        "filter": "has-abstract:true,type:journal-article",
        "query": topic, "rows": str(rows),
        "select": "DOI,title,abstract,container-title",
    }
    try:
        resp = http_get(_CROSSREF_SEARCH, params=params,
                        headers=_crossref_headers(mailto), timeout=30.0)
        if int(getattr(resp, "status_code", 0) or 0) != 200:
            return []
        items = (resp.json() or {}).get("message", {}).get("items", []) or []
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for it in items:
        doi = str(it.get("DOI") or "").strip()
        abstract = clean_abstract(it.get("abstract"))
        finding = extract_finding_sentence(abstract)
        if not doi or not finding:
            continue
        out.append({
            "doi": doi,
            "title": (it.get("title") or [""])[0][:200],
            "abstract": abstract,
            "finding": finding,
            "topic": topic,
        })
    return out


def build_slice(candidates_by_topic: dict[str, list[dict[str, Any]]]
                ) -> list[dict[str, Any]]:
    """Assemble entailed + misattributed fixtures from per-topic candidates.

    Deterministic given the candidate pool. One entailed fixture per topic (its
    first candidate). Misattributed fixtures cross-pair a topic's SECOND candidate
    claim with the NEXT topic's first-candidate DOI (guaranteed off-source)."""
    topics = [t for t in TOPICS if candidates_by_topic.get(t)]
    fixtures: list[dict[str, Any]] = []
    idx = 0

    # Entailed: claim + DOI from the SAME paper (first candidate of each topic).
    for t in topics:
        cand = candidates_by_topic[t][0]
        fixtures.append(build_fixture(
            case_id="RC_%02d" % idx, case_type=CASE_ENTAILED,
            claim=cand["finding"], cite_doi=cand["doi"],
            claim_source_doi=cand["doi"], claim_source_title=cand["title"],
            cite_title=cand["title"]))
        idx += 1

    # Misattributed: claim from topic i's SECOND candidate, DOI from topic i+1's
    # first candidate (different topic → off-source). Falls back to the first
    # candidate's claim if a topic has only one candidate.
    n = len(topics)
    for i, t in enumerate(topics):
        if len(fixtures) >= _MAX_FIXTURES:
            break
        pool = candidates_by_topic[t]
        claim_cand = pool[1] if len(pool) > 1 else pool[0]
        other = candidates_by_topic[topics[(i + 1) % n]][0]
        if other["doi"] == claim_cand["doi"]:
            continue  # never pair a claim with its own paper here
        fixtures.append(build_fixture(
            case_id="RC_%02d" % idx, case_type=CASE_MISATTRIBUTED,
            claim=claim_cand["finding"], cite_doi=other["doi"],
            claim_source_doi=claim_cand["doi"], claim_source_title=claim_cand["title"],
            cite_title=other["title"]))
        idx += 1

    return fixtures[:_MAX_FIXTURES]


def _write_fixtures(fixtures: list[dict[str, Any]]) -> None:
    os.makedirs(CASES_DIR, exist_ok=True)
    for fx in fixtures:
        path = os.path.join(CASES_DIR, fx["case_id"] + ".json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(fx, fh, indent=2, ensure_ascii=False)
            fh.write("\n")


def _build_snapshot(fixtures: list[dict[str, Any]]) -> None:
    """Resolve every cited DOI LIVE once and freeze the records (offline replay)."""
    from claim_graph import sources
    from claim_graph.schema import EvidenceNode

    dois = sorted({fx["cite_doi"] for fx in fixtures})
    nodes = [EvidenceNode(id="e_%03d" % i, kind="external_citation",
                          claim_ids=[], detail={"locator_type": "doi", "doi": d})
             for i, d in enumerate(dois, start=1)]
    resolved, _cov, _lim = sources.resolve_sources(nodes)
    records = sources.snapshot_from_nodes(resolved)
    sources.write_snapshot(records, SNAPSHOT_PATH)
    # Report what resolved (audit).
    for d in dois:
        rec = records.get(sources.normalize_locator(d, "doi")) or {}
        print("  snapshot %-40s -> %s (src_text=%d)" % (
            d, rec.get("status"), len(rec.get("source_text") or "")))


def _write_manifest(fixtures: list[dict[str, Any]]) -> None:
    docs = []
    for fx in fixtures:
        docs.append({
            "group": "R", "doc_id": fx["case_id"],
            "label": "real_citation_" + fx["case_type"],
            "case_type": fx["case_type"],
            "source": "crossref_metadata",
            "cite_doi": fx["cite_doi"],
            "claim_source_doi": fx["claim_source_doi"],
            "path": os.path.join("poc", "claim_graph", "eval", "real_citation_cases",
                                 fx["case_id"] + ".json"),
            "words": fx["words"],
            "text_ref": {"kind": "case_json",
                         "path": os.path.join("poc", "claim_graph", "eval",
                                              "real_citation_cases", fx["case_id"] + ".json")},
        })
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "groups": {"R": {"label": "real_citation_slice",
                         "source": "crossref_metadata (has-abstract:true)",
                         "note": "entailed = claim lifted verbatim from cited paper's "
                                 "own abstract; misattributed = real DOI, off-source claim"}},
        "counts": {"R": len(docs),
                   CASE_ENTAILED: sum(1 for d in docs if d["case_type"] == CASE_ENTAILED),
                   CASE_MISATTRIBUTED: sum(1 for d in docs if d["case_type"] == CASE_MISATTRIBUTED)},
        "snapshot": os.path.join("poc", "claim_graph", "eval", "real_citation_snapshot.json"),
        "docs": docs,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def main() -> None:
    mailto = os.environ.get("DRAFTPROOF_CROSSREF_MAILTO", _DEFAULT_MAILTO)
    http_get = _default_get
    by_topic: dict[str, list[dict[str, Any]]] = {}
    calls = 0
    for t in TOPICS:
        cands = fetch_candidates(t, http_get, mailto, rows=4)
        calls += 1
        by_topic[t] = cands
        print("topic %-45s -> %d usable candidate(s) [crossref calls=%d]" % (
            t, len(cands), calls))
        time.sleep(1.0)  # polite pacing
    fixtures = build_slice(by_topic)
    if not fixtures:
        raise RuntimeError("no fixtures built — Crossref returned nothing usable")
    _write_fixtures(fixtures)
    _build_snapshot(fixtures)   # +len(unique DOIs) crossref calls
    _write_manifest(fixtures)
    n_ent = sum(1 for f in fixtures if f["case_type"] == CASE_ENTAILED)
    n_mis = sum(1 for f in fixtures if f["case_type"] == CASE_MISATTRIBUTED)
    print("BUILT %d fixtures (%d entailed, %d misattributed) -> %s" % (
        len(fixtures), n_ent, n_mis, CASES_DIR))
    print("snapshot -> %s" % SNAPSHOT_PATH)
    print("manifest -> %s" % MANIFEST_PATH)


if __name__ == "__main__":
    main()
