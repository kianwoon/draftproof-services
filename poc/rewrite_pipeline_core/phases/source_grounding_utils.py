"""Source-grounding query and Tavily result utilities."""

from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlparse


SOURCE_SEARCH_STOPWORDS = {
    "about", "after", "also", "and", "are", "because", "been", "being", "but",
    "can", "could", "does", "from", "have", "into", "more", "must", "not",
    "only", "should", "that", "the", "their", "then", "there", "these", "this",
    "through", "with", "within", "without", "would", "when", "where", "which",
    "while", "will", "were", "what", "who", "why", "how", "they", "them",
}
SOURCE_SEARCH_LOW_VALUE_OVERLAP_TERMS = {
    "evidence", "information", "report", "research", "study", "source",
    "sources", "claim", "claims", "analysis", "context",
}
SOURCE_SEARCH_CREDIBLE_TERMS = "evidence research study report"
SOURCE_SEARCH_DEFAULT_EXCLUDE_DOMAINS = {
    "instagram.com", "facebook.com", "tiktok.com", "x.com", "twitter.com",
    "pinterest.com", "reddit.com", "youtube.com", "getyourteachon.com",
}


def source_search_keywords(text: str, limit: int = 10) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z'-]{3,}|\b[A-Z]{2,}\d*\b|\b\d{4}\b", str(text or ""))
    scored: dict[str, int] = {}
    for raw in words:
        key = raw.strip("-'").lower()
        if not key or key in SOURCE_SEARCH_STOPWORDS:
            continue
        if len(key) < 4 and not re.match(r"^[a-z]{2,}\d", key):
            continue
        scored[key] = scored.get(key, 0) + (2 if raw[:1].isupper() or any(ch.isdigit() for ch in raw) else 1)
    ordered = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
    return [word for word, _score in ordered[:max(1, limit)]]


def source_grounding_query(claim: str) -> str:
    claim_text = re.sub(r"\s+", " ", str(claim or "").strip())
    lower = claim_text.lower()
    themes: list[str] = []

    def add_theme(*items: str) -> None:
        for item in items:
            if item and item not in themes:
                themes.append(item)

    if any(term in lower for term in ("youtube", "tiktok", "social media", "online course", "ai tool", "search engine")):
        add_theme("social media", "online platforms", "AI tools")
    if any(term in lower for term in ("trust", "accurate", "misleading", "sources", "information")):
        add_theme("information literacy", "evaluating online sources")
    if any(term in lower for term in ("feedback", "discussion", "reflection", "improvement", "process")):
        add_theme("feedback", "reflection", "process improvement")
    if any(term in lower for term in ("guide", "questions", "viewpoints", "judgment", "judgement")):
        add_theme("critical thinking", "judgement", "source evaluation")
    if any(term in lower for term in ("assessment", "understanding", "measurement", "evaluation")):
        add_theme("assessment", "understanding", "evaluation")

    keywords = source_search_keywords(claim_text, limit=8)
    for keyword in keywords:
        if len(themes) >= 8:
            break
        if keyword not in themes:
            themes.append(keyword)

    base = " ".join(themes[:8]).strip()
    if not base:
        base = " ".join(keywords[:6]).strip() or claim_text[:120].rstrip(" ,.;:")
    return f"{base} {SOURCE_SEARCH_CREDIBLE_TERMS}".strip()[:260]


def source_search_domain_list(name: str, default: set[str] | None = None) -> list[str]:
    raw = os.environ.get(name, "")
    domains = set(default or set())
    for item in re.split(r"[,\\s]+", raw):
        item = item.strip().lower()
        if item:
            domains.add(item)
    return sorted(domains)


def source_search_hostname(url: str) -> str:
    try:
        host = urlparse(str(url or "")).netloc.lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def source_search_domain_blocked(url: str, excluded_domains: set[str]) -> bool:
    host = source_search_hostname(url)
    if not host:
        return False
    return any(host == domain or host.endswith("." + domain) for domain in excluded_domains)


def source_search_quality_label(url: str) -> str:
    host = source_search_hostname(url)
    if not host:
        return "unknown"
    if host.endswith(".edu") or host.endswith(".gov"):
        return "high"
    if any(domain in host for domain in ("oecd.org", "unesco.org", "worldbank.org", "who.int", "eric.ed.gov")):
        return "high"
    if any(domain in host for domain in ("springer.com", "sciencedirect.com", "tandfonline.com", "emerald.com", "sagepub.com", "frontiersin.org")):
        return "medium_high"
    if any(domain in host for domain in ("instagram.com", "facebook.com", "tiktok.com", "youtube.com", "pinterest.com")):
        return "low"
    return "medium"


def normalize_tavily_results(
    payload: dict,
    claim: str = "",
    *,
    limit: int = 3,
    excluded_domains: set[str] | None = None,
) -> list[dict]:
    results = payload.get("results") if isinstance(payload, dict) else []
    if not isinstance(results, list):
        return []
    claim_terms = set(source_search_keywords(claim, limit=12))
    normalized = []
    excluded_domains = excluded_domains or set()
    for item in results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        if source_search_domain_blocked(url, excluded_domains):
            continue
        content = re.sub(r"\s+", " ", str(item.get("content") or "").strip())
        haystack = f"{title} {content}".lower()
        overlap = sorted(term for term in claim_terms if term in haystack)
        substantive_overlap = [
            term for term in overlap
            if term not in SOURCE_SEARCH_LOW_VALUE_OVERLAP_TERMS
        ]
        try:
            provider_score = float(item.get("score") or 0.0)
        except (TypeError, ValueError):
            provider_score = 0.0
        substantive_count = len(substantive_overlap)
        overlap_ratio = substantive_count / max(len(claim_terms), 1)
        if claim_terms and substantive_count == 0:
            relevance_score = min(provider_score * 0.10, 0.08)
        else:
            relevance_score = (
                provider_score * 0.45
                + min(1.0, overlap_ratio) * 0.45
                + min(substantive_count, 3) * 0.03
            )
        normalized.append({
            "title": title[:180],
            "url": url,
            "snippet": content[:360],
            "provider_score": round(provider_score, 4),
            "claim_keyword_overlap": overlap[:8],
            "substantive_claim_keyword_overlap": substantive_overlap[:8],
            "source_quality": source_search_quality_label(url),
            "relevance_score": round(min(1.0, relevance_score), 4),
        })
    normalized.sort(key=lambda item: item.get("relevance_score", 0), reverse=True)
    return normalized[:max(1, limit)]


def source_result_confidence(sources: list[dict]) -> str:
    if not sources:
        return "none"
    best = sources[0]
    quality = str(best.get("source_quality") or "unknown")
    score = float(best.get("relevance_score") or 0.0)
    overlap = best.get("claim_keyword_overlap")
    overlap_count = len(overlap) if isinstance(overlap, list) else 0
    substantive_overlap = best.get("substantive_claim_keyword_overlap")
    substantive_count = len(substantive_overlap) if isinstance(substantive_overlap, list) else overlap_count
    if substantive_count <= 0:
        return "very_weak"
    if quality == "high" and score >= 0.45:
        return "strong"
    if quality in {"high", "medium_high"} and score >= 0.35:
        return "moderate"
    if quality in {"high", "medium_high", "medium"} and score >= 0.35 and substantive_count >= 3:
        return "moderate"
    if score >= 0.85 and quality not in {"low", "unknown"}:
        return "moderate"
    if score >= 0.25:
        return "weak"
    return "very_weak"


@dataclass(frozen=True)
class SourceGroundingTargetDeps:
    logical_paragraphs: Callable[[str], list[str]]
    split_sentences: Callable[[str], list[str]]
    paragraph_component_targets: Callable[..., list[dict]]
    paragraph_role: Callable[..., str]
    safe_index: Callable[..., int]
    float_env: Callable[[str, float], float]
    paragraph_citation_re: re.Pattern


@dataclass(frozen=True)
class SourceGroundingPromptDeps:
    word_count_band: Callable[..., dict]
    float_env: Callable[[str, float], float]
    protected_anchor_brief_for_prompt: Callable[..., list[dict]]
    logical_paragraphs: Callable[[str], list[str]]
    safe_index: Callable[..., int]
    text_word_count: Callable[[str], int]


def source_grounding_claim_targets(
    text: str,
    report_dict: dict | None,
    *,
    limit: int = 5,
    deps: SourceGroundingTargetDeps,
) -> list[dict]:
    """Find claim spans where external evidence could reduce grounding risk."""
    paragraphs = deps.logical_paragraphs(text)
    if not paragraphs:
        return []
    badge = (report_dict or {}).get("ai_risk_badge") or {}
    writing = badge.get("writing_components") or {}
    ai_components = badge.get("ai_components") or {}
    component_targets = deps.paragraph_component_targets(text, report_dict or {}, limit=max(limit * 2, 4))
    ranked: list[dict] = []
    seen: set[int] = set()
    for target in component_targets:
        index = int(target.get("index", 0) or 0)
        if index in seen or index >= len(paragraphs):
            continue
        seen.add(index)
        paragraph = paragraphs[index]
        sentences = deps.split_sentences(paragraph) or [paragraph]
        candidate_sentence = max(
            sentences,
            key=lambda sentence: (
                len(re.findall(
                    r"\b(?:should|must|need(?:s)?|important|significant|supports?|helps?|"
                    r"allows?|enables?|creates?|means|shows?|suggests?|indicates?)\b",
                    sentence,
                    flags=re.I,
                )),
                min(len(sentence.split()), 38),
            ),
        ).strip()
        if len(candidate_sentence.split()) < 8:
            candidate_sentence = paragraph[:260].strip()
        role = target.get("role") or deps.paragraph_role(paragraph, target.get("drivers") or {})
        if role == "human_anchor_rich":
            continue
        ranked.append({
            "id": f"source_claim_{len(ranked) + 1}",
            "paragraph_index": index,
            "paragraph_role": role,
            "claim": candidate_sentence[:420],
            "query": source_grounding_query(candidate_sentence),
            "target_preview": paragraph[:360],
            "why_needed": (
                "External source support can reduce grounding quality risk for this claim. "
                "It must not be treated as author-owned lived evidence."
            ),
            "scanner_context": {
                "source_grounding_risk": writing.get("source_grounding_risk"),
                "unsupported_claim_risk": writing.get("unsupported_claim_risk"),
                "generic_assertion_risk": ai_components.get("generic_assertion_risk"),
            },
        })
        if len(ranked) >= max(1, limit):
            break
    if ranked:
        return ranked
    fallback = []
    for index, paragraph in enumerate(paragraphs[:max(limit * 2, 3)]):
        if len(paragraph.split()) < 35 or deps.paragraph_citation_re.search(paragraph):
            continue
        sentence = (deps.split_sentences(paragraph) or [paragraph])[0].strip()
        fallback.append({
            "id": f"source_claim_{len(fallback) + 1}",
            "paragraph_index": index,
            "paragraph_role": deps.paragraph_role(paragraph, {"source_gap": True}),
            "claim": sentence[:420],
            "query": source_grounding_query(sentence),
            "target_preview": paragraph[:360],
            "why_needed": "This uncited claim may need a public source or a narrower wording.",
            "scanner_context": {
                "source_grounding_risk": writing.get("source_grounding_risk"),
                "unsupported_claim_risk": writing.get("unsupported_claim_risk"),
                "generic_assertion_risk": ai_components.get("generic_assertion_risk"),
            },
        })
        if len(fallback) >= max(1, limit):
            break
    return fallback


def source_grounding_targets_from_block_decisions(
    text: str,
    report_dict: dict | None,
    block_decisions: list[dict],
    *,
    limit: int = 5,
    deps: SourceGroundingTargetDeps,
) -> list[dict]:
    """Convert reinforce decisions into Tavily search targets."""
    paragraphs = deps.logical_paragraphs(text)
    if not paragraphs:
        return []
    badge = (report_dict or {}).get("ai_risk_badge") or {}
    writing = badge.get("writing_components") or {}
    ai_components = badge.get("ai_components") or {}
    targets: list[dict] = []
    seen_indexes: set[int] = set()
    for decision in block_decisions or []:
        if not isinstance(decision, dict):
            continue
        if decision.get("decision") != "reinforce_with_public_source":
            continue
        index = deps.safe_index(decision.get("paragraph_index"), -1)
        if index < 0 or index >= len(paragraphs) or index in seen_indexes:
            continue
        seen_indexes.add(index)
        paragraph = paragraphs[index]
        sentences = deps.split_sentences(paragraph) or [paragraph]
        candidate_sentence = max(
            sentences,
            key=lambda sentence: (
                len(re.findall(
                    r"\b(?:should|must|need(?:s)?|important|significant|supports?|helps?|"
                    r"allows?|enables?|creates?|means|shows?|suggests?|indicates?|changes?|"
                    r"affects?|guides?|compare|trust|source|evidence)\b",
                    sentence,
                    flags=re.I,
                )),
                min(len(sentence.split()), 42),
            ),
        ).strip()
        if len(candidate_sentence.split()) < 8:
            candidate_sentence = paragraph[:260].strip()
        targets.append({
            "id": f"source_claim_{len(targets) + 1}",
            "paragraph_index": index,
            "paragraph_role": decision.get("role") or deps.paragraph_role(paragraph, {"source_gap": True}),
            "claim": candidate_sentence[:420],
            "query": source_grounding_query(candidate_sentence),
            "target_preview": paragraph[:360],
            "why_needed": (
                "The block planner marked this claim as salvageable through public-source reinforcement. "
                "If support is weak, the fallback is narrow/remove rather than inventing evidence."
            ),
            "block_decision": {
                key: decision.get(key)
                for key in (
                    "decision",
                    "reason",
                    "fallback_if_failed",
                    "source_search_slot",
                    "allowed_operations",
                )
            },
            "scanner_context": {
                "source_grounding_risk": writing.get("source_grounding_risk"),
                "unsupported_claim_risk": writing.get("unsupported_claim_risk"),
                "generic_assertion_risk": ai_components.get("generic_assertion_risk"),
            },
        })
        if len(targets) >= max(1, limit):
            break
    return targets


def citation_reference_search_targets(
    text: str,
    report_dict: dict | None,
    *,
    limit: int = 3,
    deps: SourceGroundingTargetDeps,
) -> list[dict]:
    """Create search targets from citation markers already present in the draft."""
    if not text or limit <= 0:
        return []
    writing = ((report_dict or {}).get("ai_risk_badge") or {}).get("writing_components") or {}
    citation_risk = float(writing.get("citation_weakness_risk") or 0.0)
    source_risk = float(writing.get("source_grounding_risk") or 0.0)
    if (
        citation_risk < deps.float_env("DRAFTPROOF_CITATION_REFERENCE_SEARCH_MIN_CITATION_RISK", 45.0)
        and source_risk < deps.float_env("DRAFTPROOF_CITATION_REFERENCE_SEARCH_MIN_SOURCE_RISK", 40.0)
    ):
        return []

    paragraph_sentence_rows: list[dict] = []
    for paragraph_index, paragraph in enumerate(deps.logical_paragraphs(text)):
        for local_sentence_index, sentence in enumerate(deps.split_sentences(paragraph)):
            paragraph_sentence_rows.append({
                "sentence_global_index": len(paragraph_sentence_rows),
                "paragraph_index": paragraph_index,
                "sentence_index": local_sentence_index,
                "sentence": sentence,
                "paragraph_context": paragraph,
            })
    if not paragraph_sentence_rows:
        return []
    parenthetical_re = re.compile(r"\(([^()]*?(?:19|20)\d{2}[a-z]?[^()]*)\)")
    narrative_re = re.compile(
        r"\b([A-Z][A-Za-z'’.-]+(?:\s+(?:and|&|et\s+al\.?)\s+[A-Z][A-Za-z'’.-]+)*)\s*"
        r"\(((?:19|20)\d{2}[a-z]?)\)"
    )
    targets: list[dict] = []
    seen: set[str] = set()

    def add_target(label: str, row: dict) -> None:
        sentence = str(row.get("sentence") or "")
        label = re.sub(r"\s+", " ", label or "").strip(" ;,")
        if not label or not re.search(r"(?:19|20)\d{2}", label):
            return
        key = label.lower()
        if key in seen:
            return
        seen.add(key)
        keywords = source_search_keywords(sentence, limit=6)
        query = f'"{label}" ' + " ".join(keywords)
        targets.append({
            "id": f"citation_ref_{len(targets) + 1}",
            "paragraph_index": row.get("paragraph_index"),
            "sentence_index": row.get("sentence_index"),
            "sentence_global_index": row.get("sentence_global_index"),
            "repair_scope": "sentence_window",
            "paragraph_role": "citation_reference",
            "claim": sentence.strip()[:420],
            "query": query.strip(),
            "target_preview": sentence.strip()[:360],
            "paragraph_context": str(row.get("paragraph_context") or "").strip()[:900],
            "why_needed": (
                "The draft already contains this citation marker; search should complete or verify "
                "the source instead of inventing a new grounding claim."
            ),
            "citation_label": label,
            "scanner_context": {
                "citation_weakness_risk": citation_risk,
                "source_grounding_risk": source_risk,
            },
        })

    for row in paragraph_sentence_rows:
        sentence = str(row.get("sentence") or "")
        for match in narrative_re.finditer(sentence):
            add_target(f"{match.group(1)} {match.group(2)}", row)
            if len(targets) >= limit:
                return targets
        for match in parenthetical_re.finditer(sentence):
            for part in re.split(r";", match.group(1)):
                add_target(part, row)
                if len(targets) >= limit:
                    return targets
    return targets[:max(1, int(limit or 1))]


def source_grounding_repair_prompt(
    target: dict,
    source_result: dict,
    *,
    candidate_count: int = 2,
) -> str:
    candidate_count = max(1, int(candidate_count or 1))
    repair_scope = str(target.get("repair_scope") or "paragraph")
    scoped_to_sentence = repair_scope == "sentence_window"
    safe_sources = []
    for source in (source_result.get("sources") or [])[:3]:
        if not isinstance(source, dict):
            continue
        safe_sources.append({
            "title": source.get("title"),
            "url": source.get("url"),
            "snippet": source.get("snippet"),
            "source_quality": source.get("source_quality"),
            "relevance_score": source.get("relevance_score"),
            "claim_keyword_overlap": source.get("claim_keyword_overlap"),
        })
    scope_intro = (
        "Repair one citation sentence by reinforcing a weak public/source-groundable claim. "
        if scoped_to_sentence else
        "Repair one paragraph by reinforcing a weak public/source-groundable claim. "
    )
    target_block = (
        "PARAGRAPH CONTEXT:\n"
        f"<PARAGRAPH_CONTEXT>\n{target.get('paragraph_context') or ''}\n</PARAGRAPH_CONTEXT>\n\n"
        "TARGET SENTENCE:\n"
        f"<TARGET_SENTENCE>\n{target.get('target_preview') or ''}\n</TARGET_SENTENCE>\n\n"
        if scoped_to_sentence else
        "TARGET PARAGRAPH:\n"
        f"<TARGET_PARAGRAPH>\n{target.get('target_preview') or ''}\n</TARGET_PARAGRAPH>\n\n"
    )
    output_shape = (
        "<CANDIDATE_1>\nreplacement sentence or two-sentence window only\n</CANDIDATE_1>\n"
        "<CANDIDATE_2>\nreplacement sentence or two-sentence window only\n</CANDIDATE_2>\n"
        if scoped_to_sentence else
        "<CANDIDATE_1>\nreplacement paragraph only\n</CANDIDATE_1>\n"
        "<CANDIDATE_2>\nreplacement paragraph only\n</CANDIDATE_2>\n"
    )
    return (
        "DraftProof SOURCE_GROUNDING_REPAIR.\n"
        f"{scope_intro}"
        "If the sources do not clearly support the claim, narrow the claim instead of forcing the source.\n\n"
        "Operating rule:\n"
        "- First try to reinforce the paragraph using only the provided source candidates.\n"
        "- Do not invent evidence. Do not invent author-owned context.\n"
        "- If support is weak, remove or narrow the unsupported part inside the paragraph.\n\n"
        "Source-attribution rule:\n"
        "- If you use source support, name the source title or a short source label in the sentence.\n"
        "- Do not write 'Research shows', 'studies show', or 'evidence shows' unless the same sentence also names the source.\n"
        "- If naming the source would feel awkward, narrow the claim instead of adding source-attribution language.\n\n"
        f"Paragraph index: {target.get('paragraph_index')}.\n"
        f"Sentence index: {target.get('sentence_index')}.\n"
        f"Repair scope: {repair_scope}.\n"
        f"Paragraph role: {target.get('paragraph_role')}.\n"
        f"Source confidence: {source_result.get('source_confidence')}.\n"
        f"Claim to repair: {target.get('claim')}\n\n"
        "Source candidates, use only these:\n"
        f"{json.dumps(safe_sources, ensure_ascii=False)[:2600]}\n\n"
        "Allowed:\n"
        "- add one source-to-claim bridge if the source snippet clearly supports it\n"
        "- narrow an unsupported claim to what the source actually supports\n"
        "- remove a broad unsupported sentence when it cannot be sourced\n"
        "- keep the paragraph less polished; avoid generic academic connectors\n\n"
        "Forbidden:\n"
        "- do not create lived experience, local observation, or author-owned evidence\n"
        "- do not create statistics, dates, names, citations, or claims not present in the source candidates\n"
        "- do not add a citation if only the title/url is available and the source support is unclear\n"
        "- do not rewrite the paragraph into a smoother generic explanation\n"
        "- do not change protected names, numbers, years, unit codes, or existing citations\n\n"
        "Gate that will rescan your output:\n"
        "- Human Contribution must increase or stay safe\n"
        "- Grounding Risk should drop\n"
        "- AI Authorship must not increase\n"
        "- AI Transformation, findings, review burden, and weighted severity must not increase\n\n"
        f"{target_block}"
        f"Return exactly {candidate_count} alternatives using this format:\n"
        f"{output_shape}"
        "...continue until the requested candidate count.\n"
        "No commentary outside tags."
    )


def internet_reinforced_reauthor_prompt(
    source_text: str,
    source_layer: dict,
    *,
    candidate_count: int = 2,
    deps: SourceGroundingPromptDeps,
) -> str:
    word_band = deps.word_count_band(
        source_text,
        variance=deps.float_env("DRAFTPROOF_INTERNET_REAUTHOR_WORD_VARIANCE", 0.25),
    )
    protected_anchors = deps.protected_anchor_brief_for_prompt(source_text)
    paragraphs = deps.logical_paragraphs(source_text)
    evidence_cards = []
    targets_by_id = {
        target.get("id"): target
        for target in (source_layer.get("claim_targets") or [])
        if isinstance(target, dict)
    }
    usable_conf = {
        item.strip()
        for item in os.environ.get(
            "DRAFTPROOF_INTERNET_REAUTHOR_CONFIDENCE",
            "strong,moderate,weak",
        ).split(",")
        if item.strip()
    }
    for result in (source_layer.get("results") or []):
        if not isinstance(result, dict):
            continue
        confidence = str(result.get("source_confidence") or "")
        if confidence not in usable_conf:
            continue
        target = targets_by_id.get(result.get("claim_id")) or {}
        sources = []
        for source in (result.get("sources") or [])[:3]:
            if not isinstance(source, dict):
                continue
            sources.append({
                "title": source.get("title"),
                "url": source.get("url"),
                "snippet": source.get("snippet"),
                "quality": source.get("source_quality"),
                "relevance": source.get("relevance_score"),
            })
        if not sources:
            continue
        evidence_cards.append({
            "claim_id": result.get("claim_id"),
            "paragraph_index": target.get("paragraph_index"),
            "source_confidence": confidence,
            "original_claim": target.get("claim"),
            "paragraph_role": target.get("paragraph_role"),
            "sources": sources,
        })
    block_inventory = []
    for index, paragraph in enumerate(paragraphs):
        target_matches = [
            target for target in (source_layer.get("claim_targets") or [])
            if deps.safe_index(target.get("paragraph_index"), -1) == index
        ]
        block_inventory.append({
            "paragraph_index": index,
            "word_count": deps.text_word_count(paragraph),
            "preview": paragraph[:420],
            "source_claim_targets": [
                {
                    "claim_id": target.get("id"),
                    "claim": target.get("claim"),
                    "role": target.get("paragraph_role"),
                }
                for target in target_matches
            ],
        })
    return (
        "DraftProof INTERNET_REINFORCED_REAUTHORING.\n"
        "Generate a new document from the claim inventory and internet evidence cards. "
        "This is not sentence repair. Rebuild the document around supported claims and remove unsupported generic drag.\n\n"
        "Hard operating rule:\n"
        "- If a block can be reinforced by the evidence cards, rewrite that block around the supported claim.\n"
        "- If a block cannot be reinforced and is generic/repetitive, remove or compress it.\n"
        "- Preserve the submitted meaning coverage, but do not preserve weak wording or weak paragraph order.\n\n"
        "Protected anchors. These exact spans must appear unchanged in every candidate, including quote text:\n"
        f"{json.dumps(protected_anchors, ensure_ascii=False)[:2600]}\n\n"
        "Goal:\n"
        "Maximize Human Contribution under the AI Authorship cap. The next scan will reject candidates that raise AI Authorship, drift, findings, review burden, or severity.\n\n"
        f"Word-count band: source={word_band['source_word_count']}, min={word_band['min_words']}, max={word_band['max_words']}.\n\n"
        "Evidence cards from internet search. Use only these sources; do not invent facts beyond their snippets:\n"
        f"{json.dumps(evidence_cards, ensure_ascii=False)[:9000]}\n\n"
        "Document block inventory. Use this for meaning coverage, not wording imitation:\n"
        f"{json.dumps(block_inventory, ensure_ascii=False)[:7000]}\n\n"
        "Required output behavior:\n"
        "- Recreate the document as a fresh draft, not a paragraph-by-paragraph paraphrase.\n"
        "- Use source-supported claims where the evidence cards clearly support them.\n"
        "- Remove generic filler that does not add source support, author reasoning, or required coverage.\n"
        "- Keep the prose direct and uneven enough to avoid polished academic template flow.\n"
        "- Use fewer broad claims. Prefer bounded claims tied to the sources or the submitted document context.\n"
        "- Do not create author-owned observations, lived experience, local events, personal examples, new institutions, new dates, new statistics, or fake citations.\n"
        "- Do not drop, paraphrase, normalize, or reword protected anchors. If a quote is awkward, keep the quote exactly and rewrite around it.\n"
        "- Do not add markdown links or bibliography. Mention source titles only when useful and supported by the card.\n"
        "- Do not use generic connectors such as Furthermore, Moreover, Additionally, This highlights, This underscores, In conclusion.\n\n"
        "Selection gate:\n"
        "- Human Contribution should rise substantially.\n"
        "- AI Authorship must not rise.\n"
        "- AI Transformation should fall.\n"
        "- Grounding risk should fall.\n"
        "- Findings, review burden, and severity must not regress.\n\n"
        f"Return exactly {max(1, int(candidate_count or 1))} complete document candidates using this exact format:\n"
        "<CANDIDATE_1>\ncomplete document only\n</CANDIDATE_1>\n"
        "<CANDIDATE_2>\ncomplete document only\n</CANDIDATE_2>\n"
        "...continue until the requested candidate count.\n"
        "No commentary outside tags."
    )
