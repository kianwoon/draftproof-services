from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .compiler import compile_plain_text
from .json_io import parse_json
from .plan import Plan
from .scan import scan_text
from .text import Paragraph, source_terms, split_paragraphs, word_count


class ChatClient(Protocol):
    def chat(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> Any:
        ...


@dataclass(frozen=True)
class Variant:
    id: str
    text: str
    source: str
    author_proxy_provenance: list[dict[str, Any]] | None = None
    author_review_items: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_variants(paragraph: Paragraph, plan: Plan, *, client: ChatClient) -> list[Variant]:
    compiled = compile_plain_variant(paragraph, plan)
    variants = [compiled]
    if not _needs_live_writer(compiled, plan):
        return variants
    response = client.chat(
        build_prompt(paragraph, plan),
        system="Return valid JSON only. Preserve submitted facts as written; mark inferred bridge material for author review.",
        temperature=0.12,
        top_p=0.75,
        max_tokens=900,
        response_format={"type": "json_object"},
    )
    try:
        variants.extend(parse_variants(parse_json(getattr(response, "raw_content", "") or response.content)))
    except ValueError:
        return variants
    return variants


def build_prompt(paragraph: Paragraph, plan: Plan) -> str:
    payload = {
        "task": "paragraph_level_route_rewrite",
        "source_paragraph": paragraph.text,
        "source_sentences": [{"id": sentence.id, "text": sentence.text} for sentence in paragraph.sentences],
        "route_plan": plan.to_dict(),
        "source_terms": source_terms(paragraph.text, limit=24),
        "finding_methods": [
            {
                "sentence_id": action.sentence_id,
                "tags": action.tags,
                "method": action.method,
                "operation": action.operation,
                "preserve_terms": action.preserve_terms,
            }
            for action in plan.actions
        ],
        "execution_method": [
            "Keep all source ideas, but rebuild the paragraph route instead of polishing sentence wording.",
            "Decompress packed lists by blending listed nouns or actions into varied ordinary sentences; do not split every item into repeated-subject sentences.",
            "When repairing a list, group related items into two or three natural prose beats instead of keeping one long comma list.",
            "Use at most one short pair joined by 'and' in a sentence; avoid three-or-more-item comma lists.",
            "Break predictable starts with source-derived terms, not fixed domain wording.",
            "Keep the paragraph at least as detailed as the source; do not summarize.",
            "Do not start more than two sentences with the same first three words.",
            "Most sentences should be complete prose sentences between 8 and 24 words.",
            "Return grammatical prose, not atomic bullet-like sentences joined with periods.",
            "If bridge material is not directly supported by the submitted text, keep rewriting and mark it with author-review provenance instead of presenting it as source-confirmed.",
            "For author_proxy_bridge, context_anchor_bridge, or semantic_bridge_repair methods, include reviewable bridge wording when needed and record its provenance.",
        ],
        "list_repair_boundary": {
            "bad_under_repair": "one long comma list OR one repeated sentence per item",
            "good_under_repair": "two or three varied prose beats that preserve the source ideas without itemizing every noun or action",
            "self_check": "If three consecutive sentences start the same way, rewrite again before returning JSON.",
        },
        "method_contract": {
            "atomic_decomposition": "split packed source meaning into atomic sentences without dropping source ideas",
            "route_rebuild": "change opener, clause order, or sentence boundary",
            "claim_scope_repair": "narrow broad claims or mark the missing bridge for author review",
            "citation_relation_repair": "keep cited/source relation close to the claim; any new citation-like detail must be marked for author review instead of treated as source-confirmed",
            "context_anchor_bridge": "replace vague this/that/it routes with a source-grounded context bridge",
            "author_proxy_bridge": "draft a bridge as author-proxy using submitted context; label provenance as inferred_from_draft or needs_author_confirmation",
            "semantic_bridge_repair": "make the missing reasoning link explicit using submitted context or mark the bridge for author review",
            "transition_rebuild": "remove stacked transition wording and reconnect the sentence through source terms",
            "source_revoice": "replace smooth paraphrase texture with source-level wording, uneven sentence pressure, and provenance for any inferred bridge",
        },
        "author_proxy_policy": {
            "enabled": True,
            "non_interrupting": True,
            "allowed_provenance": ["source_preserved", "inferred_from_draft", "needs_author_confirmation", "must_replace"],
            "review_required": True,
            "responsibility_boundary": "DraftProof drafts and labels provenance; the user must review and owns facts, citations, anchors, and author-proxy bridges before submission.",
            "rule": "Do not stop for questions. When a needed anchor is not directly supported by submitted text, continue the rewrite and include it in author_review_items.",
        },
        "bad_patterns_to_avoid": [
            "same first sentence route with synonyms",
            "single smooth summary",
            "neat three-item list in one sentence",
            "repeated sentence frame such as the same subject plus verb pattern three or more times",
            "mechanical one-item-per-sentence decomposition",
            "compressed timeline conclusion",
            "new external fact or citation presented as source-confirmed without author-review provenance",
        ],
        "required_shape": {
            "paragraph_count": 1,
            "sentence_count": [max(len(paragraph.sentences), 6), max(len(paragraph.sentences) + 4, 8)],
            "word_count": [max(1, int(word_count(paragraph.text) * 0.95)), int(word_count(paragraph.text) * 1.3)],
        },
        "output_schema": {
            "variants": [
                {
                    "id": "v1",
                    "text": "replacement paragraph only",
                    "author_proxy_provenance": [],
                    "author_review_items": [],
                }
            ]
        },
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def compile_plain_variant(paragraph: Paragraph, plan: Plan) -> Variant:
    return Variant(id="compiled_plain", text=compile_plain_text(paragraph, plan), source="compiler")


def _needs_live_writer(compiled: Variant, plan: Plan) -> bool:
    if scan_text(compiled.text).findings:
        return True
    live_methods = {"author_proxy_bridge", "context_anchor_bridge", "semantic_bridge_repair"}
    return any(action.method in live_methods for action in plan.actions)


def parse_variants(payload: Any) -> list[Variant]:
    rows = payload.get("variants") if isinstance(payload, dict) else payload if isinstance(payload, list) else []
    variants: list[Variant] = []
    for index, row in enumerate(rows if isinstance(rows, list) else [], start=1):
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").strip()
        if text:
            variants.append(
                Variant(
                    id=str(row.get("id") or row.get("variant_id") or f"v{index}"),
                    text=text,
                    source="llm",
                    author_proxy_provenance=_list_of_dicts(row.get("author_proxy_provenance")),
                    author_review_items=_list_of_dicts(row.get("author_review_items")),
                )
            )
    return variants


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def choose_variant(variants: list[Variant], paragraph: Paragraph) -> Variant | None:
    if not variants:
        return None
    source_words = max(1, word_count(paragraph.text))
    return max(
        variants,
        key=lambda variant: _variant_rank(variant, paragraph, source_words),
    )


def _variant_rank(variant: Variant, paragraph: Paragraph, source_words: int) -> tuple[float, float, float, bool, int]:
    scan = scan_text(variant.text)
    words = word_count(variant.text)
    quality_penalty = _mechanical_quality_penalty(variant.text, paragraph)
    return (
        -(scan.scores["finding_count"] + quality_penalty),
        -scan.scores["mean_sentence_shape_risk"],
        _coverage(variant.text, paragraph),
        words >= source_words * 0.9,
        -abs(words - source_words),
    )


def _mechanical_quality_penalty(text: str, source_paragraph: Paragraph) -> float:
    paragraphs = split_paragraphs(text)
    sentences = [sentence for paragraph in paragraphs for sentence in paragraph.sentences]
    if not sentences:
        return 8.0
    source_sentence_count = max(1, len(source_paragraph.sentences))
    sentence_count = len(sentences)
    avg_words = sum(sentence.word_count for sentence in sentences) / max(1, sentence_count)
    short_ratio = sum(1 for sentence in sentences if sentence.word_count <= 7) / max(1, sentence_count)
    first_words: dict[str, int] = {}
    first_frames: dict[str, int] = {}
    for sentence in sentences:
        parts = [
            part.strip(".,:;!?\"'“”’").casefold()
            for part in sentence.text.split()
            if part.strip(".,:;!?\"'“”’")
        ]
        first = parts[0] if parts else ""
        if first:
            first_words[first] = first_words.get(first, 0) + 1
        if len(parts) >= 3:
            frame = " ".join(parts[:3])
            first_frames[frame] = first_frames.get(frame, 0) + 1
    repeated_first_ratio = max(first_words.values(), default=0) / max(1, sentence_count)
    repeated_frame_ratio = max(first_frames.values(), default=0) / max(1, sentence_count)
    penalty = 0.0
    if sentence_count > source_sentence_count * 1.8:
        penalty += min(4.0, (sentence_count / source_sentence_count) - 1.0)
    if short_ratio >= 0.45 and avg_words <= 9.0:
        penalty += short_ratio * 4.0
    if repeated_first_ratio >= 0.35:
        penalty += repeated_first_ratio * 3.0
    if repeated_frame_ratio >= 0.30:
        penalty += repeated_frame_ratio * 4.0
    return round(penalty, 3)

def _coverage(text: str, paragraph: Paragraph) -> float:
    anchors = source_terms(paragraph.text, limit=24)
    lowered = str(text or "").casefold()
    return sum(1 for anchor in anchors if anchor.casefold() in lowered) / len(anchors) if anchors else 1.0
