from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from poc.llm.gateway import LLMConfig, LLMGateway

from .plan import Plan, build_plan
from .scan import Scan, scan_text
from .write import Variant, choose_variant, write_variants


@dataclass(frozen=True)
class Result:
    scan: Scan
    plan: Plan
    variants: list[Variant]
    selected: Variant | None
    rewritten_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan": self.scan.to_dict(),
            "plan": self.plan.to_dict(),
            "variants": [asdict(variant) for variant in self.variants],
            "selected": asdict(self.selected) if self.selected else None,
            "rewritten_text": self.rewritten_text,
        }


@dataclass(frozen=True)
class DocumentResult:
    initial_scan: Scan
    final_scan: Scan
    passes: list[Result]
    rewritten_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_scan": self.initial_scan.to_dict(),
            "final_scan": self.final_scan.to_dict(),
            "passes": [result.to_dict() for result in self.passes],
            "rewritten_text": self.rewritten_text,
        }


def run_v6_rewrite(
    text: str,
    *,
    writer_client: Any | None = None,
    excluded_paragraph_ids: set[str] | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Result:
    scan = scan_text(text)
    paragraph, plan = build_plan(scan, excluded_paragraph_ids)
    client = writer_client or LLMGateway(
        LLMConfig(
            model=model or _writer_model(),
            api_key=api_key,
            base_url=base_url,
            max_tokens=900,
            temperature=0.12,
            top_p=0.75,
            extra_body=_writer_extra_body(model or _writer_model()),
        )
    )
    variants = write_variants(paragraph, plan, client=client)
    selected = choose_variant(variants, paragraph)
    return Result(scan=scan, plan=plan, variants=variants, selected=selected, rewritten_text=_compose(scan, paragraph.id, selected))


def run_v6_rewrite_all(
    text: str,
    *,
    writer_client: Any | None = None,
    max_passes: int | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> DocumentResult:
    initial_scan = scan_text(text)
    current = text
    passes: list[Result] = []
    limit = max_passes if max_passes is not None else max(1, len(initial_scan.paragraphs) * 3)
    attempts: dict[str, int] = {}
    exhausted: set[str] = set()
    for _ in range(limit):
        before = scan_text(current)
        if not before.findings:
            break
        if len(exhausted) >= len(before.paragraphs):
            break
        result = run_v6_rewrite(
            current,
            writer_client=writer_client,
            excluded_paragraph_ids=exhausted,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        if result.rewritten_text == current:
            exhausted.add(result.plan.paragraph_id)
            continue
        after = scan_text(result.rewritten_text)
        if not _improved(before, after):
            attempts[result.plan.paragraph_id] = attempts.get(result.plan.paragraph_id, 0) + 1
            if attempts[result.plan.paragraph_id] >= 2:
                exhausted.add(result.plan.paragraph_id)
            continue
        attempts[result.plan.paragraph_id] = 0
        exhausted.discard(result.plan.paragraph_id)
        passes.append(result)
        current = result.rewritten_text
    return DocumentResult(initial_scan=initial_scan, final_scan=scan_text(current), passes=passes, rewritten_text=current)


def _compose(scan: Scan, target_paragraph_id: str, selected: Variant | None) -> str:
    blocks = []
    for paragraph in scan.paragraphs:
        blocks.append(selected.text if selected and paragraph.id == target_paragraph_id else paragraph.text)
    return "\n\n".join(blocks)


def _improved(before: Scan, after: Scan) -> bool:
    return (
        after.scores["finding_count"] < before.scores["finding_count"]
        or after.scores["mean_sentence_shape_risk"] < before.scores["mean_sentence_shape_risk"]
    )


def _writer_model() -> str:
    import os

    return os.environ.get("DRAFTPROOF_V6_WRITER_MODEL") or os.environ.get("LLM_MODEL") or "qwen/qwen3-30b-a3b-instruct-2507"


def _writer_extra_body(model: str) -> dict[str, Any] | None:
    if "thinking" not in str(model or "").casefold():
        return None
    return {"reasoning": {"enabled": True, "exclude": True, "max_tokens": 64}, "include_reasoning": False}
