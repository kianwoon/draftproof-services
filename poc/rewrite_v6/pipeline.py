from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from poc.llm.gateway import LLMConfig, LLMGateway

from .plan import Plan, build_plan
from .planner_llm import run_planner_llm
from .scan import Scan, findings_for_paragraph, scan_text
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
    planner_client: Any | None = None,
    writer_client: Any | None = None,
    excluded_paragraph_ids: set[str] | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    progress_percent: int | None = None,
) -> Result:
    scan = scan_text(text)
    paragraph, plan = build_plan(scan, excluded_paragraph_ids)
    _emit_progress(progress_callback, progress_percent, f"Planning V6 paragraph {paragraph.id}")
    if planner_client is not None or writer_client is None:
        plan = run_planner_llm(
            paragraph,
            plan,
            findings_for_paragraph(scan, paragraph.id),
            client=planner_client or _planner_gateway(api_key=api_key, base_url=base_url),
        )
    _emit_progress(progress_callback, progress_percent, f"Writing V6 paragraph {paragraph.id}")
    client = writer_client or LLMGateway(
        LLMConfig(
            model=model or _writer_model(),
            api_key=api_key,
            base_url=base_url,
            max_tokens=None,
            temperature=0.12,
            top_p=0.75,
            extra_body=_writer_extra_body(model or _writer_model()),
        )
    )
    variants = write_variants(paragraph, plan, client=client)
    _emit_progress(progress_callback, progress_percent, f"Scanning V6 paragraph {paragraph.id} candidate")
    selected = choose_variant(variants, paragraph)
    return Result(scan=scan, plan=plan, variants=variants, selected=selected, rewritten_text=_compose(scan, paragraph.id, selected))


def run_v6_rewrite_all(
    text: str,
    *,
    planner_client: Any | None = None,
    writer_client: Any | None = None,
    max_passes: int | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> DocumentResult:
    initial_scan = scan_text(text)
    current = text
    passes: list[Result] = []
    limit = max_passes if max_passes is not None else max(1, len(initial_scan.paragraphs) * 3)
    attempts: dict[str, int] = {}
    exhausted: set[str] = set()
    for pass_index in range(limit):
        before = scan_text(current)
        if not before.findings:
            break
        if len(exhausted) >= len(before.paragraphs):
            break
        start_percent = _rewrite_progress_percent(pass_index, limit)
        _emit_progress(
            progress_callback,
            start_percent,
            f"V6 rewrite pass {pass_index + 1}: {len(before.findings)} finding(s) remaining",
        )
        result = run_v6_rewrite(
            current,
            planner_client=planner_client,
            writer_client=writer_client,
            excluded_paragraph_ids=exhausted,
            model=model,
            api_key=api_key,
            base_url=base_url,
            progress_callback=progress_callback,
            progress_percent=start_percent,
        )
        end_percent = _rewrite_progress_percent(pass_index + 1, limit)
        if result.rewritten_text == current:
            exhausted.add(result.plan.paragraph_id)
            _emit_progress(progress_callback, end_percent, f"V6 paragraph {result.plan.paragraph_id} made no change")
            continue
        after = scan_text(result.rewritten_text)
        if not _improved(before, after):
            attempts[result.plan.paragraph_id] = attempts.get(result.plan.paragraph_id, 0) + 1
            if attempts[result.plan.paragraph_id] >= 2:
                exhausted.add(result.plan.paragraph_id)
            _emit_progress(progress_callback, end_percent, f"V6 paragraph {result.plan.paragraph_id} did not improve")
            continue
        attempts[result.plan.paragraph_id] = 0
        exhausted.discard(result.plan.paragraph_id)
        passes.append(result)
        current = result.rewritten_text
        _emit_progress(progress_callback, end_percent, f"Accepted V6 paragraph {result.plan.paragraph_id}")
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


def _rewrite_progress_percent(step: int, limit: int) -> int:
    span_start = 63
    span_end = 87
    bounded_limit = max(1, int(limit or 1))
    bounded_step = max(0, min(bounded_limit, int(step or 0)))
    return span_start + int(round((span_end - span_start) * (bounded_step / bounded_limit)))


def _emit_progress(callback: Callable[[int, str], None] | None, percent: int | None, message: str) -> None:
    if callback is not None and percent is not None:
        callback(percent, message)


def _writer_model() -> str:
    import os

    return os.environ.get("DRAFTPROOF_V6_WRITER_MODEL") or os.environ.get("LLM_MODEL") or "qwen/qwen3-30b-a3b-instruct-2507"


def _planner_model() -> str:
    import os

    return (
        os.environ.get("DRAFTPROOF_V6_PLANNER_MODEL")
        or os.environ.get("DRAFTPROOF_PLANNER_MODEL")
        or os.environ.get("DRAFTPROOF_REWRITE_V5_PLANNER_MODEL")
        or "z-ai/glm-5.1"
    )


def _planner_gateway(*, api_key: str | None, base_url: str | None) -> LLMGateway:
    model = _planner_model()
    return LLMGateway(
        LLMConfig(
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=None,
            temperature=0.1,
            top_p=0.75,
            extra_body=_planner_extra_body(model),
        )
    )


def _planner_extra_body(model: str) -> dict[str, Any] | None:
    if "thinking" in str(model or "").casefold():
        return {"reasoning": {"enabled": True, "exclude": True, "max_tokens": 128}, "include_reasoning": False}
    return {"reasoning": {"enabled": False}, "include_reasoning": False}


def _writer_extra_body(model: str) -> dict[str, Any] | None:
    if "thinking" not in str(model or "").casefold():
        return None
    return {"reasoning": {"enabled": True, "exclude": True, "max_tokens": 64}, "include_reasoning": False}
