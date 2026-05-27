from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from poc.llm.gateway import LLMConfig, LLMGateway

DEFAULT_GRAMMER_MODEL = "openai/gpt-oss-120b"
from .json_io import parse_json


@dataclass(frozen=True)
class QualityRepairOperation:
    find: str
    replace: str
    reason: str


@dataclass(frozen=True)
class QualityRepairResult:
    original_text: str
    repaired_text: str
    operations: list[QualityRepairOperation] = field(default_factory=list)
    skipped_operations: list[dict[str, Any]] = field(default_factory=list)
    status: str = "not_run"
    model: str | None = None

    @property
    def changed(self) -> bool:
        return _same_text(self.original_text, self.repaired_text) is False

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "changed": self.changed,
            "model": self.model,
            "operations": [asdict(operation) for operation in self.operations],
            "skipped_operations": list(self.skipped_operations),
        }


def run_quality_repair(
    text: str,
    *,
    client: Any,
    model: str | None = None,
) -> QualityRepairResult:
    response = client.chat(
        _quality_repair_prompt(text),
        system=(
            "You are a copy editor. Return valid JSON only. "
            "Do not rewrite for style, sophistication, detector avoidance, or AI-risk reduction."
        ),
        temperature=0.0,
        top_p=0.2,
        max_tokens=None,
        response_format={"type": "json_object"},
        app_label="Grammer",
    )
    payload = parse_json(getattr(response, "raw_content", "") or response.content)
    operations = _parse_operations(payload)
    repaired, applied, skipped = apply_quality_repair_operations(text, operations)
    return QualityRepairResult(
        original_text=text,
        repaired_text=repaired,
        operations=applied,
        skipped_operations=skipped,
        status="applied" if applied else "no_changes",
        model=getattr(client, "model", model),
    )


def run_quality_repair_once(
    current: str,
    *,
    original_text: str,
    quality_client: Any | None,
    api_key: str | None,
    base_url: str | None,
    cancellation_check: Callable[[], None] | None,
    progress_callback: Callable[[int, str], None] | None,
) -> QualityRepairResult | None:
    if _same_text(current, original_text):
        return None
    if not _quality_repair_enabled():
        return None
    if quality_client is None and not _has_llm_api_key(api_key):
        return QualityRepairResult(
            original_text=current,
            repaired_text=current,
            status="skipped_missing_llm_api_key",
        )
    if cancellation_check is not None:
        cancellation_check()
    if progress_callback is not None:
        progress_callback(88, "Running V6 Grammer light-touch repair")
    try:
        return run_quality_repair(
            current,
            client=quality_client or _grammer_gateway(
                api_key=api_key,
                base_url=base_url,
                cancellation_check=cancellation_check,
            ),
            model=_grammer_model(),
        )
    except Exception as exc:
        return QualityRepairResult(
            original_text=current,
            repaired_text=current,
            status=f"failed:{type(exc).__name__}",
        )


def _grammer_model() -> str:
    import os

    return (
        os.environ.get("DRAFTPROOF_V6_GRAMMER_MODEL")
        or os.environ.get("DRAFTPROOF_V6_WRITER_MODEL")
        or os.environ.get("LLM_MODEL")
        or DEFAULT_GRAMMER_MODEL
    )


def _grammer_gateway(
    *,
    api_key: str | None,
    base_url: str | None,
    cancellation_check: Callable[[], None] | None = None,
) -> LLMGateway:
    model = _grammer_model()
    return LLMGateway(
        LLMConfig(
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=None,
            temperature=0.0,
            top_p=0.2,
            provider=_provider_from_env("GRAMMER", model),
            extra_body=_grammer_extra_body(model),
            app_label="Grammer",
            cancellation_check=cancellation_check,
        )
    )


def _grammer_extra_body(model: str) -> dict[str, Any] | None:
    normalized = str(model or "").casefold()
    if "gpt-oss" in normalized:
        return {"reasoning": {"effort": "low", "exclude": True}, "include_reasoning": False}
    if "thinking" in normalized:
        return {"reasoning": {"enabled": True, "exclude": True, "max_tokens": 64}, "include_reasoning": False}
    return None


def apply_quality_repair_operations(
    text: str,
    operations: list[QualityRepairOperation],
) -> tuple[str, list[QualityRepairOperation], list[dict[str, Any]]]:
    current = text
    applied: list[QualityRepairOperation] = []
    skipped: list[dict[str, Any]] = []
    for operation in operations[:80]:
        reason = _skip_reason(current, operation)
        if reason:
            skipped.append({**asdict(operation), "skip_reason": reason})
            continue
        current = current.replace(operation.find, operation.replace, 1)
        applied.append(operation)
    return current, applied, skipped


def _quality_repair_prompt(text: str) -> str:
    payload = {
        "task": "minimal_quality_repair",
        "contract": [
            "Return patch operations only. Do not return a full rewritten document.",
            "Fix only obvious grammar, typo, punctuation, subject-verb agreement, missing article, and broken sentence-fragment defects.",
            "Do not optimize for AI detectors. Do not make the text more polished, generic, formal, or academic.",
            "Do not rewrite paragraphs or restructure argument flow.",
            "Preserve all meaning, citations, names, numbers, unit codes, examples, paragraph order, and author voice.",
            "If a sentence is awkward but grammatically acceptable, leave it unchanged.",
            "Each find string must be an exact substring from the submitted text.",
            "Each replacement must be the smallest edit that fixes the defect.",
        ],
        "output_schema": {
            "operations": [
                {
                    "find": "exact original substring",
                    "replace": "minimal corrected substring",
                    "reason": "typo|grammar|punctuation|fragment|agreement|article",
                }
            ]
        },
        "text": text,
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def _parse_operations(payload: Any) -> list[QualityRepairOperation]:
    if not isinstance(payload, dict):
        return []
    rows = payload.get("operations")
    if not isinstance(rows, list):
        return []
    operations: list[QualityRepairOperation] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        find = _clean_patch_text(row.get("find"))
        replace = _clean_patch_text(row.get("replace"))
        reason = " ".join(str(row.get("reason") or "").split())[:80]
        if find and replace and find != replace:
            operations.append(QualityRepairOperation(find=find, replace=replace, reason=reason))
    return operations


def _skip_reason(text: str, operation: QualityRepairOperation) -> str:
    if text.count(operation.find) != 1:
        return "find_text_not_unique"
    if "\n\n" in operation.find or "\n\n" in operation.replace:
        return "paragraph_boundary_change"
    find_words = _word_count(operation.find)
    replace_words = _word_count(operation.replace)
    if find_words > 36:
        return "find_text_too_large"
    if replace_words > max(find_words + 8, int(find_words * 1.6) + 1):
        return "replacement_too_expansive"
    protected_find = _protected_tokens(operation.find)
    protected_replace = _protected_tokens(operation.replace)
    if protected_find != protected_replace:
        return "protected_token_changed"
    return ""


def _protected_tokens(text: str) -> list[str]:
    return re.findall(
        r"\b[A-Z]{2,}[A-Z0-9]*\b|\b[A-Z]{2,}[A-Z0-9]*\d+[A-Z0-9]*\b|\b\d+(?:\.\d+)?\b|\([A-Za-z][^)]*\d{4}[^)]*\)",
        str(text or ""),
    )


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", str(text or "")))


def _clean_patch_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").strip()


def _same_text(left: str, right: str) -> bool:
    return re.sub(r"\s+", " ", str(left or "").strip()) == re.sub(r"\s+", " ", str(right or "").strip())


def _quality_repair_enabled() -> bool:
    value = _bool_env("DRAFTPROOF_V6_GRAMMER_REPAIR_ENABLED")
    return True if value is None else value


def _has_llm_api_key(api_key: str | None) -> bool:
    if str(api_key or "").strip():
        return True
    return bool(_first_env("OPENROUTER_API_KEY", "LLM_API_KEY"))


def _provider_from_env(role: str, model: str) -> dict[str, Any] | None:
    import json

    prefix = f"DRAFTPROOF_V6_{role}_PROVIDER"
    raw_json = _first_env(f"{prefix}_ROUTING_JSON")
    if raw_json:
        parsed = json.loads(raw_json)
        if not isinstance(parsed, dict):
            raise ValueError(f"V6 {role.casefold()} provider routing JSON must be an object")
        return parsed

    provider: dict[str, Any] = {}
    for key, env_name in (
        ("order", f"{prefix}_ORDER"),
        ("only", f"{prefix}_ONLY"),
        ("ignore", f"{prefix}_IGNORE"),
    ):
        values = _csv_env(env_name)
        if values:
            provider[key] = values
    allow_fallbacks = _bool_env(f"{prefix}_ALLOW_FALLBACKS")
    if allow_fallbacks is not None:
        provider["allow_fallbacks"] = allow_fallbacks
    sort = _first_env(f"{prefix}_SORT")
    if sort:
        provider["sort"] = sort
    return provider or None


def _first_env(*names: str) -> str | None:
    import os

    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _csv_env(name: str) -> list[str]:
    value = _first_env(name)
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _bool_env(name: str) -> bool | None:
    value = _first_env(name)
    if value is None:
        return None
    normalized = value.casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None
