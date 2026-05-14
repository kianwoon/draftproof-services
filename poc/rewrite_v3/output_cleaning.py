"""Output cleaning for rewrite V3 model candidates."""

from __future__ import annotations


def _looks_like_wrapper_line(line: str) -> bool:
    stripped = str(line or "").strip()
    if not stripped:
        return False
    lower = stripped.lower()
    if not lower.endswith(":"):
        return False
    if len(stripped.split()) > 24:
        return False
    signals = [
        "here is",
        "repaired text",
        "rewritten text",
        "rewritten essay",
        "adjusted paragraph",
        "matching the",
        "source structure",
        "unit structure",
    ]
    hits = sum(1 for signal in signals if signal in lower)
    if hits >= 2:
        return True
    return hits >= 1 and ("source" in lower or "structure" in lower or "boundaries" in lower)


def clean_v3_candidate_output(raw: str) -> str:
    """Remove generic wrappers without changing candidate prose."""

    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if text.startswith('"""') and text.endswith('"""'):
        text = text[3:-3].strip()
    units = [unit.strip() for unit in text.split("\n\n") if unit.strip()]
    while units and _looks_like_wrapper_line(units[0]):
        units = units[1:]
    return "\n\n".join(units).strip()
