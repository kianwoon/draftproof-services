"""Text integrity helpers for V3 LLM candidate boundaries."""

from __future__ import annotations

import unicodedata
from typing import Any


def raw_completion_integrity(text: str, *, max_chars: int = 24000) -> dict[str, Any]:
    """Detect corrupted raw LLM responses before JSON parsing or salvage."""

    value = str(text or "")
    chars = list(value)
    char_count = max(1, len(chars))
    failures: list[str] = []
    format_count = 0
    control_count = 0
    emoji_like_count = 0
    symbol_count = 0
    non_language_run = 0
    max_non_language_run = 0
    for char in chars:
        category = unicodedata.category(char)
        if category == "Cf":
            format_count += 1
        if category == "Cc" and char not in "\n\t\r":
            control_count += 1
        if category.startswith("S"):
            symbol_count += 1
        if ord(char) >= 0x1F000:
            emoji_like_count += 1
        if char.isalnum() or char.isspace() or char in "\"':,._-":
            non_language_run = 0
        else:
            non_language_run += 1
            max_non_language_run = max(max_non_language_run, non_language_run)

    open_braces = value.count("{")
    close_braces = value.count("}")
    open_brackets = value.count("[")
    close_brackets = value.count("]")
    symbol_ratio = symbol_count / char_count
    if len(value) > max_chars:
        failures.append("completion_too_long")
    if format_count:
        failures.append("format_character_injection")
    if control_count:
        failures.append("control_character_injection")
    if emoji_like_count:
        failures.append("emoji_or_decorative_symbol_injection")
    if symbol_ratio > 0.08:
        failures.append("unicode_symbol_burst")
    if max_non_language_run > 80:
        failures.append("non_language_symbol_run")
    if abs(open_braces - close_braces) > 3 or abs(open_brackets - close_brackets) > 3:
        failures.append("container_balance_corruption")
    return {
        "passed": not failures,
        "failures": failures,
        "metrics": {
            "char_count": len(chars),
            "max_chars": max_chars,
            "format_count": format_count,
            "control_count": control_count,
            "emoji_like_count": emoji_like_count,
            "symbol_ratio": round(symbol_ratio, 4),
            "max_non_language_run": max_non_language_run,
            "brace_balance": open_braces - close_braces,
            "bracket_balance": open_brackets - close_brackets,
        },
    }


def minimal_replacement_text_integrity(text: str) -> dict[str, Any]:
    """Reject non-language artifacts before candidate text reaches scanners."""

    value = str(text or "")
    chars = list(value)
    char_count = max(1, len(chars))
    failures: list[str] = []
    symbol_count = 0
    emoji_like_count = 0
    format_count = 0
    control_count = 0
    angle_count = 0
    max_non_language_run = 0
    non_language_run = 0
    repeated_markup_runs: list[dict[str, Any]] = []
    current_repeat_char = ""
    current_repeat_start = 0
    current_repeat_len = 0

    def flush_repeat(end_index: int) -> None:
        nonlocal current_repeat_char, current_repeat_start, current_repeat_len
        if current_repeat_len >= 2 and current_repeat_char:
            category = unicodedata.category(current_repeat_char)
            if category.startswith(("P", "S")) and current_repeat_char not in ".!?,-":
                repeated_markup_runs.append({
                    "char_category": category,
                    "start": current_repeat_start,
                    "end": end_index,
                    "length": current_repeat_len,
                })
        current_repeat_char = ""
        current_repeat_len = 0
        current_repeat_start = end_index

    for index, char in enumerate(chars):
        category = unicodedata.category(char)
        if char.isalnum() or char.isspace():
            non_language_run = 0
        else:
            non_language_run += 1
            max_non_language_run = max(max_non_language_run, non_language_run)
        if category.startswith("S"):
            symbol_count += 1
        if ord(char) >= 0x1F000:
            emoji_like_count += 1
        if category == "Cf":
            format_count += 1
        if category == "Cc" and char not in "\n\t\r":
            control_count += 1
        if char in "<>":
            angle_count += 1
        if char == current_repeat_char:
            current_repeat_len += 1
        else:
            flush_repeat(index)
            current_repeat_char = char
            current_repeat_start = index
            current_repeat_len = 1
    flush_repeat(len(chars))

    symbol_ratio = symbol_count / char_count
    if control_count:
        failures.append("control_character_injection")
    if format_count:
        failures.append("format_character_injection")
    if emoji_like_count:
        failures.append("emoji_or_decorative_symbol_injection")
    if symbol_ratio > 0.03:
        failures.append("unicode_symbol_burst")
    if angle_count:
        failures.append("markup_angle_bracket_artifact")
    if repeated_markup_runs:
        failures.append("repeated_markup_punctuation_artifact")
    if max_non_language_run > 12:
        failures.append("non_language_symbol_run")
    return {
        "passed": not failures,
        "failures": failures,
        "metrics": {
            "char_count": len(chars),
            "symbol_ratio": round(symbol_ratio, 4),
            "emoji_like_count": emoji_like_count,
            "format_count": format_count,
            "control_count": control_count,
            "angle_count": angle_count,
            "max_non_language_run": max_non_language_run,
            "repeated_markup_run_count": len(repeated_markup_runs),
        },
        "repeated_markup_runs": repeated_markup_runs[:4],
    }
