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
    embedded_word_period_count = 0
    sentence_punctuation_spacing_count = 0
    nested_parenthetical_count = 0
    max_parenthetical_span = 0
    parenthetical_start_stack: list[int] = []
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
        if char == "." and _period_embeds_word_fragment(value, index):
            embedded_word_period_count += 1
        if char in ".?!" and _sentence_punctuation_missing_space(value, index):
            sentence_punctuation_spacing_count += 1
        if char == "(":
            if parenthetical_start_stack:
                nested_parenthetical_count += 1
            parenthetical_start_stack.append(index)
        elif char == ")" and parenthetical_start_stack:
            start_index = parenthetical_start_stack.pop()
            max_parenthetical_span = max(max_parenthetical_span, index - start_index + 1)
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
    if embedded_word_period_count:
        failures.append("embedded_sentence_punctuation_word_artifact")
    if sentence_punctuation_spacing_count:
        failures.append("sentence_punctuation_spacing_artifact")
    if nested_parenthetical_count:
        failures.append("nested_parenthetical_artifact")
    if max_parenthetical_span > 160:
        failures.append("overlong_parenthetical_artifact")
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
            "embedded_word_period_count": embedded_word_period_count,
            "sentence_punctuation_spacing_count": sentence_punctuation_spacing_count,
            "nested_parenthetical_count": nested_parenthetical_count,
            "max_parenthetical_span": max_parenthetical_span,
            "max_non_language_run": max_non_language_run,
            "repeated_markup_run_count": len(repeated_markup_runs),
        },
        "repeated_markup_runs": repeated_markup_runs[:4],
    }


def _period_embeds_word_fragment(value: str, index: int) -> bool:
    if index <= 0 or index >= len(value) - 1:
        return False
    before_char = value[index - 1]
    after_char = value[index + 1]
    if not before_char.isalpha() or not after_char.isalpha() or not after_char.islower():
        return False
    before_start = index - 1
    while before_start >= 0 and value[before_start].isalpha():
        before_start -= 1
    after_end = index + 1
    while after_end < len(value) and value[after_end].isalpha():
        after_end += 1
    before_word = value[before_start + 1:index]
    after_word = value[index + 1:after_end]
    return len(before_word) >= 3 and len(after_word) >= 3


def _sentence_punctuation_missing_space(value: str, index: int) -> bool:
    if index >= len(value) - 1:
        return False
    next_char = value[index + 1]
    if not next_char.isalpha():
        return False
    previous_char = value[index - 1] if index > 0 else ""
    if previous_char.isdigit():
        return False
    if previous_char.isalpha() and next_char.isupper() and index + 2 < len(value) and value[index + 2] == ".":
        return False
    return next_char.isupper() or not previous_char.isalpha()
