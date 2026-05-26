from __future__ import annotations

import re
from dataclasses import replace
from typing import Any


def annotate_review_items(variant: Any, source_text: str) -> Any:
    if getattr(variant, "source", "") == "source_preserved":
        return variant
    if getattr(variant, "author_review_items", None) or getattr(variant, "author_proxy_provenance", None):
        return variant
    source_words = _content_word_set(source_text)
    candidate_words = _content_word_set(getattr(variant, "text", ""))
    flagged = [
        word for word in sorted(candidate_words - source_words)
        if _word_base(word) not in source_words
        if len(word) >= 8 or word.endswith(("tion", "ment", "ity", "ness", "ance", "ence", "form"))
    ]
    if not flagged:
        return variant
    return replace(variant, author_review_items=[{
        "item_id": "auto_bridge_001",
        "provenance": "needs_author_confirmation",
        "target_text": " ".join(flagged[:12]),
        "generated_text": "Generated bridge wording not directly present in the submitted paragraph.",
        "user_input_needed": "Confirm, replace, or remove these inferred bridge terms before final use.",
        "author_task": "Review the rewritten paragraph for factual scope and author-owned wording.",
    }])


def _word_base(word: str) -> str:
    return str(word or "").removesuffix("'s").rstrip("s")


def _content_word_set(text: str) -> set[str]:
    stop = {
        "about", "above", "after", "again", "against", "also", "because", "before",
        "being", "between", "could", "every", "from", "have", "into", "more",
        "most", "only", "other", "over", "should", "still", "their", "there",
        "these", "those", "through", "under", "where", "which", "while", "would",
    }
    return {
        token.casefold().strip("'’")
        for token in re.findall(r"[A-Za-z][A-Za-z'’-]{2,}", str(text or ""))
        if token.casefold().strip("'’") not in stop
    }
