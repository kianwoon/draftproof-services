"""Shared author_review_card cap, importable by both the v6 producer
(``poc/rewrite_v6/production.py``) and the PDF renderer (``poc/report/render_rewrite.py``)
without creating a circular import (``production.py`` already imports ``render_rewrite`` at
module load, so the cap could not live in ``production.py`` itself once the renderer needed it
too).
"""

from __future__ import annotations

import os

AUTHOR_REVIEW_CARD_CAP_DEFAULT = 12


def author_review_card_cap() -> int:
    """Max author_review_cards emitted (both renderers truncate to this). Env-tunable, same
    style as the other v6 switches in this package (unset/invalid -> default)."""
    raw = os.environ.get("DRAFTPROOF_V6_AUTHOR_REVIEW_CARD_CAP", "").strip()
    try:
        value = int(raw)
        if value > 0:
            return value
    except (TypeError, ValueError):
        pass
    return AUTHOR_REVIEW_CARD_CAP_DEFAULT
