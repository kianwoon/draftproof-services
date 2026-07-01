# KEEP IN SYNC with poc/detect/deberta_signal.py (read-time copy; see _composers/__init__.py).
"""Read-time pass-through for the ai_signal_deberta field.

The real inference runs in the worker (poc/detect/deberta_signal.py) during scan and the
result is baked into ai_risk_badge["ai_signal_deberta"] at scan time, so scan reports need
no read-time derivation — the stored field is served as-is. This mirror exists for parity with
the other additive composers and as the hook for any future read-time enrichment (e.g. if a
rewrite report ever needs to re-derive the signal without re-running the worker). MVP: no-op
pass-through.
"""
from __future__ import annotations


def pass_through(ai_signal_deberta: dict | None) -> dict | None:
    """Return the stored field unchanged. Hook for future read-time derivation."""
    return ai_signal_deberta
