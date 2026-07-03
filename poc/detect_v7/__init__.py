"""V7 Authorship Clarity Breakdown — additive detection layer.

Purely additive on top of the production ``poc/detect/`` pipeline: imports
FROM ``poc/detect`` signals, never modifies them, and never mutates
``ai_likelihood`` / tier / badge. See
``docs/draftproof_v7_authorship_clarity_spec.md`` for the full design.

Phase 1A scope (this package, as scaffolded): config loader (``config.py``)
and weight/threshold data (``weights.json``) only. Signal adaptation,
detector fusion, category scoring, aggregation, and the report composer are
later Phase 1A steps per spec §10.
"""
