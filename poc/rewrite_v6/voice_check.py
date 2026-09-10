"""P1 voice-fidelity check: deterministic distance between the source voice profile and
the rewritten voice profile. PURE measurement — no LLM, no re-loop, no text mutation.

Fail-open everywhere: any error / missing profile / short text => abstain with
score 1.0 (the old behavior — the rewrite result is never blocked or changed by
this check; it is observability only).

Distance axes (all absolute diffs, blended into a 0..1 score where 1.0 == matched):
- formality_mean abs diff
- sentence-length-bucket L1 distance (normalized)
- first-person density diff
- hedging density diff
"""

from __future__ import annotations

try:
    from .voice_profile import build_voice_profile
except ImportError:  # pragma: no cover - dual-import convention (poc root vs pkg)
    from poc.rewrite_v6.voice_profile import build_voice_profile

_ABSTAIN = {
    "abstain": True,
    "score": 1.0,
    "drift_axes": [],
    "source_profile": None,
    "rewrite_profile": None,
}


def _abstain(source_profile: dict | None = None, rewrite_profile: dict | None = None) -> dict:
    out = dict(_ABSTAIN)
    out["source_profile"] = source_profile
    out["rewrite_profile"] = rewrite_profile
    return out


def _bucket_l1(a: dict | None, b: dict | None) -> float:
    total_a = sum(int(v) for v in (a or {}).values()) or 0
    total_b = sum(int(v) for v in (b or {}).values()) or 0
    denom = max(1, total_a, total_b)
    keys = ("short", "med", "long")
    return sum(abs(float((a or {}).get(k) or 0) - float((b or {}).get(k) or 0)) for k in keys) / denom


def voice_fidelity(original_text: str, rewritten_text: str) -> dict:
    """Pure, fail-open voice match score in 0..1 (1.0 == matched / abstained).

    Abstains (score 1.0, abstain=True) when either side is missing or has fewer
    than 4 sentences (build_voice_profile marks them abstain). Never raises."""
    try:
        if (voice_options_disabled()):
            return _abstain()
        src = build_voice_profile(original_text)
        rw = build_voice_profile(rewritten_text)
        if not src or not rw or src.get("abstain") or rw.get("abstain"):
            return _abstain(src, rw)
        formality_drift = abs(float(src["formality_mean"]) - float(rw["formality_mean"]))
        bucket_drift = _bucket_l1(src.get("sent_len_buckets"), rw.get("sent_len_buckets"))
        fp_drift = abs(float(src["first_person_density"]) - float(rw["first_person_density"]))
        hedge_drift = abs(float(src["hedging_density"]) - float(rw["hedging_density"]))
        score = max(0.0, min(1.0, 1.0 - (formality_drift + bucket_drift + 10 * fp_drift + 10 * hedge_drift) / 4.0))
        drift_axes: list[dict] = [
            {"axis": "formality", "source": src["formality_mean"], "rewrite": rw["formality_mean"], "drift": round(formality_drift, 4)},
            {"axis": "sentence_length_mix", "source": src.get("sent_len_buckets"), "rewrite": rw.get("sent_len_buckets"), "drift": round(bucket_drift, 4)},
            {"axis": "first_person", "source": src["first_person_density"], "rewrite": rw["first_person_density"], "drift": round(fp_drift, 4)},
            {"axis": "hedging", "source": src["hedging_density"], "rewrite": rw["hedging_density"], "drift": round(hedge_drift, 4)},
        ]
        return {
            "abstain": False,
            "score": round(score, 4),
            "drift_axes": drift_axes,
            "source_profile": src,
            "rewrite_profile": rw,
        }
    except Exception:
        return _abstain()


def voice_options_disabled() -> bool:
    """voice_options {"disabled": true} (set via contextvar upstream) skips the check."""
    try:
        from .voice_profile import voice_options_var
    except ImportError:  # pragma: no cover
        from poc.rewrite_v6.voice_profile import voice_options_var
    return bool((voice_options_var.get() or {}).get("disabled"))
