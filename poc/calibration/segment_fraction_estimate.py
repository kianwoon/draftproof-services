#!/usr/bin/env python3
"""Two-signal Turnitin estimator (segment-fraction model).

Turnitin reports its AI score as the **fraction of qualifying sentences it flags**, and — from the
one labeled doc we have (Reflection AT3: real Turnitin AI = 27%, flagged segments in the formal
Critical-Analysis middle, conclusion spared) — it keys on BOTH:
  * predictability  (our per-sentence top-10 token ratio), AND
  * formal-academic register (nominalisations + academic connectives + long Latinate words,
    minus first-person voice).

Predictability alone over-flagged the *plain, personal conclusion* that Turnitin left alone;
multiplying in register re-localises the flag onto the formal argumentative middle. A sentence is
flagged when BOTH signals clear their bar; the estimate is the flagged share of qualifying words.

CALIBRATION HONESTY: the two thresholds below were fit to reproduce ONE labeled doc's 27% (and to
exclude its conclusion). They are UNVALIDATED for other documents — treat the output as an estimate
band, not a precise Turnitin %. Add more labeled reports to poc/calibration/turnitin_cases/ and
re-fit before trusting it as a displayed number. See poc/calibration/calibrate.py.

Run:  ~/.pyenv/versions/3.11.0/bin/python3 poc/calibration/segment_fraction_estimate.py test_content12.txt
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# --- Calibrated to Reflection AT3 (Turnitin AI 27%); 1 doc, UNVALIDATED for generalisation. ---
TOP10_THRESHOLD = 0.45       # per-sentence predictability bar
REGISTER_THRESHOLD = 1.0     # per-sentence formal-academic register bar

_NOMINAL = re.compile(r"(tion|sion|ment|ance|ence|ity|ism)s?$", re.I)
_CONNECTIVES = {
    "furthermore", "thereby", "whilst", "hence", "thus", "moreover", "consequently",
    "nevertheless", "whereas", "inasmuch", "therein", "herein", "notwithstanding",
}
_FIRST_PERSON = {"i", "my", "me", "i'm", "i've", "myself", "mine"}


def register_score(text: str) -> float:
    """Formal-academic register of a sentence (higher = more 'generated-essay' sounding).
    nominalisation density + academic connectives + long-word bonus - first-person voice."""
    words = re.findall(r"[A-Za-z']+", text.lower())
    n = max(1, len(words))
    nominal = sum(1 for t in words if _NOMINAL.search(t)) / n
    conn = sum(1 for t in words if t in _CONNECTIVES) / n
    first_person = sum(1 for t in words if t in _FIRST_PERSON) / n
    mean_len = sum(len(t) for t in words) / n
    return nominal * 4 + conn * 6 + max(0.0, mean_len - 4.3) * 0.5 - first_person * 5


def _per_sentence(text: str) -> list[dict]:
    """Per-sentence (predictability top10 + register) from the detector's scan."""
    from rewrite_v3.pipeline import _scan_report
    sents = (_scan_report(text).get("predictability") or {}).get("sentences") or []
    out = []
    for s in sents:
        t = s.get("text", "")
        top10 = s.get("top10")
        words = len(t.split())
        if isinstance(top10, (int, float)) and words:
            out.append({
                "sentence_id": s.get("sentence_id"),
                "text": t,
                "words": words,
                "top10": round(float(top10), 3),
                "register": round(register_score(t), 2),
            })
    return out


def segment_fraction_estimate(text: str) -> dict:
    """Estimate a Turnitin-style AI percentage as the flagged share of qualifying sentences.

    Returns: {estimate_pct, flagged:[...], qualifying_sentences, qualifying_words, params}.
    A sentence is flagged when top10 >= TOP10_THRESHOLD AND register >= REGISTER_THRESHOLD.
    """
    sents = _per_sentence(text)
    total_words = sum(s["words"] for s in sents)
    flagged = [s for s in sents
               if s["top10"] >= TOP10_THRESHOLD and s["register"] >= REGISTER_THRESHOLD]
    flagged_words = sum(s["words"] for s in flagged)
    pct = round(100 * flagged_words / total_words, 1) if total_words else 0.0
    return {
        "estimate_pct": pct,
        "flagged": flagged,
        "qualifying_sentences": len(sents),
        "qualifying_words": total_words,
        "params": {"top10_threshold": TOP10_THRESHOLD, "register_threshold": REGISTER_THRESHOLD,
                   "calibrated_on": "reflection_at3 (Turnitin 27%)", "validated": False},
    }


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "test_content12.txt"
    repo = Path(__file__).resolve().parent.parent.parent
    env = repo / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v.strip().strip('"').strip("'"))
    sys.path.insert(0, str(repo)); sys.path.insert(0, str(repo / "poc"))
    text = (repo / path).read_text() if not Path(path).is_absolute() else Path(path).read_text()

    r = segment_fraction_estimate(text)
    print(f"Turnitin estimate (segment-fraction, 2-signal): ~{r['estimate_pct']:.0f}%   "
          f"[{len(r['flagged'])}/{r['qualifying_sentences']} sentences, UNVALIDATED — 1-doc calibration]")
    print(f"\nFlagged (formal + predictable) sentences:")
    for s in r["flagged"]:
        print(f"  {s['sentence_id']}  top10={s['top10']:.2f} reg={s['register']:.1f}  {s['text'][:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
