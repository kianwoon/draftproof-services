"""The standalone formulaic-conclusion / -progression detectors are INTENTIONALLY DISABLED
(NO-HARDCODE): they relied on baked phrase lists (FORMULAIC_CONCLUSION_PATTERNS /
FORMULAIC_PROGRESS_MARKERS / GENERIC_ESSAY_STARTERS), which were overfit. "Formulaic AI prose"
is low-surprisal and is now caught by the agnostic statistical predictability signals
(predictability/top-k/burstiness/paragraph_uniformity) -- validated by the labeled-corpus
end-to-end gate (calibration/measure_end_to_end.py), not by these standalone functions, which
now return 0.0."""
from poc.detect.layer3_scoring import (
    estimate_formulaic_conclusion_risk as concl,
    estimate_formulaic_progression_risk as prog,
)


def test_conclusion_detector_disabled_no_hardcoded_phrases():
    # Formulaic-shaped close: previously flagged by the hardcoded phrase list; the detector is
    # now disabled (the statistical layer carries this signal). Must not fire on baked phrases.
    cooking = ("The kitchen stands at a turning point. In an age of fast food, the goal should be to "
               "cook with care. Ultimately, it is essential that we slow down.")
    assert concl(cooking) == 0.0


def test_conclusion_ignores_specific_grounded_close():
    grounded = "We cut response time from 4.2s to 900ms by caching the three slowest queries in March."
    assert concl(grounded) == 0.0


def test_conclusion_not_triggered_by_bare_adjectives():
    # The old overfit list flagged "thoughtful/capable/responsible"; the agnostic version must not.
    assert concl("She is thoughtful, capable, and responsible in her daily work at the shop.") == 0.0


def test_progression_detector_disabled_no_hardcoded_phrases():
    # Generic scaffolding: previously flagged by FORMULAIC_PROGRESS_MARKERS; detector now disabled.
    march = ("In the past, farms were small. Today, they are industrial. However, this shift created "
             "problems. Another issue is runoff. Overall, the goal should be balance.")
    assert prog(march) == 0.0


def test_progression_low_for_varied_specific_prose():
    varied = ("The 1923 drought killed two-thirds of the herd. Maria rebuilt with goats, which "
              "tolerate the dry ridge behind her farm.")
    assert prog(varied) <= 0.25
