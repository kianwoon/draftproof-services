import json

from calibration.v12_validation import tune_weights as tw


def _rows():
    # Two synthetic classes whose signals are trivially separable on
    # generic_density so ANY sane search finds a discriminating vector.
    base = {k: 0.5 for k in (
        "specificity_score", "grounding_gap", "sentence_variance",
        "generic_density", "author_voice_absence", "sentence_smoothness",
        "local_style_shift", "semantic_drift", "predictable_structure",
        "paraphrase_pattern_score", "meaning_preservation_score",
    )}
    rows = []
    for i in range(10):
        s = dict(base, generic_density=0.05, specificity_score=0.9)
        rows.append({"label": "student_owned", "doc_key": f"h{i}",
                     "v7_signals": {**s, "signal_status": {}},
                     "calibrated_detector_score": 0.1, "has_comparison_text": False})
        a = dict(base, generic_density=0.95, specificity_score=0.1)
        rows.append({"label": "ai_generated_like", "doc_key": f"a{i}",
                     "v7_signals": {**a, "signal_status": {}},
                     "calibrated_detector_score": 0.9, "has_comparison_text": False})
    return rows


def test_split_is_stratified_and_deterministic():
    rows = _rows()
    t1, h1 = tw.split_rows(rows, seed=42)
    t2, h2 = tw.split_rows(rows, seed=42)
    assert [r["doc_key"] for r in t1] == [r["doc_key"] for r in t2]
    labels_t = {r["label"] for r in t1}
    assert labels_t == {"student_owned", "ai_generated_like"}


def test_evaluate_uses_real_scorer(tmp_path):
    rows = _rows()
    cand = tw.load_base_weights()
    path = tmp_path / "cand.json"
    path.write_text(json.dumps(cand))
    metrics = tw.evaluate_candidate(path, rows)
    assert set(metrics) >= {"macro_primary_accuracy", "student_owned_false_ai_rate", "per_class"}
    assert 0.0 <= metrics["macro_primary_accuracy"] <= 1.0


def test_candidate_weights_normalized():
    import random
    rng = random.Random(42)
    cand = tw.sample_candidate(tw.load_base_weights(), rng)
    for cat, entries in cand["category_weights"].items():
        total = sum(e["weight"] for e in entries)
        assert abs(total - 1.0) < 1e-6, cat
        # Floor is enforced up to float precision: _floor_and_renormalize's
        # final `x/sum` step can leave a pinned entry a few ULPs below 0.02
        # (e.g. 0.02 - 3.5e-18), and the water-filling itself uses a 1e-12
        # tolerance internally. Assert against that same tolerance rather than
        # exact >= 0.02. This surfaced when the detector-gated specificity split
        # (2026-07-08) gave ai_generated_like a 7th signal with 5 entries pinned
        # at the floor — see weights.json _notes.category_weights_tuning.
        assert all(e["weight"] >= 0.02 - 1e-9 for e in entries), cat
    band = cand["ai_assisted_polished_band"]
    lo, hi = tw.band_bounds(band)
    assert lo < hi
