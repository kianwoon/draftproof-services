from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from poc.rewrite_v6 import production as v6_production
from poc.rewrite_v6.scan import scan_text


def test_v6_production_adapter_uses_configured_three_pass_budget(tmp_path, monkeypatch):
    captured: dict[str, int] = {}

    def fake_run_v6_rewrite_all(text, **kwargs):
        captured["max_passes"] = kwargs["max_passes"]
        scan = scan_text(text)
        return SimpleNamespace(
            initial_scan=scan,
            final_scan=scan,
            passes=[],
            pass_trace=[],
            rewritten_text=text,
            final_text_before_quality_repair=None,
            quality_repair=None,
        )

    monkeypatch.setenv("DRAFTPROOF_V6_MAX_PASSES", "3")
    monkeypatch.setattr(v6_production, "run_v6_rewrite_all", fake_run_v6_rewrite_all)
    monkeypatch.setattr(v6_production, "render_pdf", lambda _md, path: Path(path).write_bytes(b"%PDF"))
    monkeypatch.setattr(v6_production, "render_rewrite_report", lambda **_kwargs: "# Rewrite")
    monkeypatch.setattr(v6_production, "_sentence_comparison", lambda _original, _final: [])
    monkeypatch.setattr(v6_production, "_scan_report_for_summary", lambda text, **_kwargs: v6_production._scan_report_shape(scan_text(text).to_dict()))

    result = v6_production.run_rewrite_pipeline_v6(
        detect_json={"input_text": "The review moved from intake to approval."},
        output_dir=str(tmp_path),
        model="writer-model",
    )
    summary = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))

    assert captured["max_passes"] == 3
    assert summary["rewrite_effective_config"]["max_passes"] == 3
