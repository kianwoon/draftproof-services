"""STAGING deploy (2026-07-14): fine-tune v1 checkpoint served from the
draftproof-finetune-v1 volume on a SEPARATE app. Prod deep-scan untouched.
"""
"""DraftProof V7 — Deep Scan DeBERTa-v3-large detector, Modal serverless GPU endpoint.

STATUS: LIVE INFERENCE, CALIBRATED CHECKPOINT (2026-07-04). Checkpoint =
desklib/ai-text-detector-academic-v1.01 (public, MIT-licensed, DeBERTa-v3-large
fine-tuned on academic text, #1 on the RAID benchmark at time of writing).
SCoCESLE ESL calibration for this checkpoint landed 2026-07-04 via sentence
threshold-proportion — parameters live client-side in
poc/detect_v7/weights.json's ``deep_scan_calibration`` (see its
``_provenance`` for the gate numbers: sentence AUC 0.929, parity gap -1.2pp).
The endpoint itself still returns RAW per-chunk sigmoid probabilities; the
worker applies the threshold-proportion calibration. ``"calibrated"`` in the
response reports whether the SERVING checkpoint is one with a landed
SCoCESLE calibration (see CALIBRATED_CHECKPOINT_IDS below) — the client
(poc/detect_v7/pipeline_bridge.py) treats that flag as authoritative for the
``deep_scan_uncalibrated`` uncertainty flag.
  - DRAFTPROOF_V7_DEEP_SCAN remains an operational kill switch (Modal call
    cost/latency), no longer a calibration gate.

Spec: docs/draftproof_v7_authorship_clarity_spec.md
  - §3.2 "Modal endpoint responsibilities (Deep Scan only)" — this file implements
    that shape.
  - §10 Phase 1B "Deep Scan on Modal".
  - §11 Risks — "DeBERTa-large ESL FPR like raw fakespot (20.5%)" is exactly why
    the calibration step below still matters; do not skip it before flipping
    DRAFTPROOF_V7_DEEP_SCAN or wiring this into real fusion weights.

Task 1B.0 calibration status (was "remaining work" before 2026-07-04):
  DONE — SCoCESLE gate + sentence threshold-proportion calibration ran
  against desklib/ai-text-detector-academic-v1.01 and landed in
  poc/detect_v7/weights.json (``deep_scan_calibration``) and
  poc/calibration/v7_deberta_academic_baseline.json. If the checkpoint is
  ever swapped, the new one starts UNCALIBRATED: leave it out of
  CALIBRATED_CHECKPOINT_IDS (so responses report calibrated=false) until its
  own calibration pass lands.

Deploy command: modal deploy modal_endpoints/deberta_large_detector.py
Dev/local serve (hot-reload, does NOT deploy):  modal serve modal_endpoints/deberta_large_detector.py

Secret setup (already done for this deployment — see modal_endpoints/README.md
for how to rotate):
    modal secret create draftproof-v7-modal-endpoint-auth DRAFTPROOF_MODAL_ENDPOINT_TOKEN=<value>

Worker-side env vars that consume this endpoint (see spec §3.2, §10 Phase 1B.2):
    DRAFTPROOF_MODAL_ENDPOINT_URL
    DRAFTPROOF_MODAL_ENDPOINT_TOKEN
"""

import logging
import os

import modal
from fastapi import Header
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ScoreRequest(BaseModel):
    """Request body shape: {"chunks": ["...", "..."]}. Using a typed Pydantic
    model (rather than a raw fastapi.Request) because raw Request injection
    on an @app.cls bound method reproducibly 422'd on this Modal version
    (1.5.1) — verified live across both sync and async attempts, see score()
    docstring. Header-based auth via fastapi.Header() is the compatible
    alternative for reading the bearer token."""

    chunks: list[str]

# Checkpoint is PROVISIONAL — see module docstring. Named as a constant (not
# buried inline) specifically so swapping it after SCoCESLE calibration is a
# one-line change, not a search-and-replace.
CHECKPOINT_ID = "draftproof/finetune-v1-gpt55"  # STAGING: fine-tuned 2026-07-14, served from the finetune volume
# Checkpoints with a LANDED SCoCESLE calibration (parameters in
# poc/detect_v7/weights.json deep_scan_calibration). The response's
# "calibrated" flag derives from membership here — never a hardcoded bool —
# so swapping CHECKPOINT_ID to an uncalibrated model automatically reports
# calibrated=false to the client until its own calibration pass lands.
CALIBRATED_CHECKPOINT_IDS = {
    # fine-tune v1: SCoCESLE-calibrated 2026-07-14 (sweep: sent>=0.999/floor 0.3 holds,
    # 0.0/0.0 ESL FPR, TPR 96.9; fused gate PASS: AUC 0.9982, FPR 0.0 at 40/50/60, parity 0)
    "draftproof/finetune-v1-gpt55",
    "desklib/ai-text-detector-academic-v1.01",  # SCoCESLE-calibrated 2026-07-04
}
MAX_LEN = 512  # spec §3.2 512-token chunks

# App name is V7/Deep-Scan-scoped on purpose — distinct from the spec's example
# name ("draftproof-deberta-deep-scan") to avoid any future collision with a
# non-V7 Modal app in the same workspace.
app = modal.App("draftproof-staging-finetune-v1")


def _bake_checkpoint():
    """STAGING: weights come from the mounted finetune volume, not HF — nothing
    to bake. Kept as a no-op so the image definition below stays unchanged."""
    return None


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi",
        "transformers==4.46.3",
        "torch==2.5.1",
        "huggingface_hub",
    )
    .run_function(_bake_checkpoint)
)

# Bearer token for worker -> Modal auth (spec §3.2 "Auth"). References a
# Modal secret by name; created out-of-band via `modal secret create` (see
# module docstring "Secret setup"), not by this file.
auth_secret = modal.Secret.from_name("draftproof-v7-modal-endpoint-auth")


vol = modal.Volume.from_name("draftproof-finetune-v1")


@app.cls(
    gpu="L4",
    min_containers=0,
    scaledown_window=120,
    image=image,
    secrets=[auth_secret],
    volumes={"/finetune": vol},
)
class DebertaDetector:
    """Deep Scan detector container. Loads once per container (@modal.enter),
    serves POST /score. Scope is intentionally narrow per spec §3.2: chunk
    inference + raw scores only — no report logic, no fakespot, no MiniLM,
    no LanguageTool.
    """

    @modal.enter()
    def load(self):
        import torch
        import torch.nn as nn
        from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

        class DesklibAIDetectionModel(PreTrainedModel):
            """Verbatim architecture from the checkpoint's model card: mean
            pooling over the DeBERTa backbone + a single linear logit head."""

            config_class = AutoConfig

            def __init__(self, config):
                super().__init__(config)
                self.model = AutoModel.from_config(config)
                self.classifier = nn.Linear(config.hidden_size, 1)
                self.init_weights()

            def forward(self, input_ids, attention_mask=None):
                outputs = self.model(input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs[0]
                mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                summed = torch.sum(last_hidden_state * mask, dim=1)
                counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                pooled = summed / counts
                logits = self.classifier(pooled)
                return {"logits": logits}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("/finetune/model_out")
        self.model = DesklibAIDetectionModel.from_pretrained("/finetune/model_out")
        self.model.to(self.device)
        self.model.eval()
        logger.info(
            "DebertaDetector.load(): loaded %s on %s (calibrated=%s — see "
            "CALIBRATED_CHECKPOINT_IDS / module docstring)",
            CHECKPOINT_ID,
            self.device,
            CHECKPOINT_ID in CALIBRATED_CHECKPOINT_IDS,
        )

    @modal.fastapi_endpoint(method="POST")
    def score(self, body: ScoreRequest, authorization: str = Header(None)):
        """POST body: {"chunks": ["...", "..."]}
        Auth: Authorization: Bearer <DRAFTPROOF_MODAL_ENDPOINT_TOKEN> header,
        checked against the auth_secret's env var (spec §3.2 "Auth").

        Returns per-chunk RAW sigmoid probabilities; the client applies the
        sentence threshold-proportion calibration (weights.json
        deep_scan_calibration). `"calibrated"` reports whether the serving
        checkpoint has a landed SCoCESLE calibration (membership in
        CALIBRATED_CHECKPOINT_IDS) — pipeline_bridge.py treats it as
        authoritative for the deep_scan_uncalibrated uncertainty flag.

        NOTE on signature: raw `fastapi.Request` injection on this @app.cls
        bound method reproducibly 422'd ("query.request: Field required") on
        Modal 1.5.1 across both sync and async attempts — verified live, not
        assumed. Switched to the more standard FastAPI idiom instead: a typed
        Pydantic body model (`ScoreRequest`) plus `Header()` for the bearer
        token. This is the pattern that actually works on this SDK version.
        """
        import torch
        from fastapi.responses import JSONResponse

        expected_token = os.environ.get("DRAFTPROOF_MODAL_ENDPOINT_TOKEN")
        provided_token = (
            authorization[len("Bearer "):]
            if authorization and authorization.startswith("Bearer ")
            else None
        )

        if not expected_token or provided_token != expected_token:
            return JSONResponse(
                status_code=401,
                content={"error": "unauthorized", "available": False},
            )

        chunks = body.chunks
        if not chunks:
            return JSONResponse(
                status_code=400,
                content={"error": "expected {'chunks': [str, ...]}", "available": False},
            )

        encoded = self.tokenizer(
            chunks,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs["logits"]).squeeze(-1).tolist()

        if isinstance(probs, float):
            probs = [probs]

        return JSONResponse(
            status_code=200,
            content={
                "available": True,
                "calibrated": CHECKPOINT_ID in CALIBRATED_CHECKPOINT_IDS,
                "checkpoint": CHECKPOINT_ID,
                "chunk_scores": probs,
                "document_score": sum(probs) / len(probs) if probs else None,
            },
        )
