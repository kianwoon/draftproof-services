"""DraftProof V7 — Deep Scan DeBERTa-v3-large detector, Modal serverless GPU endpoint.

STATUS: PROVISIONAL LIVE INFERENCE (2026-07-04). Checkpoint =
desklib/ai-text-detector-academic-v1.01 (public, MIT-licensed, DeBERTa-v3-large
fine-tuned on academic text, #1 on the RAID benchmark at time of writing).
This unblocks "get a real Modal inference endpoint running" quickly, per an
explicit owner decision to narrow Phase 1B scope for now. It does NOT satisfy
the full Task 1B.0 bar below — SCoCESLE ESL calibration has NOT been run
against this checkpoint yet. Consequences, enforced elsewhere, not here:
  - DRAFTPROOF_V7_DEEP_SCAN stays OFF by default (worker-side kill switch).
  - detector_fusion.py's deep_scan_2detector weight (0.60 deberta_large) must
    NOT be treated as validated until calibration lands — this endpoint
    returns a RAW, UNCALIBRATED probability (see score() response) and the
    caller is responsible for not silently treating it as calibrated.

Spec: docs/draftproof_v7_authorship_clarity_spec.md
  - §3.2 "Modal endpoint responsibilities (Deep Scan only)" — this file implements
    that shape.
  - §10 Phase 1B "Deep Scan on Modal".
  - §11 Risks — "DeBERTa-large ESL FPR like raw fakespot (20.5%)" is exactly why
    the calibration step below still matters; do not skip it before flipping
    DRAFTPROOF_V7_DEEP_SCAN or wiring this into real fusion weights.

Remaining Task 1B.0 work (NOT done by this file):
  1. Run the SCoCESLE ESL false-positive gate + AI-set evaluation against
     desklib/ai-text-detector-academic-v1.01 (see poc/calibration/fpr_subgroup_gate.py
     for the pattern used by the existing detector stack). If it fails the
     gate, swap the checkpoint — this pick is provisional, not final.
  2. Calibrate the raw sigmoid output (isotonic or equivalent — see
     poc/calibration/deberta_isotonic.pkl for the fakespot precedent) and
     record the run in MLflow.
  3. Once calibrated, flip DRAFTPROOF_V7_DEEP_SCAN=1 on the worker.

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
CHECKPOINT_ID = "desklib/ai-text-detector-academic-v1.01"
MAX_LEN = 512  # spec §3.2 512-token chunks

# App name is V7/Deep-Scan-scoped on purpose — distinct from the spec's example
# name ("draftproof-deberta-deep-scan") to avoid any future collision with a
# non-V7 Modal app in the same workspace.
app = modal.App("draftproof-v7-deberta-deep-scan")


def _bake_checkpoint():
    """Build-time step: download CHECKPOINT_ID into the image layer so the
    running container never hits HuggingFace at request time (spec §3.2 —
    runtime download is the 60-180s cold-start failure mode)."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=CHECKPOINT_ID, local_dir="/model")


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


@app.cls(
    gpu="L4",  # 24 GB — spec §7.3 recommended tier
    min_containers=0,  # scale to zero — pay per request only, no idle GPU cost
    scaledown_window=120,  # keep warm ~2 min after a request for burst reuse
    image=image,
    secrets=[auth_secret],
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
        self.tokenizer = AutoTokenizer.from_pretrained("/model")
        self.model = DesklibAIDetectionModel.from_pretrained("/model")
        self.model.to(self.device)
        self.model.eval()
        logger.info(
            "DebertaDetector.load(): loaded %s on %s (PROVISIONAL — not yet "
            "SCoCESLE-calibrated, see module docstring)",
            CHECKPOINT_ID,
            self.device,
        )

    @modal.fastapi_endpoint(method="POST")
    def score(self, body: ScoreRequest, authorization: str = Header(None)):
        """POST body: {"chunks": ["...", "..."]}
        Auth: Authorization: Bearer <DRAFTPROOF_MODAL_ENDPOINT_TOKEN> header,
        checked against the auth_secret's env var (spec §3.2 "Auth").

        Returns per-chunk RAW (uncalibrated) probabilities. The caller
        (worker-side detector_fusion.py) must not treat these as calibrated
        until Task 1B.0's SCoCESLE calibration pass lands — see module
        docstring. `"calibrated": false` is included explicitly in every
        response so this can never be silently mistaken for a validated score.

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
                "calibrated": False,
                "checkpoint": CHECKPOINT_ID,
                "chunk_scores": probs,
                "document_score": sum(probs) / len(probs) if probs else None,
            },
        )
