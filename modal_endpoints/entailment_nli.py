"""DraftProof Phase-2 — Claim↔Source NLI entailment cross-encoder, Modal endpoint.

STATUS: CODE ONLY — NOT DEPLOYED. This is the N3 ([G1]) serving side of the
Phase-2 entailment pipeline (docs/plans/phase2_entailment_grounding_scope.md
§3/§4). It hosts an NLI cross-encoder that scores premise↔hypothesis pairs and
returns per-pair softmax probabilities over {entailment, neutral, contradiction}.
The client that consumes it is ``poc/claim_graph/entailment.ModalEntailmentScorer``.

WHY A CROSS-ENCODER, NOT AN LLM-JUDGE ([G1], §I): the premise is RETRIEVED WEB
TEXT. An LLM-judge reading it is directly prompt-injectable ("this source
supports all claims") the moment entailment becomes an optimization target. A
cross-encoder consumes premise/hypothesis as SCORED TEXT, not instructions, and
§C caps LLM-judged engines at Moderate confidence — letting one mint Level-4
``verified`` would gut §A. Only this model's score grants ``verified``.

Model: ``MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`` (FEVER/ANLI/
LingNLI/WANLI-trained — apt for claim↔source verification; the plain
``...-mnli-fever-anli`` large id does NOT exist on HF, verified 2026-07-15).
Env-overridable via ``ENTAILMENT_MODEL_ID``. Cheap fallback (swap the env var /
CHECKPOINT_ID) is ``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`` — smaller/
faster, lower ceiling.

LABEL ORDER (do not assume): MNLI/FEVER/ANLI models commonly order labels
{entailment, neutral, contradiction} but this is VERIFIED at load from
``config.id2label`` and the response is NORMALISED to that named order, so a
checkpoint with a different index order still returns correct named scores.

── DEPLOY RUNBOOK (run these MANUALLY — this file is not auto-deployed) ────────
  1. Create the bearer-auth secret (value is what the worker sends as the token):
       modal secret create draftproof-entailment-endpoint-auth \
           DRAFTPROOF_ENTAILMENT_ENDPOINT_TOKEN=<generate-a-strong-token>
  2. (Optional) pick a non-default checkpoint at build time:
       modal deploy modal_endpoints/entailment_nli.py \
           --env ENTAILMENT_MODEL_ID=MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
     …or just:
       modal deploy modal_endpoints/entailment_nli.py
  3. Copy the deployed endpoint URL Modal prints, then set in the worker's .env
     (mirrors DRAFTPROOF_MODAL_ENDPOINT_URL/_TOKEN for the deep-scan detector):
       DRAFTPROOF_ENTAILMENT_ENDPOINT_URL=<deployed url>/score
       DRAFTPROOF_ENTAILMENT_ENDPOINT_TOKEN=<the same token from step 1>
     Until both are set, the client fail-opens → every claim stays ``unverified``
     (the default state pre-deploy; the build always succeeds).
  Dev/local serve (hot-reload, does NOT deploy):
       modal serve modal_endpoints/entailment_nli.py
"""

import logging
import os

import modal
from fastapi import Header
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Pair(BaseModel):
    premise: str
    hypothesis: str


class PairsRequest(BaseModel):
    """Request body: {"pairs": [{"premise": "...", "hypothesis": "..."}, ...]}.

    Typed Pydantic body + ``Header()`` auth mirrors deberta_large_detector.py —
    raw ``fastapi.Request`` injection on an @app.cls bound method reproducibly
    422'd on this Modal version; this is the pattern that works."""

    pairs: list[Pair]


# Checkpoint is PROVISIONAL + env-overridable so swapping it (to the cheap base
# fallback, or a re-calibrated model) is a one-line/one-env change.
CHECKPOINT_ID = os.environ.get(
    "ENTAILMENT_MODEL_ID", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
MAX_LEN = 512  # premise+hypothesis truncated to the encoder's window

# Canonical named NLI classes the endpoint always returns (order-independent —
# populated by index from config.id2label at load).
NLI_CLASSES = ("entailment", "neutral", "contradiction")

app = modal.App("draftproof-entailment-nli")


def _bake_checkpoint():
    """Build-time: download CHECKPOINT_ID into the image so the running container
    never hits HuggingFace at request time (cold-start failure mode)."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=CHECKPOINT_ID, local_dir="/model")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi",
        "transformers==4.46.3",
        "torch==2.5.1",
        "sentencepiece",  # DeBERTa-v3 tokenizer dependency
        "huggingface_hub",
    )
    .env({"ENTAILMENT_MODEL_ID": CHECKPOINT_ID})
    .run_function(_bake_checkpoint)
)

# Bearer token for worker -> Modal auth. References a Modal secret by name;
# created out-of-band via `modal secret create` (see DEPLOY RUNBOOK), not here.
auth_secret = modal.Secret.from_name("draftproof-entailment-endpoint-auth")


@app.cls(
    gpu="L4",  # 24 GB — matches the deep-scan detector tier
    min_containers=0,  # scale to zero — pay per request only
    scaledown_window=120,  # keep warm ~2 min for burst reuse
    image=image,
    secrets=[auth_secret],
)
class EntailmentNLI:
    """NLI cross-encoder container. Loads once per container (@modal.enter),
    serves POST /score. Scope is narrow: pair scoring + raw softmax only — no
    thresholds, no verdict (that is the client's ``classify``), no report logic."""

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("/model")
        self.model = AutoModelForSequenceClassification.from_pretrained("/model")
        self.model.to(self.device)
        self.model.eval()

        # Resolve the model's label order → index map for each named NLI class.
        # id2label maps {int index -> label string}; normalise by lowercased,
        # substring-matched name so a class-order permutation is handled.
        id2label = getattr(self.model.config, "id2label", None) or {}
        self.class_index = {}
        for idx, label in id2label.items():
            name = str(label).strip().lower()
            for cls in NLI_CLASSES:
                if cls in name or name in cls:
                    self.class_index[cls] = int(idx)
        # Fallback to the canonical order if id2label was missing/unmappable.
        for i, cls in enumerate(NLI_CLASSES):
            self.class_index.setdefault(cls, i)
        logger.info(
            "EntailmentNLI.load(): %s on %s; class_index=%s",
            CHECKPOINT_ID, self.device, self.class_index,
        )

    @modal.fastapi_endpoint(method="POST")
    def score(self, body: PairsRequest, authorization: str = Header(None)):
        """POST body: {"pairs": [{"premise": "...", "hypothesis": "..."}, ...]}
        Auth: ``Authorization: Bearer <DRAFTPROOF_ENTAILMENT_ENDPOINT_TOKEN>``.

        Returns per-pair, IN ORDER:
          {"results": [{"label": <argmax class>,
                        "scores": {"entailment": p, "neutral": p, "contradiction": p}}, ...]}
        where scores are softmax probabilities mapped to the named classes via the
        load-time id2label index (so the client never depends on the model's raw
        index order). The client (claim_graph.entailment) applies the §3 thresholds.
        """
        import torch
        from fastapi.responses import JSONResponse

        expected_token = os.environ.get("DRAFTPROOF_ENTAILMENT_ENDPOINT_TOKEN")
        provided_token = (
            authorization[len("Bearer "):]
            if authorization and authorization.startswith("Bearer ")
            else None
        )
        if not expected_token or provided_token != expected_token:
            return JSONResponse(status_code=401,
                                content={"error": "unauthorized", "available": False})

        pairs = body.pairs
        if not pairs:
            return JSONResponse(
                status_code=400,
                content={"error": "expected {'pairs': [{'premise','hypothesis'}, ...]}",
                         "available": False})

        premises = [p.premise for p in pairs]
        hypotheses = [p.hypothesis for p in pairs]
        encoded = self.tokenizer(
            premises, hypotheses,
            padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1).tolist()

        ei = self.class_index["entailment"]
        ni = self.class_index["neutral"]
        ci = self.class_index["contradiction"]
        results = []
        for row in probs:
            scores = {
                "entailment": float(row[ei]),
                "neutral": float(row[ni]),
                "contradiction": float(row[ci]),
            }
            label = max(scores, key=scores.get)
            results.append({"label": label, "scores": scores})

        return JSONResponse(
            status_code=200,
            content={"available": True, "checkpoint": CHECKPOINT_ID, "results": results},
        )
