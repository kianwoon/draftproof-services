# modal_endpoints/

Modal serverless GPU endpoint(s) for DraftProof V7 "Authorship Clarity Breakdown"
Deep Scan. See `docs/draftproof_v7_authorship_clarity_spec.md` §3.2 and §10
Phase 1B for the full design.

**Status: LIVE, PROVISIONAL (deployed 2026-07-04).** `deberta_large_detector.py`
is deployed and serving real inference from `desklib/ai-text-detector-academic-v1.01`
(public, MIT-licensed, DeBERTa-v3-large fine-tuned on academic text, #1 on the
RAID benchmark). This was an explicit scope narrowing — "get a real Modal
endpoint running quickly" — separate from full Task 1B.0 compliance below,
which is still incomplete. Weights are baked into the image at build time
(never runtime-downloaded).

Deployed endpoint: `https://kianwoon--draftproof-v7-deberta-deep-scan-debertadetector-score.modal.run`
Auth token: in repo `.env` as `DRAFTPROOF_MODAL_ENDPOINT_TOKEN` (gitignored, not committed).

**Known limitation, confirmed by direct testing, not a guess:** the raw
(uncalibrated) sigmoid output is heavily skewed toward "AI" — a clearly
human, personal anecdote scored 93.9%, a generic AI-style paragraph scored
99.99%. There IS discriminative signal (the two are ~6 points apart), but the
raw threshold is nowhere near a usable operating point. Verified this is not
a `max_length` preprocessing bug (512 vs 768 tokens gave identical results,
as expected — mean pooling is invariant to extra padding). This is exactly
the uncalibrated-sigmoid behavior Task 1B.0's SCoCESLE calibration exists to
fix — every response is marked `"calibrated": false` for this reason, and
`DRAFTPROOF_V7_DEEP_SCAN` must stay OFF on the worker until that calibration
lands.

## Deploy

```bash
modal deploy modal_endpoints/deberta_large_detector.py
```

If you change the code and a redeploy doesn't seem to take effect (identical
error persists across unrelated signature changes), stop the app first —
Modal appears to cache web-endpoint routing more aggressively than function
code across `deploy` calls in some cases:

```bash
modal app list                      # find the App ID
modal app stop <app-id> --yes
modal deploy modal_endpoints/deberta_large_detector.py
```

## Local dev serve (hot-reload, does not deploy)

```bash
modal serve modal_endpoints/deberta_large_detector.py
```

## Calling the endpoint

```bash
curl -X POST "$DRAFTPROOF_MODAL_ENDPOINT_URL" \
  -H "Authorization: Bearer $DRAFTPROOF_MODAL_ENDPOINT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"chunks": ["text chunk 1", "text chunk 2"]}'
```

Response shape:
```json
{"available": true, "calibrated": false, "checkpoint": "desklib/ai-text-detector-academic-v1.01",
 "chunk_scores": [0.99, 0.94], "document_score": 0.965}
```

Signature note: the endpoint method uses a Pydantic body model (`ScoreRequest`)
plus `authorization: str = Header(None)` for auth — NOT a raw `fastapi.Request`
parameter. Raw `Request` injection (both sync and async) reproducibly 422'd
("query.request: Field required") on this `@app.cls`-bound method under Modal
1.5.1, verified live across multiple attempts. The Pydantic+Header pattern is
what actually works.

## Worker-side env vars (consumer, not set here)

- `DRAFTPROOF_MODAL_ENDPOINT_URL` — the deployed endpoint URL (in `.env`)
- `DRAFTPROOF_MODAL_ENDPOINT_TOKEN` — bearer token, must match the Modal secret below (in `.env`)

## Secret (already created for this deployment)

```bash
modal secret create draftproof-v7-modal-endpoint-auth DRAFTPROOF_MODAL_ENDPOINT_TOKEN=<value> --force
```

`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` in the repo `.env` are deploy-time
credentials only — unrelated to the runtime bearer token above.

## Remaining Task 1B.0 work (before DRAFTPROOF_V7_DEEP_SCAN can be flipped on)

- [x] Get a live Modal inference endpoint running (this file, 2026-07-04)
- [ ] Confirm `desklib/ai-text-detector-academic-v1.01` is the final checkpoint,
      or evaluate alternatives — the RAID-benchmark ranking doesn't guarantee
      good behavior on this specific SCoCESLE/DraftProof distribution
- [ ] Run SCoCESLE ESL false-positive gate + AI-set evaluation on this checkpoint
- [ ] Calibrate raw output (isotonic or equivalent, same pattern as
      `poc/calibration/deberta_isotonic.pkl`); record run in MLflow
- [ ] Wire `worker/`-side `detector_fusion.py` deep-scan (2-detector) path to
      actually call this endpoint (not yet done — `pipeline_bridge.py`
      currently only exercises the quick-scan/fakespot-only fusion path)
- [ ] Flip `DRAFTPROOF_V7_DEEP_SCAN=1` only after the above is green
