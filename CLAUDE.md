# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

DraftProof is an AI writing integrity detection and rewrite platform. Users submit documents, receive a tier classification (the AI-risk badge tier — `green`/`amber`/`orange`/`red`) with detailed findings, and can request rewrites that preserve academic intent while removing AI-generated passages.

## Objective & Rewrite Philosophy (read before touching the rewrite/guards)

**The objective is to MITIGATE AI risk for the user.** The rewritten content DraftProof produces is **a shown "solution" / reviewable draft** — it demonstrates *how* to mitigate the AI risk. It is **NOT a final submission**. The **user is responsible** to review the before/after diff, learn from it, and **edit with their own real content**.

Consequences that must shape the rewrite pipeline and its guards:
- **Most AI risk comes from CONTENT LACKING** (generic, unanchored, ungrounded claims — the dominant detector signals are `generic_assertion_risk` and `citation_grounding_risk`, not word choice). You **cannot** mitigate by reshuffling/synonym-swapping the submitted words; that stays generic.
- **Filling the gap by adding content IS the solution.** The writer/author-proxy SHOULD bump in concrete anchors, scenarios, examples, and specifics. The user "doesn't know what they don't know," so DraftProof must surface the missing grounding. Added/illustrative content is reviewed and replaced by the user (the frontend shows a highlighted before/after diff at `/rewrite`).
- **Guards must ANNOTATE, never SUPPRESS.** Rejecting a flagged paragraph → `source_preserved` ("no change") is the **worst** outcome: the user sees no solution and learns nothing. Guards that block *adding content, rephrasing, or restructuring* (e.g. `unsupported_semantic_padding`, term-exact `required_source_terms_missing`, "fabricated specifics", `sentence_starts_with_conjunction`) **conflict with the objective** — demote them to advisory review-flags. Only fall back to the original when there is genuinely **no usable rewrite** (empty/stub/broken grammar).
- **Keep only meaning-fidelity + readability guards**, and as review-flags where possible: don't *distort the user's actual argument* (`source_polarity_inversion`, dropped original ideas) and don't ship broken grammar. Everything else rides along as a flag for the user to verify.
- **Lean direct path** (`poc/rewrite_v6/direct_rewrite.py`) is the **DEFAULT** rewrite path: scanner diagnosis → one simple writer call per flagged paragraph → always show the solution + review flags. Kill switch: set `DRAFTPROOF_V6_DIRECT_REWRITE=0` to fall back to the legacy planner/selector pipeline. It beats the heavy pipeline (~38 vs ~52 final_risk) because the elaborate planner→writer machinery buried the simple fix and the guards blocked the content-filling solution.
- **Measure with the deterministic harness** (`DRAFTPROOF_V6_DETERMINISTIC=1 python poc/_measure_baseline.py N`, N≥4) — single runs are noise (the gpt-oss writer is high-variance).

Four deployable components:
- **draftproof-api** — FastAPI backend (production)
- **draftproof-frontend** — React + Vite SPA
- **worker** — Celery async task processor (scan & rewrite jobs)
- **poc** — AI detection & rewrite pipeline library (core logic imported by worker)

---

## Commands

### Frontend
```bash
cd draftproof-frontend
npm install
npm run dev          # Vite dev server on port 3000, proxies /api → port 8000
npm run build        # Output to dist/
```

### API
```bash
cd draftproof-api
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000   # dev
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 650  # prod
```

### Worker
```bash
cd worker
pip install -r requirements.txt
celery -A app.celery_app worker -l info -c 1 --timeout 600
```

### Tests
```bash
cd draftproof-api && pytest              # API tests
cd worker && pytest                      # worker tests
cd poc && python test_rewrite_v2.py     # POC integration tests (run individually)
```

**Before shipping ANY change to the detection scoring (`poc/detect/`), run the ESL false-positive gate** — it scores the SCoCESLE corpus by proficiency and FAILS (exit 1) if ESL FPR rises, AUC drops, or the higher-vs-lower parity gap widens vs the committed baseline. The corpus is local-only (no redistribution), so this is a local gate, not CI. ~13 min for the full 272 essays.
```bash
cd poc && python calibration/fpr_subgroup_gate.py --compare   # GATE vs poc/calibration/fpr_subgroup_baseline.json
cd poc && python calibration/fpr_subgroup_gate.py --limit 12  # quick smoke (~1 min)
```
This gate is **auto-enforced** by the `pre-push` hook in `maintenances/githooks/` (active via
`core.hooksPath`; set it once on a fresh clone with `git config core.hooksPath maintenances/githooks`).
The hook runs the gate **only when a push touches `poc/detect/`**, blocks on a regression, and skips
gracefully if the corpus / ML stack is absent. Bypass an intentional, re-baselined change with
`git push --no-verify`.

### Database Migrations
```bash
cd draftproof-api
alembic upgrade head                    # apply all migrations
alembic revision --autogenerate -m "description"  # generate new migration
```

### Health Check
```bash
curl http://localhost:8000/api/health   # returns {"status":"ok","db":"up"} or 503
```

---

## Architecture

### Request → Task → Stream Flow

1. Frontend submits text/document → `POST /api/documents/text` or `/upload`
2. Frontend calls `POST /api/scans/` or `POST /api/rewrites/` → API creates DB job record, enqueues Celery task, returns job ID immediately
3. Frontend opens `EventSource` to `/api/scans/{id}/events` or `/api/rewrites/{id}/events`
4. Worker processes job (see below), publishes progress to a **Redis Stream**
5. API reads Redis stream and forwards as SSE. Client sends `last-event-id` header for resumption.
6. On completion, worker uploads report JSON to **Cloudflare R2**; frontend fetches via `/api/reports/{id}`

### Scan Pipeline (`worker/app/tasks.py` → `poc/detect/`)

`scan_document` Celery task calls `poc.detect.run.DetectionRunner`:
- Scores AI generation probability and writing quality per paragraph
- `layer3_scoring.py` assigns final tier
- `postprocess.py` applies mitigation filtering
- `repair_units.py` + `rewrite_targets.py` group content into rewritable units
- Outputs findings array → `poc/report/report.py` builds report JSON → uploaded to R2

### Rewrite Pipeline (`worker/app/tasks.py` → `poc/rewrite_v6/`)

`run_rewrite` Celery task calls `poc.rewrite_v6.production.run_rewrite_pipeline_v6` (there is **no**
`RewritePipeline` class — `pipeline.py` holds the legacy planner functions, not the entry point).
That function branches on two independent, stacked kill switches:

1. `DRAFTPROOF_REWRITE_V6_ENABLED` (default `True`) — v6 vs. `poc.rewrite_pipeline.run_rewrite_pipeline`,
   a **separate, pre-v6 standalone codebase** with no evidence of live production or CI use.
2. Inside v6, `DRAFTPROOF_V6_DIRECT_REWRITE` (default `"1"`) — **the direct path is the actual
   default and the objective-aligned winner** (see `poc/rewrite_v6/direct_rewrite.py`'s module
   docstring): one writer call per flagged paragraph, hard-rejects only on broken grammar or an
   unusable/stub result, everything else ships as a review flag. Set to `0` to fall back to the
   internal legacy planner pipeline below — also untested in production.

Both non-default branches log a loud warning when selected (2026-07-10 risk review) — flipping either
during an incident is not a validated fallback, just an untested code path.

**Internal legacy planner pipeline** (only reached with `DRAFTPROOF_V6_DIRECT_REWRITE=0`):
- **plan.py / planner_llm.py** — identify problematic paragraphs, plan per-paragraph rewrites
- **write.py** — multi-variant generation per paragraph via LLM
- **writer_feedback.py** — feedback loop on variants
- **selector_diagnostics.py** — pick best variant
- **integrity_guard.py / coverage_guard.py** — safety checks (no content loss); its `ADVISORY_BLOCKERS`
  frozenset is FATAL-heavy in a way that no longer matches the direct path's "annotate, don't suppress"
  philosophy — moot only because this pipeline isn't the default
- **author_proxy.py** — preserves author intent
- Final comparison report uploaded to R2; credits deducted via `worker/app/db.py`

`poc/rewrite_v3/`, `v4/`, `v5/`, and `poc/rewrite/rewrite.py` (an unversioned pre-v6 standalone
orchestrator) are all legacy and not reachable from the production worker. Use `rewrite_v6/`.

### LLM Routing (`poc/rewrite_v6/llm_config.py`, `poc/llm/`)

V6 uses separate planner/writer/selector models with fallback chains:
- Providers: OpenRouter (OpenAI-compatible), Cerebras, Groq, AWS Bedrock, Google Vertex
- Model selection driven by env vars: `DRAFTPROOF_PLANNER_MODEL`, `DRAFTPROOF_GENERATOR_MODEL`, etc.
- Optional Tavily API for source grounding in rewrites

### Credit & Payment Flow

`POST /api/payments/create-checkout-session` → Stripe checkout → Stripe webhook → `credit_ledger` entry → `credit_accounts.balance_tokens` increases. Scans and rewrites reserve then deduct tokens; `credit_reservations` prevents over-run.

---

## Key Files

| File | Purpose |
|------|---------|
| `draftproof-api/app/main.py` | FastAPI app setup, middleware, SPA fallback routing |
| `draftproof-api/app/config.py` | All env config (DB, JWT, Stripe, R2, OAuth) |
| `draftproof-api/app/models/db.py` | SQLAlchemy models (User, ScanJob, RewriteJob, CreditAccount, etc.) |
| `worker/app/tasks.py` | `scan_document` and `run_rewrite` Celery task entrypoints |
| `poc/detect/run.py` | `DetectionRunner` — main detection orchestrator |
| `poc/rewrite_v6/production.py` | `run_rewrite_pipeline_v6` — real rewrite entry point |
| `poc/rewrite_v6/direct_rewrite.py` | Default rewrite path (`run_direct_rewrite_all`) |
| `worker/app/db.py` | DB ops called from tasks (status updates, credit capture) |
| `worker/app/progress.py` | Publish progress events to Redis streams |

---

## Database Schema (Alembic, `draftproof-api/migrations/`)

**Auth/Credits**: `users`, `user_identities` (Google/Microsoft OAuth), `credit_accounts`, `credit_ledger`, `credit_reservations`, `payments`

**Jobs**: `scan_jobs` (status, tier, ai_score, report_urls JSON), `rewrite_jobs` (status, scan_id FK), `usage_events`, `pricing_plans`

`scan_jobs.tier` values: `green` / `amber` / `orange` / `red` — this is the **AI-risk badge tier**
(`poc/report/builder.py` `ai_risk_badge["tier"]`, produced by the Layer3 / DeBERTa / V7-fused
scoring paths and `.lower()`ed to lowercase). It is what the report PDF/MD shows and what the
frontend colors (`reportHelpers.js` `DRAFTPROOF_TIER_COLORS = {GREEN, AMBER, ORANGE, RED}`).
Do NOT confuse it with two other tier vocabularies in the codebase:
- `critical` / `high` / `medium` / `low` / `clean` — `poc/report/models.py` `Tier`, the
  findings-based `overall_tier` (drives the findings list, not the headline badge).
- `clean` / `acceptable` / `concerning` / `strong` — the V7 **detection-result** tier
  (Turnitin-style; consumed by the V7 guard config and `test_detect_v7_pipeline_bridge.py`),
  a separate surface that is not what `scan_jobs.tier` stores.
`scan_jobs.status` / `rewrite_jobs.status` values: `pending` / `processing` / `completed` / `failed` (+ `canceled` for rewrites)

---

## Environment Variables

See `.env.example`. Critical ones:

| Var | Where used |
|-----|-----------|
| `DATABASE_URL` | asyncpg PostgreSQL URL |
| `SECRET_KEY` | JWT signing |
| `REDIS_URL` | Celery broker + result backend + progress streams |
| `R2_ENDPOINT_URL`, `R2_BUCKET_NAME`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` | Report storage |
| `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET` | Payments |
| `GOOGLE_CLIENT_ID/SECRET`, `MICROSOFT_CLIENT_ID/SECRET` | OAuth |
| `DRAFTPROOF_PLANNER_MODEL`, `DRAFTPROOF_GENERATOR_MODEL` | LLM model selection |
| `DRAFTPROOF_REWRITE_V6_ENABLED=1` | Enable v6 rewrite pipeline |
| `TAVILY_API_KEY` | Optional web search grounding |

---

## Deployment

Root `Dockerfile` is multi-stage: builds frontend (node:20), then installs Python deps and copies built frontend to `/static`. The API serves the SPA via a fallback route in `main.py`.

Worker deploys separately using `worker/Dockerfile` + `entrypoint.sh`.

Pushing to `main` triggers Koyeb auto-deploy (no manual redeploy needed).
