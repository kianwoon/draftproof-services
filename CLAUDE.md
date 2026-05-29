# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

DraftProof is an AI writing integrity detection and rewrite platform. Users submit documents, receive a tier classification (clean/acceptable/concerning/strong) with detailed findings, and can request rewrites that preserve academic intent while removing AI-generated passages.

## Objective & Rewrite Philosophy (read before touching the rewrite/guards)

**The objective is to MITIGATE AI risk for the user.** The rewritten content DraftProof produces is **a shown "solution" / reviewable draft** — it demonstrates *how* to mitigate the AI risk. It is **NOT a final submission**. The **user is responsible** to review the before/after diff, learn from it, and **edit with their own real content**.

Consequences that must shape the rewrite pipeline and its guards:
- **Most AI risk comes from CONTENT LACKING** (generic, unanchored, ungrounded claims — the dominant detector signals are `generic_assertion_risk` and `citation_grounding_risk`, not word choice). You **cannot** mitigate by reshuffling/synonym-swapping the submitted words; that stays generic.
- **Filling the gap by adding content IS the solution.** The writer/author-proxy SHOULD bump in concrete anchors, scenarios, examples, and specifics. The user "doesn't know what they don't know," so DraftProof must surface the missing grounding. Added/illustrative content is reviewed and replaced by the user (the frontend shows a highlighted before/after diff at `/rewrite`).
- **Guards must ANNOTATE, never SUPPRESS.** Rejecting a flagged paragraph → `source_preserved` ("no change") is the **worst** outcome: the user sees no solution and learns nothing. Guards that block *adding content, rephrasing, or restructuring* (e.g. `unsupported_semantic_padding`, term-exact `required_source_terms_missing`, "fabricated specifics", `sentence_starts_with_conjunction`) **conflict with the objective** — demote them to advisory review-flags. Only fall back to the original when there is genuinely **no usable rewrite** (empty/stub/broken grammar).
- **Keep only meaning-fidelity + readability guards**, and as review-flags where possible: don't *distort the user's actual argument* (`source_polarity_inversion`, dropped original ideas) and don't ship broken grammar. Everything else rides along as a flag for the user to verify.
- **Lean direct path** (`poc/rewrite_v6/direct_rewrite.py`, flag `DRAFTPROOF_V6_DIRECT_REWRITE`): scanner diagnosis → one simple writer call per flagged paragraph → always show the solution + review flags. This beats the heavy planner/selector pipeline (~38 vs ~52 final_risk) because the elaborate planner→writer machinery buried the simple fix and the guards blocked the content-filling solution.
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

`run_rewrite` Celery task calls `poc.rewrite_v6.pipeline.RewritePipeline`:
- **plan.py / planner_llm.py** — identify problematic paragraphs, plan per-paragraph rewrites
- **write.py** — multi-variant generation per paragraph via LLM
- **writer_feedback.py** — feedback loop on variants
- **selector_diagnostics.py** — pick best variant
- **integrity_guard.py / coverage_guard.py** — safety checks (no content loss)
- **author_proxy.py** — preserves author intent
- Final comparison report uploaded to R2; credits deducted via `worker/app/db.py`

`poc/rewrite_v3/`, `v4/`, `v5/` are legacy and not used in production. Use `rewrite_v6/`.

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
| `poc/rewrite_v6/pipeline.py` | `RewritePipeline` — main rewrite orchestrator |
| `worker/app/db.py` | DB ops called from tasks (status updates, credit capture) |
| `worker/app/progress.py` | Publish progress events to Redis streams |

---

## Database Schema (Alembic, `draftproof-api/migrations/`)

**Auth/Credits**: `users`, `user_identities` (Google/Microsoft OAuth), `credit_accounts`, `credit_ledger`, `credit_reservations`, `payments`

**Jobs**: `scan_jobs` (status, tier, ai_score, report_urls JSON), `rewrite_jobs` (status, scan_id FK), `usage_events`, `pricing_plans`

`scan_jobs.tier` values: `clean` / `acceptable` / `concerning` / `strong`
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
