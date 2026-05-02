# Rewrite Feature Design

## Overview

Add a "Rewrite" button to the report page that lets users rewrite AI-flagged findings in their document. The rewrite pipeline uses a third-party AI key (DraftProof-owned) to produce rewritten text and a detailed report. Results are stored in R2 alongside the original scan report.

## Decisions

| Decision | Choice |
|----------|--------|
| Credits | 2 tokens flat rate per rewrite |
| Output | Rewritten text + rewrite report (PDF + MD + JSON) |
| UI | New `/report/:id/rewrite` page |
| Scope | AI-generation findings only |
| AI Key | Third-party key owned by DraftProof (env var on worker) |

## Architecture

```
User clicks "Rewrite" on Report page
       |
POST /api/rewrites  (deduct 2 tokens, create rewrite job)
       |
Celery worker loads report.json from R2
       |
Runs poc/rewrite_pipeline.py (ai_only=True)
       |
Uploads to R2: reports/{scan_id}/rewrite/
  ├── rewrite.json    (summary + sentence comparison)
  ├── rewrite.md      (markdown report)
  ├── rewrite.pdf     (PDF report)
  └── rewritten.txt   (clean rewritten text)
       |
Frontend polls GET /api/rewrites/{id} -> shows rewrite page when done
```

## Backend

### DB: RewriteJob table

New table in `app/models/db.py`:

```python
class RewriteJob(Base):
    __tablename__ = "rewrite_jobs"

    id          = Column(UUID, primary_key=True, default=uuid4)
    scan_id     = Column(UUID, ForeignKey("scan_jobs.id"), nullable=False, index=True)
    user_id     = Column(UUID, ForeignKey("users.id"), nullable=False, index=True)
    status      = Column(String(20), nullable=False, default="pending")  # pending/processing/completed/failed
    error       = Column(Text, nullable=True)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
```

### API routes (`app/routes/rewrites.py`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | POST | Create rewrite job (check balance, deduct 2 tokens, enqueue Celery) |
| `/{rewrite_id}` | GET | Poll rewrite status |
| `/{rewrite_id}/report` | GET | Fetch rewrite report data from R2 |
| `/{rewrite_id}/download/{format}` | GET | Presigned URL for PDF/MD/TXT download |

**POST / behavior:**
1. Accept `scan_id` in request body
2. Verify scan exists, belongs to user, status=completed
3. Check no rewrite already in-progress for this scan
4. Check user has >= 2 tokens, deduct atomically (with_for_update)
5. Create RewriteJob row (status=pending)
6. Enqueue Celery `run_rewrite` task with (rewrite_id, scan_id)
7. Return rewrite job info

### Celery task (`worker/app/tasks.py`)

New `run_rewrite(rewrite_id, scan_id)` task:
1. Mark RewriteJob as processing
2. Fetch `reports/{scan_id}/report.json` from R2
3. Run `poc/rewrite_pipeline.py` via `run_rewrite_pipeline(detect_json=data, ai_only=True)`
4. Upload results to R2 at `reports/{scan_id}/rewrite/`
5. Mark RewriteJob as completed (or failed with error message)

### Rewrite service (`app/services/rewrite_service.py`)

```python
async def create_rewrite(scan_id: str, user_id: str) -> dict
async def get_rewrite(rewrite_id: str, user_id: str) -> dict | None
async def get_rewrite_report(rewrite_id: str, user_id: str) -> dict | None
async def get_rewrite_download(rewrite_id: str, fmt: str, user_id: str) -> str | None
```

### Config (`app/config.py`)

- `REWRITE_TOKEN_COST = 2` — flat token cost per rewrite

## Frontend

### Report.jsx changes

Add "Rewrite" button near the tier badge in the report header:
- Disabled if no AI findings exist
- Shows loading state while rewrite is processing
- Navigates to `/report/:id/rewrite` on click
- If rewrite already exists for this scan, show "View Rewrite" instead

### New Rewrite.jsx page (`/report/:id/rewrite`)

Layout:
1. **Header** — document name, back link to report
2. **Progress** — spinner while processing (polls every 3s)
3. **When complete:**
   - Before/After risk score comparison table
   - Outcome badge (Converged / Partially Improved / Floor Reached)
   - Side-by-side original vs rewritten text
   - Sentence-level changes table (which sentences changed, tier changes)
   - Download links for PDF / MD / TXT

### API client (`draftproofApi.js`)

```javascript
export const createRewrite = (scanId) => api.post('/rewrites', { scan_id: scanId });
export const getRewriteStatus = (rewriteId) => api.get(`/rewrites/${rewriteId}`);
export const getRewriteReport = (rewriteId) => api.get(`/rewrites/${rewriteId}/report`);
export const getRewriteDownload = (rewriteId, format) => api.get(`/rewrites/${rewriteId}/download/${format}`);
```

## R2 Storage

```
reports/{scan_id}/
├── report.json          # existing detect results
├── report.md            # existing
├── report.pdf           # existing
└── rewrite/             # NEW
    ├── rewrite.json     # rewrite summary + sentence comparison
    ├── rewrite.md       # rewrite markdown report
    ├── rewrite.pdf      # rewrite PDF report
    └── rewritten.txt    # clean rewritten text
```

## Edge Cases

| Case | Behavior |
|------|----------|
| Scan not completed | 400 error: "Scan must be completed before rewriting" |
| No AI findings in scan | 400 error: "No AI findings to rewrite" |
| Rewrite already in progress | 409 conflict: "Rewrite already in progress for this scan" |
| Rewrite already completed | Return existing rewrite (don't re-run) |
| Insufficient tokens | 402 error: "Insufficient tokens (need 2)" |
| Rewrite fails (LLM error) | Mark as failed, show error message to user |
| Stale rewrite (>10 min pending) | Auto-mark as failed on next poll |

## Files to Create/Modify

### New files
- `draftproof-api/app/services/rewrite_service.py`
- `draftproof-frontend/src/pages/Rewrite.jsx`

### Modified files
- `draftproof-api/app/models/db.py` — add RewriteJob table
- `draftproof-api/app/routes/rewrites.py` — replace TODO stubs
- `draftproof-api/app/models/__init__.py` — add rewrite schemas
- `draftproof-api/app/main.py` — mount rewrite router
- `worker/app/tasks.py` — add run_rewrite task
- `worker/app/storage.py` — add rewrite R2 helpers if needed
- `draftproof-frontend/src/pages/Report.jsx` — add Rewrite button
- `draftproof-frontend/src/App.jsx` — add /report/:id/rewrite route
- `draftproof-frontend/src/api/draftproofApi.js` — add rewrite API calls
