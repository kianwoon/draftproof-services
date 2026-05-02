# Rewrite Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Rewrite" button on the report page that runs the rewrite pipeline via Celery, stores results in R2, and shows them on a new `/report/:id/rewrite` page.

**Architecture:** POST endpoint creates a RewriteJob row + enqueues Celery task. Worker loads report.json from R2, runs rewrite pipeline, uploads results to R2. Frontend polls status then shows results.

**Tech Stack:** FastAPI, SQLAlchemy, Celery, Cloudflare R2, React, React Router

---

### Task 1: DB — Add RewriteJob model + migration

**Files:**
- Modify: `draftproof-api/app/models/db.py`
- Create: `migrations/versions/xxxx_add_rewrite_jobs.py`

- [ ] **Step 1:** Add RewriteJob model to `db.py` after ScanJob class:

```python
class RewriteJob(Base):
    __tablename__ = "rewrite_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scan_id = Column(UUID(as_uuid=True), ForeignKey("scan_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(Text, nullable=False, default="pending", index=True)
    error = Column(Text)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True))
```

- [ ] **Step 2:** Create migration file. Run:

```bash
cd draftproof-api && alembic revision --autogenerate -m "add_rewrite_jobs"
```

If alembic not set up, create raw SQL migration:

```sql
CREATE TABLE rewrite_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id UUID NOT NULL REFERENCES scan_jobs(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);
CREATE INDEX ix_rewrite_jobs_scan_id ON rewrite_jobs(scan_id);
CREATE INDEX ix_rewrite_jobs_user_id ON rewrite_jobs(user_id);
CREATE INDEX ix_rewrite_jobs_status ON rewrite_jobs(status);
```

- [ ] **Step 3:** Run migration on DB.

- [ ] **Step 4:** Commit

```bash
git add -A && git commit -m "feat: add RewriteJob model and migration"
```

---

### Task 2: Backend — Rewrite service

**Files:**
- Create: `draftproof-api/app/services/rewrite_service.py`
- Modify: `draftproof-api/app/config.py`

- [ ] **Step 1:** Add config constant to `app/config.py`:

```python
REWRITE_TOKEN_COST = 2
```

- [ ] **Step 2:** Create `app/services/rewrite_service.py`:

```python
"""Rewrite service — create rewrite jobs, fetch results from R2."""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from app.config import REWRITE_TOKEN_COST
from app.models.db import async_session, RewriteJob, ScanJob, CreditAccount, CreditReservation

logger = logging.getLogger("rewrite_service")

_STALE_THRESHOLD = timedelta(minutes=15)


async def create_rewrite(scan_id: str, user_id: str) -> dict:
    """Create a rewrite job: validate scan, check balance, deduct tokens, enqueue."""
    uid = uuid.UUID(user_id)
    scan_uuid = uuid.UUID(scan_id)

    async with async_session() as session:
        # 1. Verify scan exists, belongs to user, is completed
        result = await session.execute(
            select(ScanJob).where(
                ScanJob.id == scan_uuid,
                ScanJob.user_id == uid,
                ScanJob.status == "completed",
            )
        )
        scan = result.scalar_one_or_none()
        if not scan:
            raise ValueError("Completed scan not found")

        # 2. Check no rewrite in progress
        existing = await session.execute(
            select(RewriteJob).where(
                RewriteJob.scan_id == scan_uuid,
                RewriteJob.status.in_(["pending", "processing"]),
            )
        )
        if existing.scalar_one_or_none():
            raise ValueError("Rewrite already in progress for this scan")

        # 3. Check for existing completed rewrite
        done = await session.execute(
            select(RewriteJob).where(
                RewriteJob.scan_id == scan_uuid,
                RewriteJob.user_id == uid,
                RewriteJob.status == "completed",
            )
        )
        existing_rewrite = done.scalar_one_or_none()

        # 4. Deduct tokens (with_for_update for atomicity)
        acct_result = await session.execute(
            select(CreditAccount).where(CreditAccount.user_id == uid).with_for_update()
        )
        acct = acct_result.scalar_one_or_none()
        if not acct:
            raise ValueError("No credit account found")
        if acct.balance_tokens - acct.reserved_tokens < REWRITE_TOKEN_COST:
            raise ValueError(f"Insufficient tokens (need {REWRITE_TOKEN_COST})")

        if existing_rewrite:
            # Already have a completed rewrite — return it, no charge
            return _rewrite_to_dict(existing_rewrite)

        acct.reserved_tokens += REWRITE_TOKEN_COST
        job_id = uuid.uuid4()
        rewrite_job = RewriteJob(
            id=job_id,
            scan_id=scan_uuid,
            user_id=uid,
            status="pending",
        )
        session.add(rewrite_job)

        reservation = CreditReservation(
            user_id=uid,
            credit_account_id=acct.id,
            job_type="rewrite",
            job_id=job_id,
            tokens_reserved=REWRITE_TOKEN_COST,
            status="active",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
        )
        session.add(reservation)
        await session.commit()

    # Enqueue Celery task (outside session)
    from app.services.celery_client import run_rewrite
    run_rewrite.delay(str(job_id), scan_id)

    return {"id": str(job_id), "scan_id": scan_id, "status": "pending"}


async def get_rewrite(rewrite_id: str, user_id: str | None = None) -> dict | None:
    """Get rewrite job status. Auto-fails stale jobs."""
    async with async_session() as session:
        q = select(RewriteJob).where(RewriteJob.id == uuid.UUID(rewrite_id))
        if user_id:
            q = q.where(RewriteJob.user_id == uuid.UUID(user_id))
        result = await session.execute(q)
        job = result.scalar_one_or_none()
        if not job:
            return None

        # Stale recovery
        if job.status in ("pending", "processing") and job.created_at:
            age = datetime.now(timezone.utc) - job.created_at
            if age > _STALE_THRESHOLD:
                job.status = "failed"
                job.error = "Rewrite timed out"
                await session.commit()
                await session.refresh(job)

        return _rewrite_to_dict(job)


async def get_rewrite_report(rewrite_id: str, user_id: str) -> dict | None:
    """Fetch rewrite report JSON from R2."""
    job_info = await get_rewrite(rewrite_id, user_id)
    if not job_info or job_info["status"] != "completed":
        return None

    from app.services.report_service import _r2, _fetch_report_json_sync
    if not _r2:
        return None

    scan_id = job_info["scan_id"]
    r2_key = f"reports/{scan_id}/rewrite/rewrite.json"
    try:
        data = await asyncio.to_thread(_fetch_report_json_sync, r2_key)
        return data
    except Exception as e:
        logger.warning("Failed to fetch rewrite JSON from R2: %s", e)
        return None


async def get_rewrite_download_url(rewrite_id: str, fmt: str, user_id: str) -> str | None:
    """Generate presigned download URL for rewrite output."""
    job_info = await get_rewrite(rewrite_id, user_id)
    if not job_info or job_info["status"] != "completed":
        return None

    from app.services.report_service import _r2
    from app.config import R2_BUCKET_NAME
    if not _r2:
        return None

    scan_id = job_info["scan_id"]
    fmt_map = {"pdf": "rewrite.pdf", "md": "rewrite.md", "txt": "rewritten.txt"}
    filename = fmt_map.get(fmt)
    if not filename:
        return None

    key = f"reports/{scan_id}/rewrite/{filename}"
    try:
        url = await asyncio.to_thread(
            _r2.generate_presigned_url,
            "get_object",
            {"Bucket": R2_BUCKET_NAME, "Key": key},
            ExpiresIn=3600,
        )
        return url
    except Exception:
        return None


def _rewrite_to_dict(job: RewriteJob) -> dict:
    return {
        "id": str(job.id),
        "scan_id": str(job.scan_id),
        "status": job.status,
        "error": job.error,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }
```

- [ ] **Step 3:** Add Celery task signature to `app/services/celery_client.py`:

```python
# Add to task_routes:
"app.tasks.run_rewrite": {"queue": "scan"},

# Add after scan_document:
run_rewrite = celery_app.signature("app.tasks.run_rewrite")
```

- [ ] **Step 4:** Commit

```bash
git add -A && git commit -m "feat: add rewrite service with balance deduction and R2 read"
```

---

### Task 3: Backend — API routes

**Files:**
- Modify: `draftproof-api/app/routes/rewrites.py`
- Modify: `draftproof-api/app/models/__init__.py`

- [ ] **Step 1:** Add Pydantic schemas to `app/models/__init__.py`:

```python
class RewriteCreateRequest(BaseModel):
    scan_id: str

class RewriteOut(BaseModel):
    id: str
    scan_id: str
    status: str
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class RewriteReportOut(BaseModel):
    summary: Optional[Any] = None
    sentence_comparison: Optional[list] = None
    ai_findings: Optional[list] = None
```

- [ ] **Step 2:** Replace `app/routes/rewrites.py` entirely:

```python
from fastapi import APIRouter, HTTPException, Depends
from app.models import RewriteCreateRequest, RewriteOut, RewriteReportOut
from app.services import rewrite_service
from app.routes.auth import get_current_user

router = APIRouter()


@router.post("/", response_model=RewriteOut)
async def create_rewrite(req: RewriteCreateRequest, user: dict = Depends(get_current_user)):
    try:
        result = await rewrite_service.create_rewrite(req.scan_id, user["id"])
        return RewriteOut(**result)
    except ValueError as e:
        msg = str(e)
        if "Insufficient" in msg:
            raise HTTPException(status_code=402, detail=msg)
        if "already in progress" in msg:
            raise HTTPException(status_code=409, detail=msg)
        if "not found" in msg:
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)


@router.get("/{rewrite_id}", response_model=RewriteOut)
async def get_rewrite(rewrite_id: str, user: dict = Depends(get_current_user)):
    result = await rewrite_service.get_rewrite(rewrite_id, user_id=user["id"])
    if not result:
        raise HTTPException(status_code=404, detail="Rewrite not found")
    return RewriteOut(**result)


@router.get("/{rewrite_id}/report", response_model=RewriteReportOut)
async def get_rewrite_report(rewrite_id: str, user: dict = Depends(get_current_user)):
    data = await rewrite_service.get_rewrite_report(rewrite_id, user["id"])
    if not data:
        raise HTTPException(status_code=404, detail="Rewrite report not found")
    return RewriteReportOut(**data)


@router.get("/{rewrite_id}/download/{fmt}")
async def download_rewrite(rewrite_id: str, fmt: str, user: dict = Depends(get_current_user)):
    if fmt not in ("pdf", "md", "txt"):
        raise HTTPException(status_code=400, detail="Format must be pdf, md, or txt")
    url = await rewrite_service.get_rewrite_download_url(rewrite_id, fmt, user["id"])
    if not url:
        raise HTTPException(status_code=404, detail="Download not available")
    return {"url": url}
```

- [ ] **Step 3:** Commit

```bash
git add -A && git commit -m "feat: add rewrite API routes (create, status, report, download)"
```

---

### Task 4: Worker — Celery rewrite task + R2 upload

**Files:**
- Modify: `worker/app/tasks.py`
- Modify: `worker/app/storage.py`
- Modify: `worker/app/db.py`

- [ ] **Step 1:** Add rewrite upload helper to `worker/app/storage.py`:

```python
def upload_rewrite_files(scan_id: str, md_text: str, pdf_bytes: bytes, json_data: dict, rewritten_text: str) -> Dict[str, str]:
    """Upload rewrite results to R2 under reports/{scan_id}/rewrite/."""
    s3 = _client()
    bucket = settings.R2_BUCKET_NAME
    prefix = f"reports/{scan_id}/rewrite"
    urls = {}

    uploads = [
        ("rewrite.json", json.dumps(json_data, indent=2, ensure_ascii=False).encode(), "application/json"),
        ("rewrite.md", md_text.encode(), "text/markdown"),
        ("rewrite.pdf", pdf_bytes, "application/pdf"),
        ("rewritten.txt", rewritten_text.encode("utf-8"), "text/plain"),
    ]
    for filename, data, content_type in uploads:
        key = f"{prefix}/{filename}"
        s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
        urls[filename] = _presign(s3, bucket, key)

    return urls
```

- [ ] **Step 2:** Add rewrite DB helpers to `worker/app/db.py`:

```python
def get_rewrite_job(job_id: str) -> Optional[dict]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM rewrite_jobs WHERE id = %s", (job_id,))
        return cur.fetchone()

def update_rewrite_status(job_id: str, status: str, error: str = None):
    sets = ["status = %s"]
    vals = [status]
    if status == "completed":
        sets.append("completed_at = now()")
    if error:
        sets.append("error = %s")
        vals.append(error)
    with get_conn() as conn:
        conn.cursor().execute(
            f"UPDATE rewrite_jobs SET {', '.join(sets)} WHERE id = %s",
            vals + [job_id],
        )

def capture_rewrite_credits(user_id: str, job_id: str):
    """Capture credit reservation for a rewrite job."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT id, credit_account_id, tokens_reserved
               FROM credit_reservations
               WHERE job_id = %s AND status = 'active'
               LIMIT 1""",
            (job_id,),
        )
        row = cur.fetchone()
        if not row:
            return
        res_id = row["id"]
        acct_id = row["credit_account_id"]
        tokens = row["tokens_reserved"]

        cur.execute("UPDATE credit_reservations SET status = 'captured' WHERE id = %s", (res_id,))
        cur.execute(
            "UPDATE credit_accounts SET balance_tokens = balance_tokens - %s, reserved_tokens = reserved_tokens - %s WHERE id = %s",
            (tokens, tokens, acct_id),
        )
        cur.execute(
            """INSERT INTO usage_events
               (user_id, credit_account_id, event_type, tokens_charged, job_id)
               VALUES (%s, %s, 'rewrite', %s, %s)""",
            (user_id, acct_id, tokens, job_id),
        )
```

- [ ] **Step 3:** Add `run_rewrite` task to `worker/app/tasks.py`:

```python
@app.task(bind=True, max_retries=2, default_retry_delay=30, soft_time_limit=600, time_limit=660)
def run_rewrite(self, rewrite_id: str, scan_id: str) -> dict:
    """Run the rewrite pipeline on a completed scan's results."""
    from .storage import upload_rewrite_files
    from .db import get_rewrite_job, update_rewrite_status, capture_rewrite_credits, get_scan_job
    import tempfile

    try:
        update_rewrite_status(rewrite_id, "processing")

        # 1. Fetch report.json from R2
        scan_job = get_scan_job(scan_id)
        report_json = _fetch_report_from_r2(scan_id)

        if not report_json:
            update_rewrite_status(rewrite_id, "failed", error="Original report not found in R2")
            return {"status": "failed", "error": "report not found"}

        # 2. Check if there are AI findings
        findings = report_json.get("findings", [])
        ai_findings = [f for f in findings if f.get("category") == "ai_generation" or f.get("scanner") == "ai_generation"]
        if not ai_findings:
            update_rewrite_status(rewrite_id, "failed", error="No AI findings to rewrite")
            return {"status": "failed", "error": "no AI findings"}

        # 3. Run rewrite pipeline
        from poc.rewrite_pipeline import run_rewrite_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_rewrite_pipeline(
                detect_json=report_json,
                output_dir=tmpdir,
                max_passes=3,
                ai_only=True,
                verbose=False,
            )

        if result["status"] == "skipped" or result["status"] == "clean":
            update_rewrite_status(rewrite_id, "failed", error=result.get("message", "Rewrite not needed"))
            return {"status": "skipped"}

        # 4. Upload results to R2
        rw = result["result"]
        summary = rw.summary if hasattr(rw, "summary") else {}

        # Read generated files
        import os
        md_path = result.get("md_path")
        pdf_path = result.get("pdf_path")

        md_text = ""
        pdf_bytes = b""
        if md_path and os.path.exists(md_path):
            with open(md_path) as f:
                md_text = f.read()
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

        rewritten_text = ""
        if hasattr(rw, "mp_result") and rw.mp_result:
            rewritten_text = rw.mp_result.final_text or ""

        # Build rewrite JSON output
        rewrite_json = _build_rewrite_json(result, summary)

        upload_rewrite_files(scan_id, md_text, pdf_bytes, rewrite_json, rewritten_text)

        # 5. Capture credits
        user_id = scan_job.get("user_id", "") if scan_job else ""
        if user_id:
            capture_rewrite_credits(str(user_id), rewrite_id)

        update_rewrite_status(rewrite_id, "completed")
        return {"status": "completed"}

    except SoftTimeLimitExceeded:
        update_rewrite_status(rewrite_id, "failed", error="Rewrite timed out (10 min limit)")
        return {"status": "failed", "error": "timeout"}
    except Exception as e:
        if self.request.retries < self.max_retries:
            update_rewrite_status(rewrite_id, "retrying", error=str(e))
            raise self.retry(exc=e)
        else:
            update_rewrite_status(rewrite_id, "failed", error=str(e))
            raise


def _fetch_report_from_r2(scan_id: str) -> dict | None:
    """Fetch report.json from R2 for a scan."""
    from .storage import _client
    from .config import settings
    try:
        s3 = _client()
        resp = s3.get_object(Bucket=settings.R2_BUCKET_NAME, Key=f"reports/{scan_id}/report.json")
        return json.loads(resp["Body"].read())
    except Exception:
        return None


def _build_rewrite_json(result: dict, summary: dict) -> dict:
    """Build the rewrite.json output stored in R2."""
    rw = result.get("result")
    output = {
        "status": result.get("status"),
        "elapsed": result.get("elapsed"),
        "summary": summary,
    }
    if rw and hasattr(rw, "mp_result") and rw.mp_result:
        mp = rw.mp_result
        output["original_text"] = mp.original_text
        output["final_text"] = mp.final_text
        output["passes"] = len(mp.passes)
        output["converged"] = mp.converged
        output["convergence_reason"] = mp.convergence_reason
    if rw and hasattr(rw, "sentence_comparison"):
        output["sentence_comparison"] = rw.sentence_comparison
    if rw and hasattr(rw, "summary"):
        output["rewrite_summary"] = rw.summary
    return output
```

- [ ] **Step 4:** Commit

```bash
git add -A && git commit -m "feat: add Celery rewrite task with R2 upload"
```

---

### Task 5: Frontend — API client + routes

**Files:**
- Modify: `draftproof-frontend/src/api/draftproofApi.js`
- Modify: `draftproof-frontend/src/App.jsx`

- [ ] **Step 1:** Replace rewrite section in `draftproofApi.js`:

```javascript
// Rewrites
export const createRewrite = (scanId) => api.post('/rewrites/', { scan_id: scanId });
export const getRewriteStatus = (rewriteId) => api.get(`/rewrites/${rewriteId}`);
export const getRewriteReport = (rewriteId) => api.get(`/rewrites/${rewriteId}/report`);
export const getRewriteDownload = (rewriteId, format) => api.get(`/rewrites/${rewriteId}/download/${format}`);
```

- [ ] **Step 2:** Add route to `App.jsx`. Add import:

```javascript
import Rewrite from './pages/Rewrite';
```

Add route (after the report/:id route):

```jsx
<Route path="/report/:id/rewrite" element={<ProtectedRoute><Rewrite /></ProtectedRoute>} />
```

- [ ] **Step 3:** Commit

```bash
git add -A && git commit -m "feat: add rewrite API client and route"
```

---

### Task 6: Frontend — Rewrite button on Report page

**Files:**
- Modify: `draftproof-frontend/src/pages/Report.jsx`

- [ ] **Step 1:** Add "Rewrite" button in the report hero section, near the tier badge. Import `createRewrite` from the API. Add state for rewrite loading.

Key logic:
- Button visible only when `status === 'completed'`
- On click: call `createRewrite(scanId)`, get back rewrite job ID, navigate to `/report/${id}/rewrite`
- Handle 402 (insufficient tokens), 409 (already in progress) with user-friendly messages
- If rewrite already exists (check via scan metadata or API), show "View Rewrite" instead

- [ ] **Step 2:** Commit

```bash
git add -A && git commit -m "feat: add Rewrite button to Report page"
```

---

### Task 7: Frontend — Rewrite results page

**Files:**
- Create: `draftproof-frontend/src/pages/Rewrite.jsx`

- [ ] **Step 1:** Create the Rewrite page with these sections:

```jsx
// State: rewriteStatus, report, loading, error
// On mount: get rewrite ID from URL params or create one
// Poll getRewriteStatus every 3s while pending/processing
// When completed: fetch getRewriteReport
//
// Layout:
// 1. Header with document name, back link to report
// 2. If processing: spinner with "Rewriting your document..."
// 3. If completed:
//    - Outcome badge (Converged / Partially Improved / Floor Reached)
//    - Before/After risk score table
//    - Side-by-side original vs rewritten text
//    - Download buttons (PDF, MD, TXT)
// 4. If failed: error message with retry button
```

- [ ] **Step 2:** Commit

```bash
git add -A && git commit -m "feat: add Rewrite results page"
```

---

### Task 8: Integration test + deploy

**Files:**
- None (manual testing)

- [ ] **Step 1:** Run DB migration locally or on Koyeb
- [ ] **Step 2:** Test full flow: scan → view report → click rewrite → wait → view rewrite results
- [ ] **Step 3:** Push and verify Koyeb deployment
- [ ] **Step 4:** Commit any fixes
