import asyncio
import json
import logging
import re

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from app.config import SSE_SCAN_STREAM_MAX_SECONDS
from app.models import ScanRequest, ScanOut
from app.services.scan_service import create_scan, get_scan, list_scans, delete_scan
from app.services import progress_stream
from app.routes.auth import get_current_user

router = APIRouter()

_REDIS_STREAM_ID_RE = re.compile(r"^\d+-\d+$")


@router.get("/")
async def list_scans_route(
    user: dict = Depends(get_current_user),
    page: int = 1,
    per_page: int = 10,
):
    return await list_scans(user["id"], page=page, per_page=per_page)


@router.post("/", response_model=ScanOut)
async def create_scan_route(req: ScanRequest, user: dict = Depends(get_current_user)):
    try:
        result = await create_scan(
            req.document_id, user_id=user["id"], text=req.text, ai_policy=req.ai_policy,
        )
        return ScanOut(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        # Log server-side; never leak the raw exception text to the client (L11).
        logging.getLogger(__name__).exception("Scan creation failed")
        raise HTTPException(status_code=500, detail="Scan could not be started. Please try again.")


@router.get("/{scan_id}/events")
async def stream_scan_events(
    scan_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
):
    async def event_stream():
        redis_unavailable = False
        last_stream_id = request.headers.get("last-event-id") or "$"
        if last_stream_id != "$" and not _REDIS_STREAM_ID_RE.match(last_stream_id):
            last_stream_id = "$"

        # Send initial state from DB
        result = await get_scan(scan_id, user_id=user["id"])
        if not result:
            yield "event: scan-error\ndata: {\"detail\":\"Scan not found\"}\n\n"
            return

        current_payload = {
            "id": result["id"],
            "status": result["status"],
            "report_id": result["report_id"],
            "progress_percent": result["progress_percent"],
            "progress_message": result["progress_message"],
        }
        data = json.dumps(current_payload)
        yield f"event: progress\ndata: {data}\n\n"
        last_payload = data

        if result["status"] in ("completed", "failed"):
            return

        stream_started = asyncio.get_running_loop().time()
        last_db_check = stream_started

        while True:
            if await request.is_disconnected():
                break

            # Absolute lifetime cap (SSE_SCAN_STREAM_MAX_SECONDS): close cleanly, the
            # same way the terminal-status path below does, so the client's polling
            # fallback (GET /scans/{id}) takes over instead of the connection hanging
            # open indefinitely. This is a backstop on top of disconnect detection and
            # the stale-job reaper (scan_service._processing_scan_is_stale), not a
            # replacement for either.
            if asyncio.get_running_loop().time() - stream_started > SSE_SCAN_STREAM_MAX_SECONDS:
                logging.getLogger(__name__).info(
                    "SSE scan stream %s closed at lifetime cap (%ss)", scan_id, SSE_SCAN_STREAM_MAX_SECONDS
                )
                break

            # Try Redis stream first (fast, no DB hit)
            if not redis_unavailable:
                events = await progress_stream.read_scan_progress(
                    scan_id,
                    last_stream_id,
                    count=10,
                )
                if events is None:
                    redis_unavailable = True
                else:
                    for event_id, fields in events:
                        last_stream_id = event_id
                        payload = dict(current_payload)
                        if fields.get("status"):
                            payload["status"] = fields["status"]
                        if "progress_percent" in fields:
                            try:
                                payload["progress_percent"] = max(0, min(100, int(fields["progress_percent"])))
                            except (ValueError, TypeError):
                                pass
                        if "progress_message" in fields:
                            payload["progress_message"] = fields["progress_message"]
                        if "error" in fields:
                            payload["error"] = fields.get("error") or None
                        current_payload = payload
                        data = json.dumps(payload)
                        if data != last_payload:
                            yield f"event: progress\ndata: {data}\n\n"
                            last_payload = data

                    if current_payload.get("status") in ("completed", "failed"):
                        return

            # Durable fallback: even when Redis is healthy, periodically check
            # Postgres so a missed stream event cannot leave the UI waiting.
            now = asyncio.get_running_loop().time()
            if now - last_db_check < 5:
                if redis_unavailable:
                    await asyncio.sleep(1)
                continue

            result = await get_scan(scan_id, user_id=user["id"])
            if not result:
                yield "event: scan-error\ndata: {\"detail\":\"Scan not found\"}\n\n"
                break

            last_db_check = now
            payload = {
                "id": result["id"],
                "status": result["status"],
                "report_id": result["report_id"],
                "progress_percent": result["progress_percent"],
                "progress_message": result["progress_message"],
            }
            data = json.dumps(payload)
            if data != last_payload:
                yield f"event: progress\ndata: {data}\n\n"
                last_payload = data

            if result["status"] in ("completed", "failed"):
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{scan_id}", response_model=ScanOut)
async def get_scan_route(scan_id: str, user: dict = Depends(get_current_user)):
    result = await get_scan(scan_id, user_id=user["id"])
    if not result:
        raise HTTPException(status_code=404, detail="Scan not found")
    return ScanOut(**result)


@router.delete("/{scan_id}")
async def delete_scan_route(scan_id: str, user: dict = Depends(get_current_user)):
    deleted = await delete_scan(scan_id, user_id=user["id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Scan not found")
    return {"detail": "Deleted"}
