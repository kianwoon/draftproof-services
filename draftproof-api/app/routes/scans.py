import asyncio
import json
import re

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from app.models import ScanRequest, ScanOut
from app.services.scan_service import create_scan, get_scan, list_scans, delete_scan, get_free_scan_usage
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
        result = await create_scan(req.document_id, user_id=user["id"], text=req.text)
        return ScanOut(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scan failed: {e}")


@router.get("/free-usage")
async def get_free_usage_route(user: dict = Depends(get_current_user)):
    return await get_free_scan_usage(user["id"])


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

        last_db_check = asyncio.get_running_loop().time()

        while True:
            if await request.is_disconnected():
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
