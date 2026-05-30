import asyncio
import json
import re

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from app.models import RewriteCreateRequest, RewriteOut, RewriteReportOut
from app.services import progress_stream
from app.services import rewrite_service
from app.routes.auth import get_current_user

router = APIRouter()
_REDIS_STREAM_ID_RE = re.compile(r"^\d+-\d+$")
_TERMINAL_REWRITE_STATUSES = ("completed", "failed", "canceled")


@router.post("/", response_model=RewriteOut)
async def create_rewrite(req: RewriteCreateRequest, user: dict = Depends(get_current_user)):
    try:
        result = await rewrite_service.create_rewrite(req.scan_id, user["id"])
        return RewriteOut(**result)
    except rewrite_service.NoRewriteableFindingsError as e:
        raise HTTPException(status_code=422, detail=str(e))
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


@router.post("/{rewrite_id}/cancel", response_model=RewriteOut)
async def cancel_rewrite(rewrite_id: str, user: dict = Depends(get_current_user)):
    result = await rewrite_service.cancel_rewrite(rewrite_id, user_id=user["id"])
    if not result:
        raise HTTPException(status_code=404, detail="Rewrite not found")
    return RewriteOut(**result)


@router.get("/{rewrite_id}/events")
async def stream_rewrite_events(
    rewrite_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
):
    def _payload_from_rewrite(result: dict) -> dict:
        return {
            "id": result["id"],
            "scan_id": result["scan_id"],
            "status": result["status"],
            "error": result["error"],
            "progress_percent": result["progress_percent"],
            "progress_message": result["progress_message"],
            "created_at": result["created_at"],
            "completed_at": result["completed_at"],
        }

    def _payload_from_stream(fields: dict, fallback: dict | None = None) -> dict:
        payload = dict(fallback or {})
        if fields.get("id"):
            payload["id"] = fields["id"]
        if fields.get("scan_id"):
            payload["scan_id"] = fields["scan_id"]
        if fields.get("status"):
            payload["status"] = fields["status"]
        if "error" in fields:
            payload["error"] = fields.get("error") or None
        if "progress_message" in fields:
            payload["progress_message"] = fields.get("progress_message")
        if "progress_percent" in fields:
            try:
                payload["progress_percent"] = max(0, min(100, int(fields["progress_percent"])))
            except (TypeError, ValueError):
                pass
        return payload

    async def event_stream():
        last_payload = None
        last_db_check = 0.0
        redis_unavailable = False
        last_stream_id = request.headers.get("last-event-id") or "$"
        if last_stream_id != "$" and not _REDIS_STREAM_ID_RE.match(last_stream_id):
            last_stream_id = "$"

        result = await rewrite_service.get_rewrite(rewrite_id, user_id=user["id"])
        if not result:
            yield "event: rewrite-error\ndata: {\"detail\":\"Rewrite not found\"}\n\n"
            return

        current_payload = _payload_from_rewrite(result)
        data = json.dumps(current_payload)
        yield f"event: progress\ndata: {data}\n\n"
        last_payload = data
        last_db_check = asyncio.get_running_loop().time()

        if result["status"] in _TERMINAL_REWRITE_STATUSES:
            return

        while True:
            if await request.is_disconnected():
                break

            if not redis_unavailable:
                events = await progress_stream.read_rewrite_progress(
                    rewrite_id,
                    last_stream_id,
                    count=10,
                )
                if events is None:
                    redis_unavailable = True
                else:
                    for event_id, fields in events:
                        last_stream_id = event_id
                        current_payload = _payload_from_stream(fields, current_payload)
                        data = json.dumps(current_payload)
                        if data != last_payload:
                            yield f"id: {event_id}\nevent: progress\ndata: {data}\n\n"
                            last_payload = data
                        if current_payload.get("status") in _TERMINAL_REWRITE_STATUSES:
                            result = await rewrite_service.get_rewrite(rewrite_id, user_id=user["id"])
                            if result:
                                current_payload = _payload_from_rewrite(result)
                                data = json.dumps(current_payload)
                                if data != last_payload:
                                    yield f"event: progress\ndata: {data}\n\n"
                            return

            now = asyncio.get_running_loop().time()
            should_check_db = redis_unavailable or now - last_db_check >= 5.0
            if should_check_db:
                result = await rewrite_service.get_rewrite(rewrite_id, user_id=user["id"])
                if not result:
                    yield "event: rewrite-error\ndata: {\"detail\":\"Rewrite not found\"}\n\n"
                    break
                current_payload = _payload_from_rewrite(result)
                data = json.dumps(current_payload)
                if data != last_payload:
                    yield f"event: progress\ndata: {data}\n\n"
                    last_payload = data
                last_db_check = now

                if result["status"] in _TERMINAL_REWRITE_STATUSES:
                    break

            if redis_unavailable:
                await asyncio.sleep(1)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{rewrite_id}/report", response_model=RewriteReportOut)
async def get_rewrite_report(rewrite_id: str, user: dict = Depends(get_current_user)):
    data = await rewrite_service.get_rewrite_report(rewrite_id, user["id"])
    if not data:
        raise HTTPException(status_code=404, detail="Rewrite report not found")
    return RewriteReportOut(**data)


@router.post("/{rewrite_id}/report/regenerate")
async def regenerate_rewrite_report(rewrite_id: str, user: dict = Depends(get_current_user)):
    try:
        result = await rewrite_service.regenerate_rewrite_report_assets(rewrite_id, user["id"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not result:
        raise HTTPException(status_code=404, detail="Rewrite not found")
    return result


@router.get("/{rewrite_id}/download/{fmt}")
async def download_rewrite(rewrite_id: str, fmt: str, user: dict = Depends(get_current_user)):
    if fmt not in ("pdf", "md", "txt", "log"):
        raise HTTPException(status_code=400, detail="Format must be pdf, md, txt, or log")
    url = await rewrite_service.get_rewrite_download_url(rewrite_id, fmt, user["id"])
    if not url:
        raise HTTPException(status_code=404, detail="Download not available")
    return {"url": url}


@router.get("/{rewrite_id}/detect-json")
async def download_detect_json(rewrite_id: str, user: dict = Depends(get_current_user)):
    url = await rewrite_service.get_detect_json_url(rewrite_id, user["id"])
    if not url:
        raise HTTPException(status_code=404, detail="Detect scan JSON not available")
    return {"url": url}
