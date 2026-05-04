import asyncio
import json

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from app.models import RewriteCreateRequest, RewriteOut, RewriteReportOut
from app.services import rewrite_service
from app.routes.auth import get_current_user

router = APIRouter()


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


@router.get("/{rewrite_id}/events")
async def stream_rewrite_events(
    rewrite_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
):
    async def event_stream():
        last_payload = None
        while True:
            if await request.is_disconnected():
                break

            result = await rewrite_service.get_rewrite(rewrite_id, user_id=user["id"])
            if not result:
                yield "event: rewrite-error\ndata: {\"detail\":\"Rewrite not found\"}\n\n"
                break

            payload = {
                "id": result["id"],
                "scan_id": result["scan_id"],
                "status": result["status"],
                "error": result["error"],
                "progress_percent": result["progress_percent"],
                "progress_message": result["progress_message"],
                "created_at": result["created_at"],
                "completed_at": result["completed_at"],
            }
            data = json.dumps(payload)
            if data != last_payload:
                yield f"event: progress\ndata: {data}\n\n"
                last_payload = data

            if result["status"] in ("completed", "failed"):
                break

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
