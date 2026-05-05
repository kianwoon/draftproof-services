import asyncio
import json

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from app.models import ScanRequest, ScanOut
from app.services.scan_service import create_scan, get_scan, list_scans, delete_scan
from app.routes.auth import get_current_user

router = APIRouter()


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


@router.get("/{scan_id}/events")
async def stream_scan_events(
    scan_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
):
    async def event_stream():
        last_payload = None
        while True:
            if await request.is_disconnected():
                break

            result = await get_scan(scan_id, user_id=user["id"])
            if not result:
                yield "event: scan-error\ndata: {\"detail\":\"Scan not found\"}\n\n"
                break

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

            await asyncio.sleep(3)

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
