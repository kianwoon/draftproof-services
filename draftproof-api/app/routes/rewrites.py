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
