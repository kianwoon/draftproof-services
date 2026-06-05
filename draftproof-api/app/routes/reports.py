from fastapi import APIRouter, HTTPException, Depends, Response
from app.models import ReportOut
from app.services.report_service import get_report as fetch_report
from app.routes.auth import get_current_user

router = APIRouter()


@router.get("/{report_id}", response_model=ReportOut)
async def get_report(report_id: str, response: Response, user: dict = Depends(get_current_user)):
    result = await fetch_report(report_id, user_id=user["id"])
    if not result:
        raise HTTPException(status_code=404, detail="Report not found")
    # Reports are immutable once written — safe to cache privately for 24h.
    response.headers["Cache-Control"] = "private, max-age=86400"
    response.headers["ETag"] = f'"{report_id}"'
    return ReportOut(**result)
