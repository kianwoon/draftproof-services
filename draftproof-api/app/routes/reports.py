from fastapi import APIRouter, HTTPException, Depends
from app.models import ReportOut
from app.services.report_service import get_report as fetch_report
from app.routes.auth import get_current_user

router = APIRouter()


@router.get("/{report_id}", response_model=ReportOut)
async def get_report(report_id: str, user: dict = Depends(get_current_user)):
    result = await fetch_report(report_id)
    if not result:
        raise HTTPException(status_code=404, detail="Report not found")
    return ReportOut(**result)
