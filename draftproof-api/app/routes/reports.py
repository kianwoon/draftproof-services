from fastapi import APIRouter, HTTPException
from app.models import ReportOut
from app.services.report_service import get_report as fetch_report

router = APIRouter()


@router.get("/{report_id}", response_model=ReportOut)
async def get_report(report_id: str):
    result = await fetch_report(report_id)
    if not result:
        raise HTTPException(status_code=404, detail="Report not found")
    return ReportOut(**result)
