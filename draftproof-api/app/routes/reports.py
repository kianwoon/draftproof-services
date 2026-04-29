from fastapi import APIRouter, HTTPException
from app.models import ReportOut

router = APIRouter()


@router.get("/{report_id}", response_model=ReportOut)
async def get_report(report_id: str):
    # TODO: fetch from report_service
    raise HTTPException(status_code=404, detail="Report not found")
