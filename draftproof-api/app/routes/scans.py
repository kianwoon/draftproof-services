from fastapi import APIRouter
from app.models import ScanRequest, ScanOut

router = APIRouter()


@router.post("/", response_model=ScanOut)
async def create_scan(req: ScanRequest):
    # TODO: trigger scan_service.run_scan(req.document_id)
    return ScanOut(
        id="scan-placeholder",
        document_id=req.document_id,
        status="pending",
        report_id=None,
    )


@router.get("/{scan_id}", response_model=ScanOut)
async def get_scan(scan_id: str):
    # TODO: look up scan status
    return ScanOut(id=scan_id, document_id="", status="pending", report_id=None)
