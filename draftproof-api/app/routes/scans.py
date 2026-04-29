from fastapi import APIRouter, HTTPException
from app.models import ScanRequest, ScanOut
from app.services.scan_service import create_scan, get_scan

router = APIRouter()


@router.post("/", response_model=ScanOut)
async def create_scan_route(req: ScanRequest):
    try:
        result = await create_scan(req.document_id)
        return ScanOut(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scan failed: {e}")


@router.get("/{scan_id}", response_model=ScanOut)
async def get_scan_route(scan_id: str):
    result = await get_scan(scan_id)
    if not result:
        raise HTTPException(status_code=404, detail="Scan not found")
    return ScanOut(**result)
