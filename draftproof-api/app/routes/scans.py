from fastapi import APIRouter, HTTPException, Depends
from app.models import ScanRequest, ScanOut
from app.services.scan_service import create_scan, get_scan, list_scans
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


@router.get("/{scan_id}", response_model=ScanOut)
async def get_scan_route(scan_id: str, user: dict = Depends(get_current_user)):
    result = await get_scan(scan_id)
    if not result:
        raise HTTPException(status_code=404, detail="Scan not found")
    return ScanOut(**result)
