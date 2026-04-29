from fastapi import APIRouter, HTTPException
from app.models import SuggestionOut, ApplySuggestionRequest

router = APIRouter()


@router.get("/{issue_id}", response_model=SuggestionOut)
async def get_suggestion(issue_id: str):
    # TODO: generate rewrite suggestion via rewrite_service
    raise HTTPException(status_code=404, detail="Issue not found")


@router.post("/{issue_id}/apply")
async def apply_suggestion(issue_id: str, req: ApplySuggestionRequest):
    # TODO: apply rewrite via rewrite_service
    return {"status": "applied"}
