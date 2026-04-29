from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import documents, scans, reports, rewrites

app = FastAPI(title="DraftProof API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(scans.router, prefix="/api/scans", tags=["scans"])
app.include_router(reports.router, prefix="/api/reports", tags=["reports"])
app.include_router(rewrites.router, prefix="/api/rewrites", tags=["rewrites"])


@app.get("/api/health")
def health():
    return {"status": "ok"}
