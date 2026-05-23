from fastapi import APIRouter, HTTPException
from app.core.job_store import job_store

router = APIRouter(tags=["jobs"])

@router.get("/v1/jobs/{job_id}")
def get_job(job_id: str):
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
