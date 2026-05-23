import threading
import uuid
from typing import Dict, Any, Optional

class JobStore:
    """
    Thread-safe store for tracking dialog and synthesis jobs.
    """
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_job(self, initial_state: str = "processing") -> str:
        """Create a new job and return its UUID string."""
        job_id = str(uuid.uuid4())
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "state": initial_state,
                "metrics": {},
                "text": ""
            }
        return job_id

    def update_job(self, job_id: str, updates: Dict[str, Any]) -> None:
        """Update job fields thread-safely."""
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"Job {job_id} not found.")
            self._jobs[job_id].update(updates)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve job details thread-safely."""
        with self._lock:
            return self._jobs.get(job_id)

# Global Job Store instance
job_store = JobStore()
