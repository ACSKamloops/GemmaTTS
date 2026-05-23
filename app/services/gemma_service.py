from fastapi import FastAPI
from app.api.generate import router as generate_router, rate_limiter
from unittest.mock import MagicMock

model_manager = MagicMock()

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "healthy", "service": "gemma-service"}

app.include_router(generate_router)
