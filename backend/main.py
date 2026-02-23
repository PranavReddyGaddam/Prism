from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import generation, explainability, models

app = FastAPI(title="LLM Explainability API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router, prefix="/models", tags=["models"])
app.include_router(generation.router, prefix="/generate", tags=["generation"])
app.include_router(explainability.router, prefix="/explain", tags=["explainability"])


@app.get("/health")
def health():
    return {"status": "ok"}
