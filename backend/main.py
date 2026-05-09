from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from routers import generation, models, explainability_remote, posthoc, interpret

app = FastAPI(title="LLM Explainability API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router, prefix="/models", tags=["models"])
app.include_router(generation.router, prefix="/generate", tags=["generation"])
app.include_router(explainability_remote.router, prefix="/explain", tags=["explainability"])
app.include_router(posthoc.router, prefix="/posthoc", tags=["posthoc"])
app.include_router(interpret.router, prefix="/ai", tags=["interpret"])


@app.get("/health")
def health():
    return {"status": "ok"}
