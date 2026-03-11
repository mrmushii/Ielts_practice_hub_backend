from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from routers.core import router as core_router
from routers.speaking import router as speaking_router

load_dotenv()

app = FastAPI(
    title="IELTS Platform API",
    description="Backend API for the IELTS Preparation Platform",
    version="0.1.0",
)

# CORS — allow the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(core_router)
app.include_router(speaking_router)


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok", "service": "ielts-platform-api"}
