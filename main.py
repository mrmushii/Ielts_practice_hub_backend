from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from routers.core import router as core_router
from routers.speaking import router as speaking_router
from routers.writing import router as writing_router
from routers.reading import router as reading_router
from routers.listening import router as listening_router
from routers.tutor import router as tutor_router
from routers.documents import router as documents_router
from utils.db import connect_to_mongo, close_mongo_connection

load_dotenv() # Loads LANGCHAIN_* vars for LangSmith tracing

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await connect_to_mongo()
    yield
    # Shutdown
    await close_mongo_connection()

app = FastAPI(
    title="IELTS Platform API",
    description="Backend API for the IELTS Preparation Platform",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
                   "https://ielts-practice-hub-nine.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(core_router)
app.include_router(speaking_router)
app.include_router(writing_router)
app.include_router(reading_router)
app.include_router(listening_router)
app.include_router(tutor_router)
app.include_router(documents_router)


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok", "service": "ielts-platform-api"}
