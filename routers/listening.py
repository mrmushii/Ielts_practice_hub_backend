"""
Listening test API routes.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from agents.listening_agent import generate_listening_test

router = APIRouter(prefix="/api/listening", tags=["listening"])

class GenerateRequest(BaseModel):
    topic: Optional[str] = "A student inquiring about a gym membership"

@router.post("/generate")
async def generate_test(req: GenerateRequest):
    """
    Generates a new listening test scenario with audio lines and questions.
    """
    result = await generate_listening_test(topic=req.topic)
    return result
