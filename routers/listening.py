"""
Listening test API routes.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from agents.listening_agent import generate_listening_test
from collections import deque
import uuid

router = APIRouter(prefix="/api/listening", tags=["listening"])

RECENT_SCENARIOS = deque(maxlen=20)

class GenerateRequest(BaseModel):
    topic: Optional[str] = "A student inquiring about a gym membership"
    seed: Optional[str] = None

@router.post("/generate")
async def generate_test(req: GenerateRequest):
    """
    Generates a new listening test scenario with audio lines and questions.
    """
    seed = req.seed or uuid.uuid4().hex
    result = await generate_listening_test(
        topic=req.topic or "A student inquiring about a gym membership",
        session_seed=seed,
        recent_scenarios=list(RECENT_SCENARIOS),
    )

    title = (result.get("title") or "").strip()
    if title:
        RECENT_SCENARIOS.append(title)

    return result
