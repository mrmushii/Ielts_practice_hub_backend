"""
Tutor Chat API routes.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from agents.tutor_agent import chat_with_tutor

from typing import Optional

router = APIRouter(prefix="/api/tutor", tags=["tutor"])

class ChatRequest(BaseModel):
    message: str
    essay_context: Optional[str] = None

class ChatResponse(BaseModel):
    response: str

@router.post("/chat", response_model=ChatResponse)
async def ask_tutor(req: ChatRequest):
    """Sends a message to the Omni-Tutor agent."""
    reply = await chat_with_tutor(req.message, essay_context=req.essay_context)
    return ChatResponse(response=reply)
