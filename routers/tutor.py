"""
Tutor Chat API routes.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from agents.tutor_agent import chat_with_tutor

router = APIRouter(prefix="/api/tutor", tags=["tutor"])

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@router.post("/chat", response_model=ChatResponse)
async def ask_tutor(req: ChatRequest):
    """Sends a message to the Omni-Tutor agent."""
    reply = await chat_with_tutor(req.message)
    return ChatResponse(response=reply)
