"""
Tutor Chat API routes.
"""

from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from agents.tutor_agent import chat_with_tutor
from utils.stt import transcribe_audio
import tempfile
import os
from typing import Optional

router = APIRouter(prefix="/api/tutor", tags=["tutor"])

class ChatRequest(BaseModel):
    message: str
    essay_context: Optional[str] = None
    history: Optional[list] = []
    session_id: Optional[str] = None
    use_langgraph: Optional[bool] = True

class ChatResponse(BaseModel):
    response: str

@router.post("/chat", response_model=ChatResponse)
async def ask_tutor(req: ChatRequest):
    """Sends a message to the Omni-Tutor agent."""
    reply = await chat_with_tutor(
        req.message,
        essay_context=req.essay_context,
        history=req.history,
        session_id=req.session_id,
        use_langgraph=req.use_langgraph,
    )
    return ChatResponse(response=reply)

@router.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Transcribes audio using Groq Whisper."""
    suffix = os.path.splitext(audio.filename or ".webm")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = await transcribe_audio(tmp_path)
        return {"text": result["text"]}
    except Exception as e:
        return {"error": str(e), "text": ""}
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
