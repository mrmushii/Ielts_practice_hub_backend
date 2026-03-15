"""
Tutor Chat API routes.
"""

from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from agents.tutor_agent import chat_with_tutor, suggest_tutor_actions
from utils.stt import transcribe_audio
import tempfile
import os
from typing import Optional, Literal

router = APIRouter(prefix="/api/tutor", tags=["tutor"])

class ChatRequest(BaseModel):
    message: str
    essay_context: Optional[str] = None
    history: Optional[list] = []
    session_id: Optional[str] = None
    use_langgraph: Optional[bool] = True


class TutorAction(BaseModel):
    id: str
    type: Literal["navigate_module", "open_tutor_workspace"]
    module: str
    route: str
    label: str
    description: str
    requires_confirmation: bool = True


class TutorResponseMeta(BaseModel):
    intent: str
    confidence: float
    reason: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    actions: list[TutorAction] = []
    meta: TutorResponseMeta

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
    action_data = suggest_tutor_actions(req.message)
    return ChatResponse(
        response=reply,
        actions=[TutorAction(**item) for item in action_data["actions"]],
        meta=TutorResponseMeta(
            intent=action_data["intent"],
            confidence=action_data["confidence"],
            reason=action_data["reason"],
            session_id=req.session_id,
        ),
    )

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
