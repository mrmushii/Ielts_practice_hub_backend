"""
Core API routes — chat, speech-to-text, and text-to-speech.
These are shared infrastructure endpoints used by all IELTS modules.
"""

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from utils.llm import get_llm
from utils.stt import transcribe_audio
from utils.tts import synthesize_speech, VOICES
import tempfile
import os

router = APIRouter(prefix="/api", tags=["core"])


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ---- Schemas ----

class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful assistant."


class ChatResponse(BaseModel):
    reply: str


class TranscriptionResponse(BaseModel):
    text: str
    duration: float | None


class TTSRequest(BaseModel):
    text: str
    voice: str = "british_female"
    rate: str = "+0%"


# ---- Chat Endpoint ----

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Simple chat endpoint to test the LLM connection."""
    llm = get_llm()
    messages = [
        SystemMessage(content=req.system_prompt),
        HumanMessage(content=req.message),
    ]
    response = llm.invoke(messages)
    return ChatResponse(reply=response.content)


# ---- Speech-to-Text ----

@router.post("/stt", response_model=TranscriptionResponse)
async def speech_to_text(audio: UploadFile = File(...)):
    """Transcribes uploaded audio using Groq Whisper API."""
    # Save uploaded file to temp
    suffix = os.path.splitext(audio.filename or ".webm")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = await transcribe_audio(tmp_path)
        return TranscriptionResponse(text=result["text"], duration=result["duration"])
    finally:
        os.unlink(tmp_path)


# ---- Text-to-Speech ----

@router.post("/tts")
async def text_to_speech(req: TTSRequest):
    """Converts text to speech and returns the MP3 audio file."""
    voice = VOICES.get(req.voice, VOICES["british_female"])
    audio_path = await synthesize_speech(text=req.text, voice=voice, rate=req.rate)
    return FileResponse(
        audio_path,
        media_type="audio/mpeg",
        filename="speech.mp3",
    )


# ---- Available Voices ----

@router.get("/voices")
async def list_voices():
    """Lists available TTS voices."""
    return {"voices": list(VOICES.keys())}


@router.get("/langgraph-status")
async def langgraph_status():
    """Reports LangGraph rollout and runtime configuration status."""
    checkpoint_mode = os.getenv("LANGGRAPH_CHECKPOINTER", "memory").strip().lower() or "memory"

    module_flags = {
        "tutor": _env_flag("ENABLE_LANGGRAPH_TUTOR", True),
        "speaking": _env_flag("ENABLE_LANGGRAPH_SPEAKING", True),
        "listening": _env_flag("ENABLE_LANGGRAPH_LISTENING", True),
        "writing": _env_flag("ENABLE_LANGGRAPH_WRITING", True),
        "reading": _env_flag("ENABLE_LANGGRAPH_READING", True),
    }

    return {
        "langgraph": {
            "enabled_modules": module_flags,
            "checkpoint_mode": checkpoint_mode,
            "mongo_checkpoint": {
                "db": os.getenv("LANGGRAPH_MONGO_DB", "ielts_platform"),
                "collection": os.getenv("LANGGRAPH_MONGO_COLLECTION", "langgraph_checkpoints"),
                "configured": bool(os.getenv("MONGODB_URI")),
            },
        }
    }


# ---- Serve Audio Files ----

@router.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """Serves generated audio files."""
    from utils.tts import AUDIO_DIR
    import os
    from fastapi import HTTPException
    
    file_path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
        
    return FileResponse(file_path, media_type="audio/mpeg")
