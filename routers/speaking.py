"""
Speaking test API routes.
Handles session management, audio transcription, examiner responses, and TTS.
"""

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from agents.speaking_agent import examiner_respond, generate_feedback
from utils.stt import transcribe_audio
from utils.tts import synthesize_speech, VOICES
import tempfile
import os
import uuid

router = APIRouter(prefix="/api/speaking", tags=["speaking"])

# ---- In-memory session store (replaced with MongoDB in Step 4) ----
sessions: dict[str, dict] = {}


# ---- Schemas ----

class StartSessionResponse(BaseModel):
    session_id: str
    part: int
    examiner_text: str
    audio_url: str


class RespondRequest(BaseModel):
    session_id: str
    candidate_text: str


class RespondResponse(BaseModel):
    examiner_text: str
    audio_url: str
    part: int
    is_complete: bool


class FeedbackResponse(BaseModel):
    feedback: str


# ---- Start a new speaking test ----

@router.post("/start", response_model=StartSessionResponse)
async def start_session(voice: str = "british_female"):
    """Starts a new IELTS speaking test session. Examiner introduces and asks the first question."""
    session_id = uuid.uuid4().hex

    # Initialize session
    sessions[session_id] = {
        "part": 1,
        "history": [],
        "voice": voice,
    }

    # Get examiner's opening (Part 1 intro + first question)
    examiner_text = await examiner_respond(part=1, history=[])

    # Save to history
    sessions[session_id]["history"].append({
        "role": "examiner",
        "content": examiner_text,
    })

    # Generate TTS audio
    tts_voice = VOICES.get(voice, VOICES["british_female"])
    audio_path = await synthesize_speech(text=examiner_text, voice=tts_voice)

    return StartSessionResponse(
        session_id=session_id,
        part=1,
        examiner_text=examiner_text,
        audio_url=f"/api/speaking/audio/{os.path.basename(audio_path)}",
    )


# ---- Candidate responds (text) ----

@router.post("/respond", response_model=RespondResponse)
async def respond_text(req: RespondRequest):
    """Candidate sends a text response; examiner replies."""
    session = sessions.get(req.session_id)
    if not session:
        raise ValueError("Session not found. Please start a new test.")

    # Add candidate's response to history
    session["history"].append({
        "role": "candidate",
        "content": req.candidate_text,
    })

    # Detect part transitions from examiner's previous messages
    current_part = session["part"]
    last_examiner = ""
    for msg in reversed(session["history"]):
        if msg["role"] == "examiner":
            last_examiner = msg["content"].lower()
            break

    if "move on to part 2" in last_examiner and current_part == 1:
        session["part"] = 2
    elif "move on to part 3" in last_examiner and current_part == 2:
        session["part"] = 3

    is_complete = "end of the speaking test" in last_examiner and current_part == 3

    if is_complete:
        return RespondResponse(
            examiner_text="The speaking test is now complete. You can request your feedback.",
            audio_url="",
            part=3,
            is_complete=True,
        )

    # Get examiner's next response
    examiner_text = await examiner_respond(
        part=session["part"],
        history=session["history"],
    )

    session["history"].append({
        "role": "examiner",
        "content": examiner_text,
    })

    # Check if THIS response signals completion
    is_complete = "end of the speaking test" in examiner_text.lower()

    # Generate TTS
    tts_voice = VOICES.get(session["voice"], VOICES["british_female"])
    audio_path = await synthesize_speech(text=examiner_text, voice=tts_voice)

    return RespondResponse(
        examiner_text=examiner_text,
        audio_url=f"/api/speaking/audio/{os.path.basename(audio_path)}",
        part=session["part"],
        is_complete=is_complete,
    )


# ---- Candidate responds (audio) ----

@router.post("/respond-audio", response_model=RespondResponse)
async def respond_audio(
    session_id: str = Form(...),
    audio: UploadFile = File(...),
):
    """Candidate sends audio; it's transcribed then processed like text."""
    # Save uploaded audio to temp file
    suffix = os.path.splitext(audio.filename or ".webm")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Transcribe
        result = await transcribe_audio(tmp_path)
        candidate_text = result["text"]
    finally:
        os.unlink(tmp_path)

    # Process as text response
    req = RespondRequest(session_id=session_id, candidate_text=candidate_text)
    return await respond_text(req)


# ---- Get feedback ----

@router.post("/feedback", response_model=FeedbackResponse)
async def get_feedback(session_id: str = Form(...)):
    """Generates detailed IELTS feedback for the completed speaking test."""
    session = sessions.get(session_id)
    if not session:
        raise ValueError("Session not found.")

    feedback = await generate_feedback(session["history"])
    return FeedbackResponse(feedback=feedback)


# ---- Serve TTS audio files ----

@router.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serves generated TTS audio files."""
    audio_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "audio_cache")
    file_path = os.path.join(audio_dir, filename)
    if not os.path.exists(file_path):
        raise ValueError("Audio file not found.")
    return FileResponse(file_path, media_type="audio/mpeg")
