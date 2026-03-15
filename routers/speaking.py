"""
Speaking test API routes.
Handles session management, audio transcription, examiner responses, and TTS.
"""

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from agents.speaking_agent import examiner_respond, generate_feedback
from utils.stt import transcribe_audio
from utils.tts import synthesize_speech, VOICES
from utils.db import get_db
from models.session import SpeakingSession, ChatMessage
import tempfile
import os
import uuid
import random

router = APIRouter(prefix="/api/speaking", tags=["speaking"])

EXAMINER_NAMES = ["Sarah", "Emma", "Daniel", "Oliver", "Hannah", "James"]
OPENING_STYLES = ["warm", "energetic", "formal", "friendly"]

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
    candidate_text: str | None = None


class FeedbackResponse(BaseModel):
    feedback: str


class TranscriptMessage(BaseModel):
    role: str
    content: str


class TranscriptFeedbackRequest(BaseModel):
    transcript: list[TranscriptMessage]


# ---- Start a new speaking test ----

@router.post("/start", response_model=StartSessionResponse)
async def start_session(
    voice: str = Form("british_female"),
    db=Depends(get_db),
):
    """Starts a new IELTS speaking test session. Examiner introduces and asks the first question."""
    session_id = uuid.uuid4().hex

    # Randomize topic to ensure infinite test variety
    topics = ["Technology and AI", "Environmental Protection", "Childhood Memories", "Modern Transportation", "Global Tourism", "Arts and Culture", "Space Exploration", "Social Media", "Historical Events", "Healthy Lifestyle"]
    topic_seed = random.choice(topics)

    examiner_name = random.choice(EXAMINER_NAMES)
    opener_style = random.choice(OPENING_STYLES)

    # Get examiner's opening (Part 1 intro + first question)
    examiner_text = await examiner_respond(
        part=1,
        topic_seed=topic_seed,
        history=[],
        examiner_name=examiner_name,
        opener_style=opener_style,
        session_id=session_id,
    )

    # Save to history
    history = [ChatMessage(role="examiner", content=examiner_text)]

    # Generate TTS audio
    tts_voice = VOICES.get(voice, VOICES["british_female"])
    audio_path = await synthesize_speech(text=examiner_text, voice=tts_voice)

    # Initialize and save session to DB
    session = SpeakingSession(
        session_id=session_id,
        part=1,
        voice=voice,
        topic_seed=topic_seed,
        history=history,
        examiner_name=examiner_name,
        opener_style=opener_style,
    )
    
    await db.speaking_sessions.insert_one(session.model_dump())

    return StartSessionResponse(
        session_id=session_id,
        part=1,
        examiner_text=examiner_text,
        audio_url=f"/api/speaking/audio/{os.path.basename(audio_path)}",
    )


# ---- Candidate responds (text) ----

@router.post("/respond", response_model=RespondResponse)
async def respond_text(req: RespondRequest, db=Depends(get_db)):
    """Candidate sends a text response; examiner replies."""
    
    # Retrieve session from DB
    session_doc = await db.speaking_sessions.find_one({"session_id": req.session_id})
    if not session_doc:
        raise HTTPException(status_code=404, detail="Session not found. Please start a new test.")
        
    session = SpeakingSession(**session_doc)
    
    # Check if already complete
    if session.is_complete:
        return RespondResponse(
            examiner_text="The speaking test is now complete. You can request your feedback.",
            audio_url="",
            part=3,
            is_complete=True,
            candidate_text=None,
        )

    # Add candidate's response to history
    session.history.append(ChatMessage(role="candidate", content=req.candidate_text))
    # Detect part transitions from examiner's previous messages
    last_examiner = ""
    for msg in reversed(session.history[:-1]): # excluding the candidate payload we just pushed
        if msg.role == "examiner":
            last_examiner = msg.content.lower()
            break

    if "move on to part 2" in last_examiner and session.part == 1:
        session.part = 2
    elif "move on to part 3" in last_examiner and session.part == 2:
        session.part = 3

    is_complete = "end of the speaking test" in last_examiner and session.part == 3

    if is_complete:
        session.is_complete = True
        await db.speaking_sessions.replace_one({"session_id": session.session_id}, session.model_dump())
        return RespondResponse(
            examiner_text="The speaking test is now complete. You can request your feedback.",
            audio_url="",
            part=3,
            is_complete=True,
            candidate_text=None,
        )

    # Convert history format for agent
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in session.history]

    # Get examiner's next response
    examiner_text = await examiner_respond(
        part=session.part,
        topic_seed=session.topic_seed,
        history=history_dicts,
        examiner_name=session.examiner_name,
        opener_style=session.opener_style,
        session_id=session.session_id,
    )

    session.history.append(ChatMessage(role="examiner", content=examiner_text))

    # Check if THIS response signals completion
    is_complete = "end of the speaking test" in examiner_text.lower()
    if is_complete:
        session.is_complete = True

    # Save updated session
    await db.speaking_sessions.replace_one({"session_id": session.session_id}, session.model_dump())

    # Generate TTS
    tts_voice = VOICES.get(session.voice, VOICES["british_female"])
    audio_path = await synthesize_speech(text=examiner_text, voice=tts_voice)

    return RespondResponse(
        examiner_text=examiner_text,
        audio_url=f"/api/speaking/audio/{os.path.basename(audio_path)}",
        part=session.part,
        is_complete=is_complete,
        candidate_text=None,
    )


# ---- Candidate responds (audio) ----

@router.post("/respond-audio", response_model=RespondResponse)
async def respond_audio(
    session_id: str = Form(...),
    audio: UploadFile = File(...),
    db=Depends(get_db)
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
    response = await respond_text(req, db)
    response.candidate_text = candidate_text
    return response


# ---- Get feedback ----

@router.post("/feedback", response_model=FeedbackResponse)
async def get_feedback(session_id: str = Form(...), db=Depends(get_db)):
    """Generates detailed IELTS feedback for the completed speaking test."""
    
    session_doc = await db.speaking_sessions.find_one({"session_id": session_id})
    if not session_doc:
        raise HTTPException(status_code=404, detail="Session not found.")
        
    session = SpeakingSession(**session_doc)
    
    # Use cached feedback if already generated
    if session.feedback:
        return FeedbackResponse(feedback=session.feedback)

    history_dicts = [{"role": msg.role, "content": msg.content} for msg in session.history]
    feedback = await generate_feedback(history_dicts)
    
    # Save feedback to DB
    session.feedback = feedback
    await db.speaking_sessions.replace_one({"session_id": session.session_id}, session.model_dump())
    
    return FeedbackResponse(feedback=feedback)


@router.post("/feedback-from-transcript", response_model=FeedbackResponse)
async def get_feedback_from_transcript(req: TranscriptFeedbackRequest):
    """Generates IELTS speaking feedback from a realtime transcript payload."""
    if not req.transcript:
        raise HTTPException(status_code=400, detail="Transcript is empty.")

    history_dicts = [
        {"role": msg.role, "content": msg.content}
        for msg in req.transcript
        if msg.content.strip()
    ]
    if not history_dicts:
        raise HTTPException(status_code=400, detail="Transcript is empty.")

    feedback = await generate_feedback(history_dicts)
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
