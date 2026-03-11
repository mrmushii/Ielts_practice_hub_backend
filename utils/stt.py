"""
Speech-to-Text utility using the Groq Whisper API.
Transcribes audio files with very low latency.
"""

from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

_groq_client: Groq | None = None


def _get_groq_client() -> Groq:
    """Returns a cached Groq client."""
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


async def transcribe_audio(file_path: str, language: str = "en") -> dict:
    """
    Transcribes an audio file using Groq's Whisper API.

    Args:
        file_path: Path to the audio file (mp3, wav, webm, etc.)
        language: Language code (default: "en" for English)

    Returns:
        dict with "text" (transcription) and "duration" (audio length in seconds)
    """
    client = _get_groq_client()

    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
            language=language,
            response_format="verbose_json",
        )

    return {
        "text": transcription.text,
        "duration": getattr(transcription, "duration", None),
    }
