"""
Text-to-Speech utility using Microsoft Edge TTS.
Free, high-quality, supports multiple English accents (perfect for IELTS).
"""

import edge_tts
import os
import uuid

# Available IELTS-relevant voices
VOICES = {
    "british_female": "en-GB-SoniaNeural",
    "british_male": "en-GB-RyanNeural",
    "american_female": "en-US-JennyNeural",
    "american_male": "en-US-GuyNeural",
    "australian_female": "en-AU-NatashaNeural",
    "australian_male": "en-AU-WilliamNeural",
}

# Default voice for the IELTS examiner
DEFAULT_VOICE = VOICES["british_female"]

# Directory for generated audio files
AUDIO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "audio_cache")
os.makedirs(AUDIO_DIR, exist_ok=True)


async def synthesize_speech(
    text: str,
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
) -> str:
    """
    Converts text to speech and saves as an MP3 file.

    Args:
        text: The text to speak.
        voice: Edge TTS voice name (use VOICES dict for presets).
        rate: Speech rate adjustment (e.g., "+10%", "-5%").

    Returns:
        Absolute path to the generated MP3 file.
    """
    filename = f"{uuid.uuid4().hex}.mp3"
    output_path = os.path.join(AUDIO_DIR, filename)

    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
    await communicate.save(output_path)

    return output_path
