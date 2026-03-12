"""
IELTS Listening Agent.
Generates a multi-speaker conversation script and comprehension questions using LLM.
Uses Edge TTS to convert the script into alternating voice audio files.
"""

import json
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from utils.llm import get_llm
from utils.tts import synthesize_speech

# ---- Structured Output Schemas ----

class DialogueLine(BaseModel):
    speaker: str = Field(description="The role of the speaker, e.g., 'Agent', 'Student', 'Manager'")
    text: str = Field(description="The spoken text for this line.")

class ListeningQuestion(BaseModel):
    id: str = Field(description="A unique ID like 'q1', 'q2'")
    text: str = Field(description="The question text")
    type: str = Field(description="Question type: 'mcq' or 'fill_blank'")
    options: Optional[List[str]] = Field(description="For MCQ only, exactly 4 options. Null for fill_blank.")
    correct_answer: str = Field(description="The exact correct text string from the options, or the fill-in-the-blank word")

class ListeningTestGeneration(BaseModel):
    title: str = Field(description="Title of the listening scenario")
    dialogue: List[DialogueLine] = Field(description="The script of the conversation")
    questions: List[ListeningQuestion] = Field(description="3 to 5 comprehension questions")

# ---- System Prompts ----

LISTENING_GENERATOR_SYSTEM = """You are an expert Cambridge IELTS Listening Test creator.

Your job is to generate a realistic IELTS Listening scenario (usually Part 1 or Part 3 style).
It must be a conversation between TWO distinct speakers.
- Part 1 style: Everyday social context (e.g., booking a hotel room, renting a car, inquiring about a club).
- Part 3 style: Educational/Training context (e.g., two students discussing an assignment, a tutor and a student).

Generate a flowing, natural dialogue of about 10-15 lines total. Include typical IELTS distractors (e.g., someone says a time, corrects themselves, or spells a name).

Then, generate 3-5 comprehension questions based on the dialogue. Mix multiple-choice (MCQ) and fill-in-the-blanks.

IMPORTANT: Strictly output the requested JSON schema.
"""

async def generate_listening_test(topic: str = "A student inquiring about a gym membership") -> dict:
    """
    Generates the test script and synthesizes the audio files.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(ListeningTestGeneration)
    
    messages = [
        SystemMessage(content=LISTENING_GENERATOR_SYSTEM),
        HumanMessage(content=f"Create an IELTS Listening test about: {topic}")
    ]
    
    # 1. Generate the script and questions
    test_data: ListeningTestGeneration = structured_llm.invoke(messages)
    
    # Identify unique speakers (expecting exactly 2)
    speakers = list(dict.fromkeys([line.speaker for line in test_data.dialogue]))
    
    # Assign voices to speakers (e.g., Voice 1 = British Male, Voice 2 = American Female)
    # Available (from our TTS util):
    # en-GB-RyanNeural, en-GB-SoniaNeural, en-US-ChristopherNeural, en-US-AriaNeural
    voice_map = {}
    default_voices = ["en-GB-RyanNeural", "en-US-AriaNeural", "en-AU-WilliamNeural"]
    for i, s in enumerate(speakers):
        voice_map[s] = default_voices[i % len(default_voices)]
        
    # 2. Synthesize audio for each line
    dialogue_with_audio = []
    
    for i, line in enumerate(test_data.dialogue):
        voice = voice_map.get(line.speaker, "en-US-AriaNeural")
        audio_filename = await synthesize_speech(text=line.text, voice=voice)
        
        dialogue_with_audio.append({
            "id": f"line_{i}",
            "speaker": line.speaker,
            "text": line.text,
            "audio_url": f"/api/audio/{audio_filename}"
        })
        
    return {
        "title": test_data.title,
        "dialogue": dialogue_with_audio,
        "questions": [q.model_dump() for q in test_data.questions]
    }
