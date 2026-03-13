"""
IELTS Listening Agent.
Generates a multi-speaker conversation script and comprehension questions using LLM.
Uses Edge TTS to convert the script into alternating voice audio files.
"""

import os
import uuid
from functools import lru_cache
from typing import List, Optional, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from utils.llm import get_llm
from utils.tts import synthesize_speech
from utils.langgraph_runtime import get_langgraph_checkpointer

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


def _format_recent_scenarios(recent_scenarios: Optional[list[str]]) -> str:
    if not recent_scenarios:
        return "- None"
    return "\n".join(f"- {item}" for item in recent_scenarios)


async def generate_listening_test(
    topic: str = "A student inquiring about a gym membership",
    session_seed: str | None = None,
    recent_scenarios: Optional[list[str]] = None,
    use_langgraph: bool = True,
) -> dict:
    """
    Generates the test script and synthesizes the audio files.
    """
    test_data = await _generate_test_data(
        topic=topic,
        session_seed=session_seed,
        recent_scenarios=recent_scenarios,
        use_langgraph=use_langgraph,
    )
    
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


class ListeningGraphState(TypedDict, total=False):
    topic: str
    session_seed: str
    recent_scenarios: list[str]
    generated: ListeningTestGeneration


def _build_generation_messages(topic: str, session_seed: str, recent_scenarios: Optional[list[str]]):
    return [
        SystemMessage(content=LISTENING_GENERATOR_SYSTEM),
        HumanMessage(
            content=(
                f"Create an IELTS Listening test about: {topic}\n\n"
                f"Uniqueness seed: {session_seed or 'default-seed'}\n"
                "Make the situation, names, places, numbers, and events different from common templates.\n"
                "Do not repeat any of these recent scenarios:\n"
                f"{_format_recent_scenarios(recent_scenarios)}\n\n"
                "Ensure at least one specific date/time and one corrected detail (IELTS-style distractor)."
            )
        ),
    ]


def _generate_structured_node(state: ListeningGraphState) -> ListeningGraphState:
    llm = get_llm()
    structured_llm = llm.with_structured_output(ListeningTestGeneration)
    messages = _build_generation_messages(
        topic=state.get("topic", "A student inquiring about a gym membership"),
        session_seed=state.get("session_seed", "default-seed"),
        recent_scenarios=state.get("recent_scenarios", []),
    )
    generated = structured_llm.invoke(messages)
    return {"generated": generated}


@lru_cache(maxsize=1)
def _get_listening_graph():
    builder = StateGraph(ListeningGraphState)
    builder.add_node("generate_structured", _generate_structured_node)
    builder.add_edge(START, "generate_structured")
    builder.add_edge("generate_structured", END)
    return builder.compile(checkpointer=get_langgraph_checkpointer())


async def _generate_test_data_langgraph(
    topic: str,
    session_seed: str,
    recent_scenarios: Optional[list[str]],
) -> ListeningTestGeneration:
    graph = _get_listening_graph()
    thread_id = f"listening-{session_seed or uuid.uuid4().hex}"
    result = await graph.ainvoke(
        {
            "topic": topic,
            "session_seed": session_seed,
            "recent_scenarios": recent_scenarios or [],
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    generated = result.get("generated")
    if not isinstance(generated, ListeningTestGeneration):
        raise ValueError("LangGraph listening generation returned invalid payload")
    return generated


async def _generate_test_data_legacy(
    topic: str,
    session_seed: str,
    recent_scenarios: Optional[list[str]],
) -> ListeningTestGeneration:
    llm = get_llm()
    structured_llm = llm.with_structured_output(ListeningTestGeneration)
    messages = _build_generation_messages(topic, session_seed, recent_scenarios)
    return structured_llm.invoke(messages)


async def _generate_test_data(
    topic: str,
    session_seed: str | None,
    recent_scenarios: Optional[list[str]],
    use_langgraph: bool,
) -> ListeningTestGeneration:
    seed = session_seed or uuid.uuid4().hex
    feature_enabled = os.getenv("ENABLE_LANGGRAPH_LISTENING", "true").lower() == "true"
    if use_langgraph and feature_enabled:
        try:
            return await _generate_test_data_langgraph(topic, seed, recent_scenarios)
        except Exception:
            return await _generate_test_data_legacy(topic, seed, recent_scenarios)
    return await _generate_test_data_legacy(topic, seed, recent_scenarios)
