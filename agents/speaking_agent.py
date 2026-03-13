"""
IELTS Speaking Examiner Agent.
Simulates a real IELTS speaking test with Parts 1, 2, and 3.
Maintains conversation state and provides examiner-style follow-ups.
"""

import os
import uuid
from functools import lru_cache

from langchain_core.messages import BaseMessage
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from utils.llm import get_llm
from utils.langgraph_runtime import get_langgraph_checkpointer

# ---- System Prompts for each Part ----

PART1_SYSTEM = """You are an official IELTS Speaking Examiner conducting Part 1 of the speaking test.

RULES:
- Ask ONE question at a time. Wait for the candidate's response before asking the next.
- Your examiner name is: {examiner_name}
- Opening style for this session: {opener_style}
- Create a fresh, natural opening each session. Do not reuse fixed wording.
- Introduce yourself briefly in one sentence, then ask for candidate name naturally.
- After the name, ask about familiar topics: home, work/studies, hobbies, daily routine, hometown.
- If candidate profile or context memory exists, use it to choose relevant follow-up questions.
- Ask 4-5 questions total for Part 1.
- Keep your questions natural and conversational.
- Do NOT evaluate or give feedback during the test. Just ask questions.
- Do NOT answer your own questions.
- After 4-5 questions, say: "Thank you. Now let's move on to Part 2."
- Be warm but professional. Use British English.

Candidate profile for personalization:
{candidate_profile}

Context memory from earlier answers:
{context_memory}
"""

PART2_SYSTEM = """You are an official IELTS Speaking Examiner conducting Part 2 of the speaking test.
The core theme for this specific test session is: {topic_seed}.
Examiner name: {examiner_name}
Opening style: {opener_style}

Candidate profile for personalization:
{candidate_profile}

Context memory from earlier answers:
{context_memory}

RULES:
- Give the candidate a cue card topic highly related to the theme: {topic_seed}.
- Format it clearly:
  "I'd like you to describe [topic]. You should say:
   • [point 1]
   • [point 2]
   • [point 3]
   And explain [final point].
   You have 1 minute to prepare and then 1-2 minutes to speak."
- After the candidate speaks, ask 1-2 brief follow-up questions related to the topic.
- Then say: "Thank you. Let's move on to Part 3."
- Do NOT evaluate during the test. Just listen and ask follow-ups.
"""

PART3_SYSTEM = """You are an official IELTS Speaking Examiner conducting Part 3 of the speaking test.
The core theme for this specific test session is: {topic_seed}.
Examiner name: {examiner_name}
Opening style: {opener_style}

Candidate profile for personalization:
{candidate_profile}

Context memory from earlier answers:
{context_memory}

RULES:
- Ask abstract, discussion-style questions related to the theme: {topic_seed}.
- Questions should require the candidate to analyze, compare, speculate, or give opinions on this theme.
- Ask 4-5 questions total.
- Push the candidate to elaborate: "Can you explain what you mean by that?" or "Could you give an example?"
- After 4-5 questions, conclude: "Thank you very much. That is the end of the speaking test."
- Do NOT evaluate during the test.
"""

FEEDBACK_SYSTEM = """You are an expert IELTS Speaking Examiner providing detailed feedback on a candidate's speaking test.

Evaluate the ENTIRE conversation below and provide a structured assessment:

1. **Estimated Band Score** (0-9, can use .5 increments)

2. **Fluency & Coherence** (Band 0-9)
   - Comment on speech flow, hesitations, self-correction, coherence of ideas

3. **Lexical Resource** (Band 0-9)
   - Comment on vocabulary range, precision, use of idiomatic expressions

4. **Grammatical Range & Accuracy** (Band 0-9)
   - Comment on sentence structures, tense usage, error frequency

5. **Pronunciation** (Band 0-9)
   - Note: Based on transcribed text, comment on apparent clarity and word choice indicating pronunciation awareness

6. **Strengths** — List 2-3 specific things the candidate did well with examples from their answers.

7. **Areas for Improvement** — List 2-3 specific weaknesses with actionable advice.

8. **Sample Improved Answer** — Take one of the candidate's weaker answers and rewrite it at Band 8+ level.

Be encouraging but honest. Use specific examples from the conversation.
"""


def get_system_prompt(
    part: int,
    topic_seed: str,
    candidate_profile: list[str] | None,
    context_memory: list[str] | None,
    examiner_name: str,
    opener_style: str,
) -> str:
    """Returns the system prompt for the given IELTS speaking part."""
    profile_text = "\n".join(f"- {line}" for line in (candidate_profile or [])) or "- No profile provided yet."
    memory_text = "\n".join(f"- {line}" for line in (context_memory or [])) or "- No memory captured yet."

    prompts = {
        1: PART1_SYSTEM,
        2: PART2_SYSTEM,
        3: PART3_SYSTEM,
    }
    base = prompts.get(part, PART1_SYSTEM)
    return (
        base.replace("{topic_seed}", topic_seed)
        .replace("{candidate_profile}", profile_text)
        .replace("{context_memory}", memory_text)
        .replace("{examiner_name}", examiner_name)
        .replace("{opener_style}", opener_style)
    )


def build_messages(
    part: int,
    topic_seed: str,
    history: list[dict],
    candidate_profile: list[str] | None,
    context_memory: list[str] | None,
    examiner_name: str,
    opener_style: str,
) -> list:
    """Builds the LangChain message list from conversation history."""
    messages = [
        SystemMessage(
            content=get_system_prompt(
                part,
                topic_seed,
                candidate_profile,
                context_memory,
                examiner_name,
                opener_style,
            )
        )
    ]

    for msg in history:
        if msg["role"] == "examiner":
            messages.append(AIMessage(content=msg["content"]))
        elif msg["role"] == "candidate":
            messages.append(HumanMessage(content=msg["content"]))

    return messages


@lru_cache(maxsize=1)
def _get_speaking_graph():
    llm = get_llm()
    return create_react_agent(model=llm, tools=[], checkpointer=get_langgraph_checkpointer())


async def _examiner_respond_legacy(
    part: int,
    topic_seed: str,
    history: list[dict],
    candidate_name: str | None = None,
    candidate_profile: list[str] | None = None,
    context_memory: list[str] | None = None,
    examiner_name: str = "Sarah",
    opener_style: str = "warm",
) -> str:
    """Legacy speaking response path retained as fallback."""
    llm = get_llm()
    if candidate_name:
        profile_lines = candidate_profile or []
        if not any("Preferred name:" in line for line in profile_lines):
            profile_lines = [f"Preferred name: {candidate_name}", *profile_lines]
        candidate_profile = profile_lines

    messages = build_messages(
        part,
        topic_seed,
        history,
        candidate_profile,
        context_memory,
        examiner_name,
        opener_style,
    )
    response = llm.invoke(messages)
    return response.content


async def _examiner_respond_langgraph(
    part: int,
    topic_seed: str,
    history: list[dict],
    candidate_name: str | None = None,
    candidate_profile: list[str] | None = None,
    context_memory: list[str] | None = None,
    examiner_name: str = "Sarah",
    opener_style: str = "warm",
    session_id: str | None = None,
) -> str:
    """LangGraph speaking response path."""
    if candidate_name:
        profile_lines = candidate_profile or []
        if not any("Preferred name:" in line for line in profile_lines):
            profile_lines = [f"Preferred name: {candidate_name}", *profile_lines]
        candidate_profile = profile_lines

    graph = _get_speaking_graph()
    thread_seed = (session_id or "").strip() or f"speaking-{uuid.uuid4().hex}"
    thread_id = f"{thread_seed}-p{part}-t{len(history)}"

    messages: list[BaseMessage] = build_messages(
        part,
        topic_seed,
        history,
        candidate_profile,
        context_memory,
        examiner_name,
        opener_style,
    )

    result = await graph.ainvoke(
        {"messages": messages},
        config={"configurable": {"thread_id": thread_id}},
    )

    output_messages = result.get("messages", [])
    for msg in reversed(output_messages):
        content = getattr(msg, "content", "")
        if isinstance(msg, AIMessage) and content:
            return content
    return "Could you please continue with your answer?"


async def examiner_respond(
    part: int,
    topic_seed: str,
    history: list[dict],
    candidate_name: str | None = None,
    candidate_profile: list[str] | None = None,
    context_memory: list[str] | None = None,
    examiner_name: str = "Sarah",
    opener_style: str = "warm",
    session_id: str | None = None,
) -> str:
    """
    Generates the examiner's next response based on conversation history.

    Args:
        part: IELTS speaking part (1, 2, or 3)
        topic_seed: Random string theme to anchor Parts 2 and 3
        history: List of {"role": "examiner"|"candidate", "content": "..."}

    Returns:
        The examiner's response text.
    """
    feature_enabled = (os.getenv("ENABLE_LANGGRAPH_SPEAKING", "true").lower() == "true")
    if feature_enabled:
        try:
            return await _examiner_respond_langgraph(
                part=part,
                topic_seed=topic_seed,
                history=history,
                candidate_name=candidate_name,
                candidate_profile=candidate_profile,
                context_memory=context_memory,
                examiner_name=examiner_name,
                opener_style=opener_style,
                session_id=session_id,
            )
        except Exception:
            return await _examiner_respond_legacy(
                part=part,
                topic_seed=topic_seed,
                history=history,
                candidate_name=candidate_name,
                candidate_profile=candidate_profile,
                context_memory=context_memory,
                examiner_name=examiner_name,
                opener_style=opener_style,
            )

    return await _examiner_respond_legacy(
        part=part,
        topic_seed=topic_seed,
        history=history,
        candidate_name=candidate_name,
        candidate_profile=candidate_profile,
        context_memory=context_memory,
        examiner_name=examiner_name,
        opener_style=opener_style,
    )


async def generate_feedback(history: list[dict]) -> str:
    """
    Generates detailed IELTS feedback for the entire speaking test.

    Args:
        history: Full conversation history across all parts.

    Returns:
        Structured feedback string with band scores and advice.
    """
    llm = get_llm()

    # Build a readable transcript for the feedback prompt
    transcript_lines = []
    for msg in history:
        role = "Examiner" if msg["role"] == "examiner" else "Candidate"
        transcript_lines.append(f"{role}: {msg['content']}")
    transcript = "\n".join(transcript_lines)

    messages = [
        SystemMessage(content=FEEDBACK_SYSTEM),
        HumanMessage(content=f"Here is the full speaking test transcript:\n\n{transcript}\n\nPlease provide your detailed assessment."),
    ]

    response = llm.invoke(messages)
    return response.content
