"""
IELTS Writing Examiner Agent.
Grading essays based on the 4 official IELTS rubrics.
"""

from typing import List, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from utils.llm import get_llm
import os
import uuid
from functools import lru_cache
from groq import AsyncGroq
from pathlib import Path
import random
from PIL import Image, ImageDraw
from langgraph.graph import StateGraph, START, END
from utils.langgraph_runtime import get_langgraph_checkpointer

# ---- Structured Output Schema ----

class WritingFeedback(BaseModel):
    task_response_score: float = Field(description="Score for Task Achievement / Response (0-9)")
    coherence_score: float = Field(description="Score for Coherence and Cohesion (0-9)")
    lexical_score: float = Field(description="Score for Lexical Resource (0-9)")
    grammar_score: float = Field(description="Score for Grammatical Range and Accuracy (0-9)")
    overall_score: float = Field(description="Overall band score (average of the four, rounded to nearest 0.5)")
    strengths: List[str] = Field(description="List of 2-3 specific strengths in the essay")
    weaknesses: List[str] = Field(description="List of 2-3 specific weaknesses to improve")
    improved_version: str = Field(description="A rewrite of a weak paragraph to show how it can be improved to Band 8+")


# ---- System Prompts ----

WRITING_EVALUATOR_SYSTEM = """You are an expert, official IELTS Writing Examiner.
Your job is to read the candidate's essay and grade it strictly against the 4 official IELTS rubrics:
1. Task Achievement (Task 1) / Task Response (Task 2)
2. Coherence and Cohesion
3. Lexical Resource
4. Grammatical Range and Accuracy

Strictly output your evaluation in the required JSON format.
Be highly accurate and strict. A perfect 9 is extremely rare.
Provide clear, actionable feedback and rewrite a section to demonstrate a Band 8+ standard. 
"""

CHART_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "generated_charts"
CHART_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _draw_axes(draw: ImageDraw.ImageDraw, width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = 80, 60, width - 40, height - 70
    draw.line([(left, top), (left, bottom)], fill=(70, 70, 70), width=2)
    draw.line([(left, bottom), (right, bottom)], fill=(70, 70, 70), width=2)
    return left, top, right, bottom


def _build_bar_chart() -> tuple[str, str]:
    categories = random.sample(["Australia", "Canada", "Japan", "Brazil", "Germany", "India", "Spain"], 4)
    values = [random.randint(40, 160) for _ in categories]
    year = random.choice([2018, 2019, 2020, 2021, 2022, 2023])

    width, height = 900, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = _draw_axes(draw, width, height)

    max_v = max(values)
    bar_area_width = right - left
    bar_width = int(bar_area_width / (len(categories) * 1.8))
    gap = int((bar_area_width - bar_width * len(categories)) / (len(categories) + 1))

    for idx, (cat, value) in enumerate(zip(categories, values)):
        x1 = left + gap + idx * (bar_width + gap)
        x2 = x1 + bar_width
        bar_height = int((value / max_v) * (bottom - top - 20))
        y1 = bottom - bar_height
        draw.rectangle([(x1, y1), (x2, bottom)], fill=(67, 97, 238), outline=(38, 59, 168), width=2)
        draw.text((x1, bottom + 10), cat, fill=(50, 50, 50))
        draw.text((x1 + 8, y1 - 20), str(value), fill=(50, 50, 50))

    title = f"Bar chart: Number of weekly public library visitors in {year}"
    draw.text((80, 20), title, fill=(20, 20, 20))

    filename = f"task1_bar_{uuid.uuid4().hex}.png"
    image.save(CHART_OUTPUT_DIR / filename)

    prompt_text = (
        "The chart below shows the number of weekly visitors to public libraries in four countries "
        f"in {year}. Summarise the information by selecting and reporting the main features, and make "
        "comparisons where relevant."
    )
    return filename, prompt_text


def _build_line_chart() -> tuple[str, str]:
    years = [2019, 2020, 2021, 2022, 2023]
    city_a = [random.randint(20, 40)]
    city_b = [random.randint(30, 50)]
    for _ in years[1:]:
        city_a.append(max(10, city_a[-1] + random.randint(-5, 8)))
        city_b.append(max(10, city_b[-1] + random.randint(-6, 7)))

    width, height = 900, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = _draw_axes(draw, width, height)
    max_v = max(city_a + city_b)
    min_v = min(city_a + city_b)
    spread = max(1, max_v - min_v)

    def project(idx: int, value: int) -> tuple[int, int]:
        x = left + int((idx / (len(years) - 1)) * (right - left))
        y = bottom - int(((value - min_v) / spread) * (bottom - top - 20))
        return x, y

    points_a = [project(i, v) for i, v in enumerate(city_a)]
    points_b = [project(i, v) for i, v in enumerate(city_b)]

    draw.line(points_a, fill=(22, 163, 74), width=4)
    draw.line(points_b, fill=(225, 29, 72), width=4)
    for i, year in enumerate(years):
        x = left + int((i / (len(years) - 1)) * (right - left))
        draw.text((x - 10, bottom + 10), str(year), fill=(50, 50, 50))

    draw.text((80, 20), "Line chart: Average daily cycling commuters (thousands)", fill=(20, 20, 20))
    draw.text((right - 190, top + 10), "Green: City A", fill=(22, 163, 74))
    draw.text((right - 190, top + 30), "Rose: City B", fill=(225, 29, 72))

    filename = f"task1_line_{uuid.uuid4().hex}.png"
    image.save(CHART_OUTPUT_DIR / filename)

    prompt_text = (
        "The line graph compares the average number of daily cycling commuters (in thousands) "
        "in two cities between 2019 and 2023. Summarise the information by selecting and reporting "
        "the main features, and make comparisons where relevant."
    )
    return filename, prompt_text


def _build_pie_chart() -> tuple[str, str]:
    labels = ["Housing", "Food", "Transport", "Leisure", "Other"]
    values = [random.randint(10, 35) for _ in labels]
    total = sum(values)
    values = [max(5, int(v * 100 / total)) for v in values]
    values[-1] = 100 - sum(values[:-1])

    width, height = 900, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    center = (300, 260)
    radius = 170
    colors = [(59, 130, 246), (245, 158, 11), (16, 185, 129), (239, 68, 68), (124, 58, 237)]

    start = 0.0
    for i, (label, value) in enumerate(zip(labels, values)):
        end = start + (value / 100) * 360
        draw.pieslice(
            [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius],
            start=start,
            end=end,
            fill=colors[i],
            outline="white",
            width=2,
        )
        draw.rectangle([(560, 120 + i * 52), (580, 140 + i * 52)], fill=colors[i])
        draw.text((590, 120 + i * 52), f"{label}: {value}%", fill=(40, 40, 40))
        start = end

    draw.text((80, 20), "Pie chart: Average monthly household spending by category", fill=(20, 20, 20))
    filename = f"task1_pie_{uuid.uuid4().hex}.png"
    image.save(CHART_OUTPUT_DIR / filename)

    prompt_text = (
        "The pie chart illustrates the average monthly household expenditure by category in a medium-sized city. "
        "Summarise the information by selecting and reporting the main features, and make comparisons where relevant."
    )
    return filename, prompt_text


def generate_unique_task1_prompts(count: int = 3) -> list[dict]:
    builders = [_build_bar_chart, _build_line_chart, _build_pie_chart]
    random.shuffle(builders)
    prompts: list[dict] = []
    for i in range(min(count, len(builders))):
        filename, prompt_text = builders[i]()
        prompts.append(
            {
                "id": f"t1_dynamic_{i + 1}_{uuid.uuid4().hex[:6]}",
                "text": prompt_text,
                "image_url": f"http://localhost:8000/api/writing/generated-chart/{filename}",
            }
        )
    return prompts


def generate_unique_task2_prompts(count: int = 4) -> list[dict]:
    templates = [
        "Some people think {topic_a} should be taught as a compulsory subject in secondary schools, while others believe students should choose freely. Discuss both views and give your opinion.",
        "In many countries, {topic_a} is becoming more common. What are the reasons for this trend, and is it a positive or negative development?",
        "Governments should spend more money on {topic_a} than on {topic_b}. To what extent do you agree or disagree?",
        "Some people say that advances in {topic_a} have improved our quality of life, while others think they have created new problems. Discuss both views and give your opinion.",
        "Young people today spend too much time on {topic_a}. What problems can this cause, and what solutions can you suggest?",
    ]
    topics = [
        "public transport",
        "online education",
        "artificial intelligence tools",
        "social media",
        "renewable energy",
        "urban green spaces",
        "remote working",
        "tourism",
        "sports facilities",
        "digital payments",
    ]

    prompts: list[dict] = []
    for i in range(count):
        template = random.choice(templates)
        a, b = random.sample(topics, 2)
        text = template.format(topic_a=a, topic_b=b)
        prompts.append({"id": f"t2_dynamic_{i + 1}_{uuid.uuid4().hex[:6]}", "text": text})
    return prompts

async def extract_text_from_image(base64_image: str) -> str:
    """Uses Groq Vision (llama-3.2-11b-vision-preview) to extract text from a base64 encoded image."""
    client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
    
    prompt = "Read and extract all the text from this handwritten or typed essay image. Provide ONLY the exact text as it appears. Do not add any conversational text or formatting outside of the essay content."
    
    response = await client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    
    return response.choices[0].message.content

class WritingGraphState(TypedDict, total=False):
    task_type: int
    prompt_text: str
    essay_text: str
    feedback: WritingFeedback


def _evaluate_structured_node(state: WritingGraphState) -> WritingGraphState:
    llm = get_llm()
    structured_llm = llm.with_structured_output(WritingFeedback)
    task_type = state.get("task_type", 2)
    task_name = "Task 1 (Report/Letter)" if task_type == 1 else "Task 2 (Essay)"

    user_prompt = f"""Evaluate the following IELTS {task_name}.

**The Prompt:**
{state.get("prompt_text", "")}

**The Candidate's Essay:**
{state.get("essay_text", "")}

Provide your detailed scoring and feedback.
"""

    messages = [
        SystemMessage(content=WRITING_EVALUATOR_SYSTEM),
        HumanMessage(content=user_prompt),
    ]
    feedback = structured_llm.invoke(messages)
    return {"feedback": feedback}


@lru_cache(maxsize=1)
def _get_writing_graph():
    builder = StateGraph(WritingGraphState)
    builder.add_node("evaluate_structured", _evaluate_structured_node)
    builder.add_edge(START, "evaluate_structured")
    builder.add_edge("evaluate_structured", END)
    return builder.compile(checkpointer=get_langgraph_checkpointer())


async def _evaluate_essay_langgraph(task_type: int, prompt_text: str, essay_text: str) -> dict:
    graph = _get_writing_graph()
    thread_id = f"writing-{uuid.uuid4().hex}"
    result = await graph.ainvoke(
        {
            "task_type": task_type,
            "prompt_text": prompt_text,
            "essay_text": essay_text,
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    feedback = result.get("feedback")
    if not isinstance(feedback, WritingFeedback):
        raise ValueError("LangGraph writing evaluation returned invalid payload")
    return feedback.model_dump()


async def _evaluate_essay_legacy(task_type: int, prompt_text: str, essay_text: str) -> dict:
    """
    Evaluates an IELTS essay using Groq strictly returning structured JSON.
    
    Args:
        task_type: 1 or 2
        prompt_text: The IELTS writing prompt the candidate is answering
        essay_text: The essay written by the candidate
        
    Returns:
        Dict matching WritingFeedback schema
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(WritingFeedback)
    
    task_name = "Task 1 (Report/Letter)" if task_type == 1 else "Task 2 (Essay)"
    
    user_prompt = f"""Evaluate the following IELTS {task_name}.

**The Prompt:**
{prompt_text}

**The Candidate's Essay:**
{essay_text}

Provide your detailed scoring and feedback.
"""

    messages = [
        SystemMessage(content=WRITING_EVALUATOR_SYSTEM),
        HumanMessage(content=user_prompt)
    ]
    
    feedback: WritingFeedback = structured_llm.invoke(messages)
    return feedback.model_dump()


async def evaluate_essay(
    task_type: int,
    prompt_text: str,
    essay_text: str,
    use_langgraph: bool = True,
) -> dict:
    """Evaluates an IELTS essay with LangGraph primary path and legacy fallback."""
    feature_enabled = os.getenv("ENABLE_LANGGRAPH_WRITING", "true").lower() == "true"
    if use_langgraph and feature_enabled:
        try:
            return await _evaluate_essay_langgraph(task_type, prompt_text, essay_text)
        except Exception:
            return await _evaluate_essay_legacy(task_type, prompt_text, essay_text)
    return await _evaluate_essay_legacy(task_type, prompt_text, essay_text)
