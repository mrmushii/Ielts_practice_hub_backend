"""
Reading test API routes and sample passages.
"""

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import List
from agents.reading_agent import evaluate_reading_answer, generate_reading_test

router = APIRouter(prefix="/api/reading", tags=["reading"])

# ---- Schemas ----

class Question(BaseModel):
    id: str
    text: str
    type: str # "mcq", "tfng" (True/False/Not Given), "fill_blank"

class Passage(BaseModel):
    id: str
    title: str
    text: str
    questions: List[Question]

class AskRequest(BaseModel):
    passage_id: str
    passage_text: str
    question: str
    user_answer: str
    use_langgraph: bool = True

class AskResponse(BaseModel):
    is_correct: bool
    feedback: str
    retrieved_context: str

# ---- Sample Data ----
# A short excerpt for testing the RAG functionality

SAMPLE_PASSAGES = [
    {
        "id": "p1",
        "title": "The History of the Bicycle",
        "text": (
            "The bicycle was invented in the 19th century and has undergone many changes. "
            "The first verifiable claim for a practically used bicycle belongs to German Baron Karl von Drais, "
            "a civil servant to the Grand Duke of Baden in Germany. Drais invented his Laufmaschine (running machine) "
            "of 1817 that was called Draisine by the press. Karl von Drais patented this design in 1818, "
            "which was the first commercially successful two-wheeled, steerable, human-propelled machine, "
            "commonly called a velocipede, and nicknamed hobby-horse or dandy horse. It was initially manufactured "
            "in Germany and France.\n\n"
            "By the 1890s, the \"safety bicycle\" had been developed. This had two wheels of identical size and "
            "a chain drive to the rear wheel. The introduction of the pneumatic tire in 1888 by John Boyd Dunlop "
            "made the ride much smoother. This period, often called the Golden Age of Bicycles, saw a massive boom "
            "in cycling popularity worldwide."
        ),
        "questions": [
            {
                "id": "q1",
                "text": "Who invented the Laufmaschine?",
                "type": "fill_blank"
            },
            {
                "id": "q2",
                "text": "The safety bicycle had different sized wheels. (True / False / Not Given)",
                "type": "tfng"
            },
            {
                "id": "q3",
                "text": "What year was the pneumatic tire introduced?",
                "type": "fill_blank"
            }
        ]
    }
]

# ---- Endpoints ----

@router.get("/generate", response_model=Passage)
async def generate_passage(use_langgraph: bool = Query(default=True)):
    """Dynamically generates a new IELTS reading passage with 3 varied questions."""
    return await generate_reading_test(use_langgraph=use_langgraph)

@router.get("/passages", response_model=List[Passage])
async def get_passages():
    """Returns sample IELTS reading passages."""
    return SAMPLE_PASSAGES

@router.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """Evaluates the user's answer against the passage using RAG."""
    try:
        result = await evaluate_reading_answer(
            passage_id=req.passage_id,
            passage_text=req.passage_text,
            question=req.question,
            user_answer=req.user_answer,
            use_langgraph=req.use_langgraph,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return AskResponse(**result)
