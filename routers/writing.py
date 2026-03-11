"""
Writing test API routes.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from agents.writing_agent import evaluate_essay

router = APIRouter(prefix="/api/writing", tags=["writing"])

# ---- Schemas ----

class EvaluateRequest(BaseModel):
    task_type: int  # 1 or 2
    prompt_text: str
    essay_text: str

class EvaluateResponse(BaseModel):
    task_response_score: float
    coherence_score: float
    lexical_score: float
    grammar_score: float
    overall_score: float
    strengths: List[str]
    weaknesses: List[str]
    improved_version: str

# ---- Endpoints ----

@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest):
    """Grading an essay strictly relying on structured JSON output from LLM."""
    feedback = await evaluate_essay(
        task_type=req.task_type,
        prompt_text=req.prompt_text,
        essay_text=req.essay_text
    )
    return EvaluateResponse(**feedback)

@router.get("/prompts")
async def get_sample_prompts():
    """Returns sample IELTS writing prompts for the frontend."""
    return {
        "task1": [
            {
                "id": "t1_1",
                "text": "The chart below shows the number of men and women in further education in Britain in three periods and whether they were studying full-time or part-time. Summarise the information by selecting and reporting the main features, and make comparisons where relevant."
            },
            {
                "id": "t1_2",
                "text": "The maps below show the centre of a small town called Islip as it is now, and plans for its development. Summarise the information by selecting and reporting the main features, and make comparisons where relevant."
            }
        ],
        "task2": [
            {
                "id": "t2_1",
                "text": "Some people believe that unpaid community service should be a compulsory part of high school programmes. To what extent do you agree or disagree?"
            },
            {
                "id": "t2_2",
                "text": "Many manufactured food and drink products contain high levels of sugar, which causes many health problems. Sugary products should be made more expensive to encourage people to consume less sugar. Do you agree or disagree?"
            }
        ]
    }
