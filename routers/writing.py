"""
Writing test API routes.
"""

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from agents.writing_agent import evaluate_essay
from agents.writing_agent import generate_unique_task1_prompts, generate_unique_task2_prompts, CHART_OUTPUT_DIR
from fastapi import HTTPException
import os

router = APIRouter(prefix="/api/writing", tags=["writing"])

# ---- Schemas ----

class EvaluateRequest(BaseModel):
    task_type: int  # 1 or 2
    prompt_text: str
    essay_text: str

class OcrRequest(BaseModel):
    image_base64: str

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

from agents.writing_agent import extract_text_from_image

@router.post("/upload_image")
async def ocr_essay(req: OcrRequest):
    """Extracts text from an uploaded handwritten essay image using Groq Vision."""
    extracted_text = await extract_text_from_image(req.image_base64)
    return {"extracted_text": extracted_text}

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
    """Returns unique IELTS writing prompts for the frontend."""
    return {
        "task1": generate_unique_task1_prompts(count=3),
        "task2": generate_unique_task2_prompts(count=4),
    }


@router.get("/generated-chart/{filename}")
async def get_generated_chart(filename: str):
    """Serves dynamically generated writing Task 1 chart images."""
    safe_name = os.path.basename(filename)
    file_path = CHART_OUTPUT_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Chart image not found")
    return FileResponse(str(file_path), media_type="image/png")
