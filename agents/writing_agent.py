"""
IELTS Writing Examiner Agent.
Grading essays based on the 4 official IELTS rubrics.
"""

from typing import List
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from utils.llm import get_llm

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

async def evaluate_essay(task_type: int, prompt_text: str, essay_text: str) -> dict:
    """
    Evaluates an IELTS essay using Groq strictly returning structured JSON.
    
    Args:
        task_type: 1 or 2
        prompt_text: The IELTS writing prompt the candidate is answering
        essay_text: The essay written by the candidate
        
    Returns:
        Dict matching WritingFeedback schema
    """
    # Create the structured LLM caller
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
    
    # Generate the structured response
    feedback: WritingFeedback = structured_llm.invoke(messages)
    
    return feedback.model_dump()
