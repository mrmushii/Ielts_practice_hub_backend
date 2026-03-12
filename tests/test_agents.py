import pytest
import os
from unittest.mock import patch, MagicMock
from agents.writing_agent import evaluate_essay, WritingFeedback
from pydantic import ValidationError

@pytest.fixture
def sample_writing_task2_prompt():
    return "Some people believe that unpaid community service should be a compulsory part of high school programmes. To what extent do you agree or disagree?"

@pytest.fixture
def sample_writing_task2_essay():
    return "I completely agree that unpaid community service should be a mandatory part of the high school curriculum. Firstly, it teaches students valuable life skills and empathy. Secondly, it helps the local community. For example, cleaning up local parks teaches environmental responsibility. In conclusion, mandatory volunteering is highly beneficial for both the student and society."

@pytest.mark.asyncio
@patch('agents.writing_agent.get_llm')
async def test_evaluate_essay_valid_schema(mock_get_llm, sample_writing_task2_prompt, sample_writing_task2_essay):
    # Mock the LLM structured output to return a predetermined valid dictionary matching WritingFeedback
    mock_llm_instance = MagicMock()
    mock_structured_llm = MagicMock()
    
    expected_feedback = WritingFeedback(
        task_response_score=7.0,
        coherence_score=6.5,
        lexical_score=7.0,
        grammar_score=6.0,
        overall_score=6.5,
        strengths=["Clear opinion presented", "Good examples used"],
        weaknesses=["Some grammatical errors", "Could use more advanced vocabulary"],
        improved_version="I wholeheartedly concur that unpaid community service ought to be an obligatory component of the high school curriculum."
    )
    
    mock_structured_llm.invoke.return_value = expected_feedback
    mock_llm_instance.with_structured_output.return_value = mock_structured_llm
    mock_get_llm.return_value = mock_llm_instance

    # Call the actual agent function
    result = await evaluate_essay(task_type=2, prompt_text=sample_writing_task2_prompt, essay_text=sample_writing_task2_essay)
    
    # Assertions
    assert isinstance(result, dict)
    assert "overall_score" in result
    assert result["overall_score"] == 6.5
    assert len(result["strengths"]) == 2
    assert "improved_version" in result

@pytest.mark.asyncio
async def test_writing_schema_validation():
    # Test that Pydantic properly validates and catches bad LLM outputs
    with pytest.raises(ValidationError):
        WritingFeedback(
            task_response_score="seven",  # Should be float
            coherence_score=6.5,
            lexical_score=7.0,
            grammar_score=6.0,
            overall_score=6.5,
            strengths=["Good"],
            weaknesses=["Bad"],
            improved_version="Improved."
        )

@pytest.mark.asyncio
async def test_evaluate_empty_essay():
    # It should still return the structure even if the essay is poor or empty
    # This invokes the real LLM endpoint, assuming GROQ_API_KEY is set in environment during local testing
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set in environment, skipping live API test.")
        
    result = await evaluate_essay(task_type=2, prompt_text="Do you agree with X?", essay_text="")
    
    assert isinstance(result, dict)
    assert "overall_score" in result
    assert "strengths" in result
    # An empty essay should score very poorly
    assert result["overall_score"] < 4.0
