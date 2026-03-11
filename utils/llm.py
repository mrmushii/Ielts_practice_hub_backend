"""
Centralized LLM factory — all agents import from here.
Uses Groq Cloud's OpenAI-compatible endpoint.
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

_llm_instance: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    """Returns a cached ChatOpenAI instance configured for Groq Cloud."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            temperature=0.7,
            max_tokens=2048,
            use_responses_api=False,
        )
    return _llm_instance
