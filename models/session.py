"""
Data models for the IELTS speaking sessions.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ChatMessage(BaseModel):
    role: str # 'examiner' or 'candidate'
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class SpeakingSession(BaseModel):
    session_id: str
    part: int = 1
    voice: str = "british_female"
    topic_seed: str = "general"
    history: List[ChatMessage] = []
    feedback: Optional[str] = None
    is_complete: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
