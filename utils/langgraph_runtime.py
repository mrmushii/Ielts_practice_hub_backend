"""
Shared LangGraph runtime utilities.
Provides a unified checkpointer factory with in-memory default and optional MongoDB persistence.
"""

from functools import lru_cache
import os

from langgraph.checkpoint.memory import MemorySaver


@lru_cache(maxsize=1)
def get_langgraph_checkpointer():
    """Returns a LangGraph checkpointer based on environment settings."""
    mode = os.getenv("LANGGRAPH_CHECKPOINTER", "memory").lower().strip()
    if mode == "mongodb":
        try:
            # Optional dependency path; fallback to memory if unavailable.
            from langgraph.checkpoint.mongodb import MongoDBSaver  # type: ignore

            uri = os.getenv("MONGODB_URI") or "mongodb://localhost:27017"
            db_name = os.getenv("LANGGRAPH_MONGO_DB", "ielts_platform")
            collection = os.getenv("LANGGRAPH_MONGO_COLLECTION", "langgraph_checkpoints")
            return MongoDBSaver.from_conn_string(uri, db_name=db_name, collection_name=collection)
        except Exception:
            return MemorySaver()
    return MemorySaver()
