"""
Omni-Tutor Agent equipped with Search, Grounding, and RAG tools.
"""

import os
import uuid
import json
import re
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from functools import lru_cache

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import create_react_agent
from utils.llm import get_llm
from utils.langgraph_runtime import get_langgraph_checkpointer


_fallback_search = DuckDuckGoSearchRun()


@tool
def current_datetime() -> str:
    """Use this tool for current date, current year, time, day, month, timezone-independent UTC timestamp questions."""
    now = datetime.utcnow()
    return (
        f"UTC date: {now.strftime('%Y-%m-%d')}\n"
        f"UTC year: {now.strftime('%Y')}\n"
        f"UTC time: {now.strftime('%H:%M:%S')}\n"
        f"UTC weekday: {now.strftime('%A')}"
    )


def _google_custom_search(query: str, site_restrict: str | None = None, num_results: int = 5) -> str:
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY", "").strip()
    engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID", "").strip()
    strict_mode = os.getenv("GOOGLE_SEARCH_STRICT", "true").strip().lower() in {"1", "true", "yes", "on"}

    search_query = query.strip()
    if site_restrict:
        search_query = f"site:{site_restrict} {search_query}"

    if not api_key or not engine_id:
        if strict_mode:
            return "GOOGLE_SEARCH_ERROR: Missing GOOGLE_SEARCH_API_KEY or GOOGLE_SEARCH_ENGINE_ID."
        fallback = _fallback_search.run(search_query)
        return "Google API keys are not configured. Fallback search result:\n\n" + fallback

    params = {
        "key": api_key,
        "cx": engine_id,
        "q": search_query,
        "num": max(1, min(num_results, 10)),
    }
    url = "https://www.googleapis.com/customsearch/v1?" + urlencode(params)

    try:
        req = Request(url, headers={"User-Agent": "ielts-platform-tutor/1.0"})
        with urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as err:
        error_body = ""
        try:
            error_body = err.read().decode("utf-8")
        except Exception:
            error_body = ""

        if "API_KEY_INVALID" in error_body or "API key not valid" in error_body:
            return "GOOGLE_SEARCH_ERROR: API_KEY_INVALID. Use a valid Google API key (starts with AIza...)."

        if strict_mode:
            return f"GOOGLE_SEARCH_ERROR: Google API request failed ({err.code})."

        fallback = _fallback_search.run(search_query)
        return f"Google Search failed ({err}). Fallback search result:\n\n{fallback}"
    except (URLError, TimeoutError) as err:
        if strict_mode:
            return f"GOOGLE_SEARCH_ERROR: Network failure ({err})."
        fallback = _fallback_search.run(search_query)
        return f"Google Search failed ({err}). Fallback search result:\n\n{fallback}"

    items = data.get("items", [])
    if not items:
        return "No relevant results found from Google Search."

    lines = []
    for idx, item in enumerate(items, start=1):
        title = item.get("title", "Untitled")
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        lines.append(f"{idx}. {title}\nURL: {link}\nSnippet: {snippet}")

    return "\n\n".join(lines)

@tool
def google_search_grounding(query: str) -> str:
    """Use this tool to ground answers using official IELTS facts or rules. 
    Uses real Google Search (Custom Search API) for authoritative IELTS information."""
    if not query or not query.strip():
        return "Error: query cannot be empty."

    ielts_org = _google_custom_search(query=query, site_restrict="ielts.org", num_results=4)
    british_council = _google_custom_search(query=query, site_restrict="britishcouncil.org", num_results=4)
    return f"Official grounding results from IELTS.org:\n\n{ielts_org}\n\nOfficial grounding results from British Council:\n\n{british_council}"

@tool
def search_uploaded_documents(query: str) -> str:
    """Search for content across any PDF documents or textbooks uploaded by the user. 
    USE THIS TOOL whenever the user asks a question about their attached files.
    Input 'query' must be specific keywords or questions derived from the user message. 
    Do NOT pass an empty string."""
    if not query or not query.strip():
        return "Error: Search query cannot be empty. Please provide specific keywords to search in your documents."
        
    from pymongo import MongoClient
    import os
    from langchain_mongodb import MongoDBAtlasVectorSearch
    from langchain_huggingface import HuggingFaceEmbeddings
    
    uri = os.getenv("MONGODB_URI") or "mongodb://localhost:27017"
    client = MongoClient(uri)
    # Ensure we use the correct collection name consistently
    collection = client["ielts_platform"]["vector_index"]
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="vector_index"
    )
    
    docs = vectorstore.similarity_search(query, k=3)
    
    if not docs:
        return "No relevant information found in your uploaded documents. Perhaps the document doesn't contain that specific info, or you haven't uploaded one yet."
    
    content = "\n\n".join([doc.page_content for doc in docs])
    return f"Retrieved from documents:\n\n{content}"

@tool
def internet_search(query: str) -> str:
    """Use this tool for real-time internet search on general queries and current topics."""
    if not query or not query.strip():
        return "Error: query cannot be empty."
    return _google_custom_search(query=query, site_restrict=None, num_results=6)

tools = [current_datetime, internet_search, google_search_grounding, search_uploaded_documents]


def _build_system_prompt(essay_context: str | None = None) -> str:
    system_text = (
        "You are an expert IELTS Omni-Tutor. You MUST use your tools to provide accurate info. "
        "User can upload PDF documents; ALWAYS check 'search_uploaded_documents' if the user mentions a file or asks a complex question about IELTS rules/materials. "
        "Use 'google_search_grounding' for official website-only rules, and 'internet_search' for real-time Google Search on general news/topics. "
        "For any question about current date/year/time, you MUST call 'current_datetime' tool first. "
        "Respond in a professional tutoring style with clear structure. "
        "Prefer short sections, concise bullet points, and direct actionable advice. "
        "Avoid vague or generic responses. "
        "When giving examples, keep them IELTS-focused and practical."
    )

    if essay_context and essay_context.strip():
        system_text += (
            "\n\nCRITICAL CONTEXT: The user is currently writing an essay in a split-pane Canvas. "
            "DO NOT grade it yet. Focus on live coaching/suggestions for the provided draft below.\n\n"
            f"--- ESSAY DRAFT ---\n{essay_context}\n--------------------"
        )
    return system_text


def _build_chat_history(history: list | None) -> list[BaseMessage]:
    chat_history: list[BaseMessage] = []
    if history:
        for msg in history:
            if msg.get("role") == "user":
                chat_history.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "tutor":
                chat_history.append(AIMessage(content=msg.get("content", "")))
    return chat_history


@lru_cache(maxsize=1)
def _get_tutor_graph():
    llm = get_llm()
    return create_react_agent(model=llm, tools=tools, checkpointer=get_langgraph_checkpointer())


async def _chat_with_tutor_legacy(message: str, essay_context: str = None, history: list = None) -> str:
    """Legacy tutor execution path retained as fallback."""
    llm = get_llm()
    system_text = _build_system_prompt(essay_context)
        
    dynamic_prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, dynamic_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    chat_history = _build_chat_history(history)
    
    result = await agent_executor.ainvoke({"input": message, "chat_history": chat_history})
    return result["output"]


async def _chat_with_tutor_langgraph(
    message: str,
    essay_context: str = None,
    history: list = None,
    session_id: str | None = None,
) -> str:
    """LangGraph tutor execution path."""
    graph = _get_tutor_graph()
    thread_id = (session_id or "").strip() or f"tutor-{uuid.uuid4().hex}"
    system_text = _build_system_prompt(essay_context)

    messages: list[BaseMessage] = []
    messages.append(HumanMessage(content=f"SYSTEM CONTEXT:\n{system_text}"))
    messages.extend(_build_chat_history(history))
    messages.append(HumanMessage(content=message))

    result = await graph.ainvoke(
        {"messages": messages},
        config={"configurable": {"thread_id": thread_id}},
    )

    output_messages = result.get("messages", [])
    for msg in reversed(output_messages):
        content = getattr(msg, "content", "")
        if isinstance(msg, AIMessage) and content:
            return content
    return "I could not generate a response this turn. Please try again."


async def chat_with_tutor(
    message: str,
    essay_context: str = None,
    history: list = None,
    session_id: str | None = None,
    use_langgraph: bool | None = True,
) -> str:
    """Invokes Omni-Tutor with LangGraph primary path and legacy fallback."""
    feature_enabled = (os.getenv("ENABLE_LANGGRAPH_TUTOR", "true").lower() == "true")
    should_use_langgraph = bool(use_langgraph) and feature_enabled

    if should_use_langgraph:
        try:
            return await _chat_with_tutor_langgraph(
                message=message,
                essay_context=essay_context,
                history=history,
                session_id=session_id,
            )
        except Exception:
            return await _chat_with_tutor_legacy(message, essay_context=essay_context, history=history)

    return await _chat_with_tutor_legacy(message, essay_context=essay_context, history=history)


def suggest_tutor_actions(message: str) -> dict:
    """Maps user intent to safe, allowlisted frontend actions for one-stop tutor routing."""
    lowered = (message or "").strip().lower()

    module_rules = [
        ("speaking", ["speaking", "speak", "oral", "mock interview", "examiner"]),
        ("listening", ["listening", "listen", "audio", "hearing"]),
        ("reading", ["reading", "read", "passage", "comprehension"]),
        ("writing", ["writing", "write", "essay", "task 1", "task 2"]),
    ]

    def has_phrase(phrases: list[str]) -> bool:
        return any(phrase in lowered for phrase in phrases)

    def build_nav_action(module: str) -> dict:
        route = f"/{module}"
        return {
            "id": f"open-{module}",
            "type": "navigate_module",
            "module": module,
            "route": route,
            "label": f"Open {module.capitalize()} Practice",
            "description": f"Go to the {module.capitalize()} test workspace.",
            "requires_confirmation": True,
        }

    matched_module = None
    for module, phrases in module_rules:
        if has_phrase(phrases):
            matched_module = module
            break

    control_phrases = [
        "one stop",
        "control",
        "dashboard",
        "workspace",
        "home",
        "navigate",
        "redirect",
        "take me",
        "open",
        "go to",
    ]
    wants_control = has_phrase(control_phrases)

    explicit_practice = bool(re.search(r"\b(practice|start|begin|take)\b", lowered))

    actions: list[dict] = []
    intent = "chat"
    confidence = 0.3
    reason = "General tutoring request"

    if matched_module and (explicit_practice or wants_control):
        actions.append(build_nav_action(matched_module))
        intent = "suggest_practice"
        confidence = 0.95
        reason = f"Detected practice/navigation request for {matched_module}."
    elif matched_module:
        actions.append(build_nav_action(matched_module))
        intent = "suggest_practice"
        confidence = 0.75
        reason = f"Detected module-specific topic: {matched_module}."
    elif "tutor" in lowered and has_phrase(["open", "workspace", "full"]):
        actions.append(
            {
                "id": "open-tutor-workspace",
                "type": "open_tutor_workspace",
                "module": "tutor",
                "route": "/tutor",
                "label": "Open Tutor Workspace",
                "description": "Use the full-screen tutor control center.",
                "requires_confirmation": True,
            }
        )
        intent = "navigate"
        confidence = 0.85
        reason = "Detected explicit request to open tutor workspace."
    elif wants_control:
        actions.extend([build_nav_action("speaking"), build_nav_action("writing"), build_nav_action("reading"), build_nav_action("listening")])
        intent = "show_help"
        confidence = 0.8
        reason = "Detected dashboard/control request, suggesting all practice modules."

    return {
        "intent": intent,
        "confidence": confidence,
        "reason": reason,
        "actions": actions,
    }
