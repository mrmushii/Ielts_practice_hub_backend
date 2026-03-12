"""
Omni-Tutor Agent equipped with Search, Grounding, and RAG tools.
"""

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from utils.llm import get_llm

@tool
def google_search_grounding(query: str) -> str:
    """Use this tool to ground answers using official IELTS facts or rules. 
    It simulates a Google Search for authoritative IELTS information."""
    search = DuckDuckGoSearchRun()
    return search.run(f"site:ielts.org OR site:britishcouncil.org {query}")

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

search_tool = DuckDuckGoSearchRun(
    name="internet_search", 
    description="Use this tool to search the internet for general queries or recent topics."
)

tools = [search_tool, google_search_grounding, search_uploaded_documents]

async def chat_with_tutor(message: str, essay_context: str = None, history: list = None) -> str:
    """Invokes the Omni-Tutor, optionally injecting context and history."""
    llm = get_llm()
    
    system_text = (
        "You are an expert IELTS Omni-Tutor. You MUST use your tools to provide accurate info. "
        "User can upload PDF documents; ALWAYS check 'search_uploaded_documents' if the user mentions a file or asks a complex question about IELTS rules/materials. "
        "Use 'google_search_grounding' for official website-only rules, and 'internet_search' for general news/topics."
    )
    
    if essay_context and essay_context.strip():
        system_text += (
            "\n\nCRITICAL CONTEXT: The user is currently writing an essay in a split-pane Canvas. "
            "DO NOT grade it yet. Focus on live coaching/suggestions for the provided draft below.\n\n"
            f"--- ESSAY DRAFT ---\n{essay_context}\n--------------------"
        )
        
    dynamic_prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, dynamic_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    chat_history = []
    if history:
        for msg in history:
            if msg.get("role") == "user":
                chat_history.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "tutor":
                chat_history.append(AIMessage(content=msg.get("content", "")))
    
    result = await agent_executor.ainvoke({"input": message, "chat_history": chat_history})
    return result["output"]
