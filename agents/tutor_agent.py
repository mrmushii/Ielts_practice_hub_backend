"""
Omni-Tutor Agent equipped with Search, Grounding, and RAG tools.
"""

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from utils.llm import get_llm

@tool
def google_search_grounding(query: str) -> str:
    """Use this tool to ground answers using official IELTS facts or rules. 
    It simulates a Google Search for authoritative IELTS information."""
    search = DuckDuckGoSearchRun()
    return search.run(f"site:ielts.org OR site:britishcouncil.org {query}")

@tool
def ielts_material_retriever(query: str) -> str:
    """Use this to search through the student's available reading materials in the database."""
    from agents.reading_agent import get_vector_collection
    from langchain_mongodb import MongoDBAtlasVectorSearch
    from langchain_huggingface import HuggingFaceEmbeddings
    
    collection = get_vector_collection()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="vector_index"
    )
    docs = vectorstore.similarity_search(query, k=2)
    return "\n".join([doc.page_content for doc in docs])

search_tool = DuckDuckGoSearchRun(
    name="internet_search", 
    description="Use this tool to search the internet for general queries or recent topics."
)

tools = [search_tool, google_search_grounding, ielts_material_retriever]

async def chat_with_tutor(message: str, essay_context: str = None) -> str:
    """Invokes the Omni-Tutor, optionally injecting the user's current writing draft."""
    llm = get_llm()
    
    system_text = (
        "You are an expert IELTS Omni-Tutor. You help students prepare for their IELTS exam. "
        "You MUST use your tools to provide accurate info. Use 'google_search_grounding' for official rules. "
        "Use 'internet_search' for general or recent topics. Use 'ielts_material_retriever' for reading passages."
    )
    
    if essay_context and essay_context.strip():
        system_text += (
            "\n\nCRITICAL CONTEXT: The user is currently writing an essay in a split-pane Canvas next to this chat. "
            "Because this is an interactive session, DO NOT grade the essay or give a band score yet. "
            "Instead, read their current draft and act as a live coach. Focus on giving them 1 or 2 specific, actionable pieces of "
            "advice (e.g., suggesting a better vocabulary word, pointing out a run-on sentence, or advising on essay structure). "
            "Keep your responses concise, encouraging, and directly relevant to the text they have written so far.\n\n"
            f"--- THE USER'S CURRENT ESSAY DRAFT ---\n{essay_context}\n--------------------------------------"
        )
        
    dynamic_prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, dynamic_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    result = await agent_executor.ainvoke({"input": message, "chat_history": []})
    return result["output"]
