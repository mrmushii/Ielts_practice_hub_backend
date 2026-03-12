"""
Omni-Tutor Agent equipped with Search, Grounding, and RAG tools.
"""

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
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

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert IELTS Omni-Tutor. You help students prepare for their IELTS exam. "
               "You MUST use your tools to provide accurate info. Use 'google_search_grounding' for official rules. "
               "Use 'internet_search' for general or recent topics. Use 'ielts_material_retriever' for reading passages."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

async def chat_with_tutor(message: str) -> str:
    llm = get_llm()
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    result = await agent_executor.ainvoke({"input": message, "chat_history": []})
    return result["output"]
