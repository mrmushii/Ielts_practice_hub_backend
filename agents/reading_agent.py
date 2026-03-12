"""
IELTS Reading Agent (RAG).
Uses LangChain, HuggingFace embeddings, and MongoDB Atlas for vector context retrieval.
"""

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List
from pymongo import MongoClient
from utils.llm import get_llm
import os
import uuid

# ---- Generation Schemas ----

class GeneratedQuestion(BaseModel):
    id: str = Field(description="A unique generic ID like q1, q2, q3")
    text: str = Field(description="The question text to ask the student")
    type: str = Field(description="Must be one of: 'mcq', 'tfng', 'fill_blank'")

class GeneratedPassage(BaseModel):
    id: str = Field(description="A unique random string ID for this passage")
    title: str = Field(description="A formal academic title for the passage")
    text: str = Field(description="A 300-400 word dense academic passage suitable for IELTS reading practice")
    questions: List[GeneratedQuestion] = Field(description="Exactly 3 diverse reading comprehension questions based ONLY on the passage text")

# Create embeddings model once to avoid reloading on every request
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# We use a synchronous pymongo client for LangChain's VectorStore compatibility
_mongo_client = None
def get_vector_collection():
    global _mongo_client
    if _mongo_client is None:
        uri = os.getenv("MONGODB_URI") or "mongodb://localhost:27017"
        _mongo_client = MongoClient(uri)
    return _mongo_client["ielts_platform"]["reading_chunks"]


READING_EVALUATOR_PROMPT = """You are an expert IELTS Reading Examiner.
Use the provided CONTEXT (which is an excerpt from the reading passage) to evaluate the user's ANSWER to the QUESTION.

RULES:
1. IF the user's answer is correct based on the context, say "CORRECT".
2. IF the user's answer is incorrect based on the context, say "INCORRECT".
3. Provide a brief, encouraging explanation (2-3 sentences max) explaining WHY, quoting the specific relevant text from the context.
4. DO NOT use outside knowledge. Only use the provided CONTEXT.

CONTEXT:
{context}

QUESTION:
{question}

USER'S ANSWER:
{user_answer}

YOUR EVALUATION (Start with CORRECT or INCORRECT, then space, then explanation):
"""

READING_GENERATOR_PROMPT = """You are an expert Cambridge IELTS Reading Test creator.
Generate a completely original, dense, university-level academic reading passage.
The topic should be randomly selected from common IELTS themes (e.g., Space Exploration, Paleontology, Marine Biology, Ancient Civilizations, Cognitive Psychology).

Then, generate EXACTLY 3 questions based strictley on the passage you wrote.
One must be 'tfng' (True/False/Not Given).
One must be 'mcq' (Multiple Choice). For MCQ just write the question and the options A, B, C, D in the text field.
One must be 'fill_blank' (Fill in the blanks).

Output strictly matching the requested JSON schema.
"""

async def generate_reading_test() -> dict:
    """Generates a dynamic IELTS reading passage with questions."""
    llm = get_llm()
    structured_llm = llm.with_structured_output(GeneratedPassage)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", READING_GENERATOR_PROMPT),
        ("user", "Generate a new IELTS reading passage and 3 questions now.")
    ])
    
    chain = prompt | structured_llm
    result: GeneratedPassage = chain.invoke({})
    
    # Ensure ID is truly unique for MongoDB
    result.id = f"passage_{uuid.uuid4().hex[:8]}"
    return result.model_dump()


def load_passage_into_vector_manager(passage_id: str, text: str):
    """
    Chunks the passage and loads it into MongoDB Atlas Vector Search if it doesn't already exist.
    """
    collection = get_vector_collection()
    
    # Check if this passage is already embedded to avoid duplicate chunks
    if collection.find_one({"passage_id": passage_id}):
        return

    # Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    
    # We add metadata so we can filter by passage_id later
    metadatas = [{"passage_id": passage_id} for _ in chunks]

    # Create and cache the vector store
    MongoDBAtlasVectorSearch.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection=collection,
        metadatas=metadatas,
        index_name="vector_index"
    )


async def evaluate_reading_answer(passage_id: str, passage_text: str, question: str, user_answer: str) -> dict:
    """
    Retrieves the most relevant chunks from the passage and uses the LLM to verify the user's answer.
    """
    # Ensure it's in the vector store
    load_passage_into_vector_manager(passage_id, passage_text)

    # Initialize the vector store connection
    collection = get_vector_collection()
    vectorstore = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="vector_index"
    )

    # Retrieve top 2 most relevant chunks for this specific passage
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 2, 
            "pre_filter": {"passage_id": {"$eq": passage_id}}
        }
    )
    docs = retriever.invoke(question)
    
    # Combine chunks into single context string
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build the prompt and call LLM
    prompt = ChatPromptTemplate.from_template(READING_EVALUATOR_PROMPT)
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()

    response_text = chain.invoke({
        "context": context,
        "question": question,
        "user_answer": user_answer
    })

    # Parse response format: "CORRECT/INCORRECT. [Explanation]"
    is_correct = response_text.strip().upper().startswith("CORRECT")
    
    return {
        "is_correct": is_correct,
        "feedback": response_text,
        "retrieved_context": context
    }
