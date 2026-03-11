"""
IELTS Reading Agent (RAG).
Uses LangChain, HuggingFace embeddings, and FAISS for local context retrieval.
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.llm import get_llm
import os

# Create embeddings model once to avoid reloading on every request
# all-MiniLM-L6-v2 is fast, tiny, and perfect for CPU retrieval
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Cache to store FAISS vector stores by passage_id
vector_stores: dict[str, FAISS] = {}

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


def load_passage_into_vector_manager(passage_id: str, text: str):
    """
    Chunks the passage and loads it into a FAISS in-memory vector store.
    """
    if passage_id in vector_stores:
        return  # Already loaded

    # Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)

    # Create and cache the vector store
    vectorstore = FAISS.from_texts(chunks, embeddings)
    vector_stores[passage_id] = vectorstore


async def evaluate_reading_answer(passage_id: str, passage_text: str, question: str, user_answer: str) -> dict:
    """
    Retrieves the most relevant chunks from the passage and uses the LLM to verify the user's answer.
    """
    # Ensure it's in the vector store
    if passage_id not in vector_stores:
        load_passage_into_vector_manager(passage_id, passage_text)

    # Retrieve top 2 most relevant chunks
    retriever = vector_stores[passage_id].as_retriever(search_kwargs={"k": 2})
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
