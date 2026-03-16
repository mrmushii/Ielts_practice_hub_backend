from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import shutil
import traceback
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch

router = APIRouter(prefix="/api/documents", tags=["documents"])

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a PDF, chunks it, and embeds it into MongoDB."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    os.makedirs("temp_uploads", exist_ok=True)
    temp_path = f"temp_uploads/{file.filename}"
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Document embeddings are disabled in this deployment. "
                    "Install optional packages: langchain-huggingface and sentence-transformers."
                ),
            ) from exc

        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        uri = os.getenv("MONGODB_URI") or "mongodb://localhost:27017"
        mongo_client = MongoClient(uri)
        collection = mongo_client["ielts_platform"]["vector_index"]
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        vectorstore = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name="vector_index"
        )
        
        vectorstore.add_documents(chunks)
        
        return {"status": "success", "chunks_added": len(chunks), "filename": file.filename}
        
    except Exception as e:
        full_traceback = traceback.format_exc()
        print(f"Error uploading document: {full_traceback}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
