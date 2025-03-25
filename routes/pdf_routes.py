from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends, Header
from fastapi.responses import JSONResponse
import os
import chromadb
from smart_pdf_reader.main import PDFProcessor, PDFSearcher
from smart_pdf_reader.ai_agent import PDFAIAgent
from typing import Optional
from functools import lru_cache
from pydantic import BaseSettings
import uuid

class Settings(BaseSettings):
    openai_api_key: str

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

@lru_cache()
def get_ai_agent(settings: Settings = Depends(get_settings)):
    return PDFAIAgent(settings.openai_api_key)

router = APIRouter()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "extracted_content"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


chroma_client = chromadb.PersistentClient(path="./chroma_db")

@router.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the PDF
        processor = PDFProcessor(
            pdf_path=file_path,
            output_folder=os.path.join(OUTPUT_DIR, file.filename.replace('.pdf', '')),
            chroma_client=chroma_client
        )
        processor.extract()
        
        return JSONResponse(
            content={
                "message": "PDF uploaded and processed successfully",
                "filename": file.filename,
                "output_folder": processor.output_folder
            },
            status_code=200
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/query-pdf/")
async def query_pdf(
    query: str = Query(..., description="The search query"),
    n_results: Optional[int] = Query(default=2, description="Number of results to return")
):
    try:
        searcher = PDFSearcher(chroma_client)
        results = searcher.search(query, n_results)
        
       
        formatted_results = []
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            formatted_results.append({
                "page_number": metadata['page_number'],
                "pdf_path": metadata['pdf_path'],
                "text_excerpt": doc,
                "images": metadata['image_paths'].split(',') if metadata['image_paths'] else []
            })
        
        return JSONResponse(
            content={
                "query": query,
                "results": formatted_results
            },
            status_code=200
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error while searching: {str(e)}"
        )

@router.post("/ai-query/")
async def ai_query(
    query: str = Query(..., description="The question to ask"),
    session_id: str = Header(None, description="Session ID for maintaining context"),
    settings: Settings = Depends(get_settings),
    ai_agent: PDFAIAgent = Depends(get_ai_agent)
):
    try:
        # Generate or use existing session ID
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Load existing context if available
        ai_agent.load_context(session_id)
        
        # If no context exists, search for relevant content
        if not ai_agent.context:
            searcher = PDFSearcher(chroma_client)
            search_results = searcher.search(query, n_results=3)
            
            # Set the context for the AI agent
            ai_agent.set_context(
                session_id=session_id,
                documents=search_results['documents'][0],
                metadata=search_results['metadatas'][0]
            )
        
        # Process the query with the AI agent
        response = await ai_agent.process_query(query)
        
        return JSONResponse(
            content={
                **response,
                "session_id": session_id
            },
            status_code=200
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing AI query: {str(e)}"
        )

@router.post("/clear-context/")
async def clear_context(
    session_id: str = Header(..., description="Session ID to clear"),
    ai_agent: PDFAIAgent = Depends(get_ai_agent)
):
    try:
        ai_agent.clear_context(session_id)
        return JSONResponse(
            content={"message": "Context cleared successfully"},
            status_code=200
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing context: {str(e)}"
        )
