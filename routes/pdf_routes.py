from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import os
import chromadb
from smart_pdf_reader.main import PDFProcessor, PDFSearcher
from typing import Optional

router = APIRouter()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "extracted_content"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize ChromaDB client
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
        
        # Format the results in a more readable way
        formatted_results = []
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            formatted_results.append({
                "page_number": metadata['page_number'],
                "pdf_path": metadata['pdf_path'],
                "text_excerpt": doc[:500] + "..." if len(doc) > 500 else doc,
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
