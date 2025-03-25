from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import chromadb
from smart_pdf_reader.main import PDFProcessor

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
