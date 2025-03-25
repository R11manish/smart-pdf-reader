from fastapi import FastAPI
from routes.pdf_routes import router as pdf_router

app = FastAPI()

# Include the PDF routes
app.include_router(pdf_router, prefix="/api/pdf", tags=["PDF"])

@app.get("/")
async def root():
    return {"message": "PDF Upload API"} 