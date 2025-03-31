import os
import fitz  
import chromadb
from typing import List, Dict, Any

class PDFProcessor:
    def __init__(self, pdf_path: str, output_folder: str, chroma_client: chromadb.Client):
        self.pdf_path = pdf_path
        self.output_folder = output_folder
        self.image_folder = os.path.join(output_folder, "images")
        self.client = chroma_client
        self.collection = None

    def setup_directories(self) -> None:
        """Create necessary output directories if they don't exist."""
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            os.makedirs(self.image_folder)

    def process_image(self, doc: fitz.Document, page_num: int, img_info: tuple, img_index: int) -> str:
        """Process and save a single image from the PDF."""
        xref = img_info[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        
        image_filename = f"page{page_num+1}_img{img_index+1}.{image_ext}"
        image_path = os.path.join(self.image_folder, image_filename)
        
        with open(image_path, "wb") as img_file:
            img_file.write(image_bytes)
            
        return image_path

    def process_page(self, doc: fitz.Document, page: fitz.Page, page_num: int, text_file) -> None:
        """Process a single page of the PDF."""
        text = page.get_text()
        text_file.write(f"\n\n--- PAGE {page_num + 1} ---\n\n")
        text_file.write(text)
        
        image_paths = [
            self.process_image(doc, page_num, img_info, idx)
            for idx, img_info in enumerate(page.get_images(full=True))
        ]
        
        self.store_in_chroma(text, page_num, image_paths)

    def store_in_chroma(self, text: str, page_num: int, image_paths: List[str]) -> None:
        """Store page content in ChromaDB."""
        metadata = {
            "page_number": page_num + 1,
            "image_paths": ",".join(image_paths),
            "pdf_path": self.pdf_path
        }
        
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[f"page_{page_num + 1}"]
        )

    def extract(self) -> None:
        """Main method to extract content from PDF."""
        self.setup_directories()
        self.collection = self.client.create_collection(name="pdf_pages")
        
        doc = fitz.open(self.pdf_path)
        text_file_path = os.path.join(self.output_folder, "extracted_text.txt")
        
        with open(text_file_path, "w", encoding="utf-8") as text_file:
            for page_num, page in enumerate(doc):
                print(f"Processing page {page_num + 1}/{len(doc)}")
                self.process_page(doc, page, page_num, text_file)
        
        print(f"Text extracted to: {text_file_path}")
        print(f"Images extracted to: {self.image_folder}")
        print("Data stored in ChromaDB collection 'pdf_pages'")
        
        doc.close()

class PDFSearcher:
    def __init__(self, chroma_client: chromadb.Client):
        self.client = chroma_client
        
    def search(self, query: str, n_results: int = 2) -> List[Dict[str, Any]]:
        """Search through the PDF content using ChromaDB."""
        collection = self.client.get_collection(name="pdf_pages")
        return collection.query(
            query_texts=[query],
            n_results=n_results
        )


