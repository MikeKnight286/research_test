"""PDF processing utilities."""

import os
import io
from typing import List, Optional
from PIL import Image
import fitz
import pymupdf as fitz


class PDFUtils:
    """PDF processing utilities"""
    
    @staticmethod
    def pdf_to_images(pdf_path: str, page_numbers: Optional[List[int]] = None, dpi: int = 300) -> List[Image.Image]:
        """Convert PDF pages to PIL Images."""
        doc = fitz.open(pdf_path)
        images = []
        page_numbers = page_numbers or list(range(len(doc)))
        
        for page_num in page_numbers:
            if page_num >= len(doc):
                print(f"Warning: Page {page_num} does not exist in PDF. Skipping.")
                continue
                
            page = doc.load_page(page_num)
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("ppm")
            
            images.append(Image.open(io.BytesIO(img_data)))
        
        doc.close()
        return images