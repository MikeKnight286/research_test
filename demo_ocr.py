#!/usr/bin/env python3
"""
Demo script showing how to use the modular OCR Document Processor
"""

import os
from dotenv import load_dotenv
from ocr_document_processor import OCRDocumentProcessor

# Load environment variables from .env file
load_dotenv()

def main():
    # Get paths from environment variables
    pdf_path = os.getenv('PDF_PATH')
    output_dir = os.getenv('OUTPUT_DIR')
    
    if not pdf_path or not output_dir:
        print("❌ Error: Please set PDF_PATH and OUTPUT_DIR in your .env file")
        return
    
    # Initialize the processor 
    processor = OCRDocumentProcessor(
        containment_threshold=0.5,
        iou_threshold=0.1,
        dpi=300
    )
    
    try:
        # Process the document 
        results = processor.process_document(
            pdf_path=pdf_path,
            output_dir=output_dir,
            page_numbers=[1, 15, 16]  # 0 indexed
        )
        
        print("✅ Modular OCR processing completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📄 Pages processed: {len(results)}")
        
        for page_key, page_data in results.items():
            print(f"\n📖 {page_key}:")
            print(f"  - Text elements: {len(page_data['text_content'])}")
            print(f"  - Paragraphs: {len(page_data.get('paragraphs', []))}")
            print(f"  - TOC entries: {len(page_data.get('toc_entries', []))}")
            print(f"  - Image segments: {len(page_data['image_segments'])}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()