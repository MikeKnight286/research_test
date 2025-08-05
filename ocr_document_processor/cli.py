"""Command Line Interface for OCR Document Processor."""

import os
import argparse
from .core.main_processor import OCRDocumentProcessor


def main():
    parser = argparse.ArgumentParser(description='Comprehensive OCR Document Processor')
    parser.add_argument('pdf_path', help='Path to PDF file')
    parser.add_argument('--output_dir', help='Output directory (default: auto-generated)')
    parser.add_argument('--pages', default="1", help='Comma-separated page numbers (1-indexed, default: 1)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found: {args.pdf_path}")
        return 1
    
    pdf_path = args.pdf_path
    textbook_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = args.output_dir or f"ocr_results/{textbook_name}"
    
    try:
        page_numbers = [int(p.strip()) - 1 for p in args.pages.split(',')]
    except ValueError:
        print(f"Error: Invalid page numbers format: {args.pages}")
        return 1
    
    processor = OCRDocumentProcessor()
    
    try:
        results = processor.process_document(
            pdf_path=pdf_path,
            output_dir=output_dir,
            page_numbers=page_numbers
        )
        
        total_pages = len(results)
        total_images = sum(len(page['image_segments']) for page in results.values())
        total_text = sum(len(page['text_content']) for page in results.values())
        total_paragraphs = sum(len(page.get('paragraphs', [])) for page in results.values())
        toc_pages = sum(1 for page in results.values() if page['is_toc_page'])
        toc_entries = sum(len(page.get('toc_entries', [])) for page in results.values())
        avg_success = sum(p['statistics']['association_success_rate'] for p in results.values()) / total_pages if total_pages > 0 else 0
        total_concepts = sum(len(p.get('hierarchical_structure', {}).get('concepts', [])) for p in results.values())
        
        print(f"\n=== Processing Summary ===")
        print(f"PDF: {pdf_path}")
        print(f"Output: {output_dir}")
        print(f"Pages processed: {total_pages}")
        print(f"Image segments: {total_images}")
        print(f"Text elements: {total_text}")
        print(f"Logical paragraphs: {total_paragraphs}")
        print(f"Concepts extracted: {total_concepts}")
        print(f"Association success: {avg_success:.2%}")
        print(f"TOC pages: {toc_pages}")
        print(f"TOC entries: {toc_entries}")
        
        if toc_entries:
            print("\n=== Sample TOC Entries ===")
            for page in results.values():
                if page.get('is_toc_page'):
                    for entry in page.get('toc_entries', []):
                        print(f"  {entry['number']} {entry['title']} ~-~ page: {entry['page']}")
        
        return 0
        
    except Exception as e:
        print(f"Error processing document: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()