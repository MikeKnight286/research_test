#!/usr/bin/env python3
"""
Test script for enhanced OCR document processor with paragraph grouping
"""

import json
from ocr_document_processor import OCRDocumentProcessor, BoundingBox

def test_paragraph_grouping():
    """Test the paragraph grouping functionality with sample data"""
    
    # Create processor instance
    print("Creating OCR Document Processor...")
    processor = OCRDocumentProcessor()
    
    # Sample text content that mimics OCR output
    sample_text_content = [
        {
            'text': 'Introduction to Mathematical Proofs',
            'original_text': 'Introduction to Mathematical Proofs',
            'type': 'text',
            'bbox': [100, 100, 400, 130],
            'polygon': [[100, 100], [400, 100], [400, 130], [100, 130]],
            'confidence': 0.95,
            'layout_type': 'Heading',
            'semantic_role': 'structural',
            'keywords': ['introduction', 'mathematical', 'proof']
        },
        {
            'text': 'A mathematical proof is a deductive argument for a mathematical statement.',
            'original_text': 'A mathematical proof is a deductive argument for a mathematical statement.',
            'type': 'text',
            'bbox': [100, 150, 450, 180],
            'polygon': [[100, 150], [450, 150], [450, 180], [100, 180]],
            'confidence': 0.92,
            'layout_type': 'Text',
            'semantic_role': 'content',
            'keywords': ['mathematical', 'proof', 'deductive']
        },
        {
            'text': 'It shows that the statement is necessarily true, assuming the axioms and',
            'original_text': 'It shows that the statement is necessarily true, assuming the axioms and',
            'type': 'text',
            'bbox': [100, 185, 430, 215],
            'polygon': [[100, 185], [430, 185], [430, 215], [100, 215]],
            'confidence': 0.89,
            'layout_type': 'Text',
            'semantic_role': 'content',
            'keywords': ['statement', 'true', 'axioms']
        },
        {
            'text': 'previously established statements are valid.',
            'original_text': 'previously established statements are valid.',
            'type': 'text',
            'bbox': [100, 220, 350, 250],
            'polygon': [[100, 220], [350, 220], [350, 250], [100, 250]],
            'confidence': 0.91,
            'layout_type': 'Text',
            'semantic_role': 'content',
            'keywords': ['statements', 'valid']
        },
        {
            'text': 'Definition 1.1',
            'original_text': 'Definition 1.1',
            'type': 'text',
            'bbox': [100, 280, 220, 310],
            'polygon': [[100, 280], [220, 280], [220, 310], [100, 310]],
            'confidence': 0.94,
            'layout_type': 'Heading',
            'semantic_role': 'structural',
            'keywords': ['definition']
        },
        {
            'text': 'A theorem is a statement that can be proven to be true.',
            'original_text': 'A theorem is a statement that can be proven to be true.',
            'type': 'text',
            'bbox': [100, 320, 420, 350],
            'polygon': [[100, 320], [420, 320], [420, 350], [100, 350]],
            'confidence': 0.93,
            'layout_type': 'Text',
            'semantic_role': 'content',
            'keywords': ['theorem', 'statement', 'proven', 'true']
        }
    ]
    
    print(f"Testing paragraph grouping with {len(sample_text_content)} text blocks...")
    
    # Test paragraph grouping
    try:
        paragraphs = processor.group_text_into_paragraphs(sample_text_content)
        print(f"✓ Successfully grouped text into {len(paragraphs)} paragraphs")
        
        # Display paragraph information
        for i, paragraph in enumerate(paragraphs):
            print(f"\nParagraph {i+1}:")
            print(f"  Type: {paragraph.block_type.value}")
            print(f"  Text lines: {len(paragraph.text_lines)}")
            print(f"  Combined text: {paragraph.combined_text[:100]}...")
            print(f"  Confidence: {paragraph.confidence:.2f}")
            print(f"  Alignment: {paragraph.alignment}")
            print(f"  Concepts: {len(paragraph.detected_concepts)}")
            
            if paragraph.detected_concepts:
                for concept in paragraph.detected_concepts:
                    print(f"    - {concept.name} ({concept.category})")
        
        # Test hierarchical structure creation
        print(f"\nTesting hierarchical structure creation...")
        hierarchical_structure = processor.create_hierarchical_structure(paragraphs)
        
        print(f"✓ Created hierarchical structure:")
        print(f"  Sections: {len(hierarchical_structure['sections'])}")
        print(f"  Total concepts: {len(hierarchical_structure['concepts'])}")
        print(f"  NLP readiness score: {hierarchical_structure['metadata']['readiness_score']:.1f}/100")
        print(f"  Document quality: {hierarchical_structure['metadata'].get('paragraph_types', {})}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Enhanced OCR Document Processor Test ===")
    success = test_paragraph_grouping()
    if success:
        print("\n✓ All tests passed! The enhanced processor is ready for NLP tasks.")
    else:
        print("\n✗ Tests failed. Check the implementation.")