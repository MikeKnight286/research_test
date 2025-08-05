"""JSON output generation utilities."""

import os
import json
import datetime
import hashlib
from typing import List, Dict, Any
from dataclasses import asdict


class JSONGenerator:
    """JSON output generation utilities"""
    
    @staticmethod
    def create_page_metadata(pdf_path: str, page_num: int, dpi: int, 
                            containment_threshold: float, iou_threshold: float) -> Dict:
        """Create page-level metadata block."""
        pdf_name = os.path.basename(pdf_path)
        book_id = hashlib.md5(pdf_path.encode()).hexdigest()[:16]
        
        metadata = {
            'book_id': book_id,
            'source_pdf': pdf_name,
            'source_path': pdf_path,
            'page_number': page_num,
            'processing_timestamp': datetime.datetime.now().isoformat(),
            'ocr_engine': 'surya-ocr',
            'layout_engine': 'surya-layout',
            'language': 'auto-detected',
            'processing_version': '2.0.0',
            'dpi': dpi,
            'processing_parameters': {
                'containment_threshold': containment_threshold,
                'iou_threshold': iou_threshold
            }
        }
        
        return metadata
    
    @staticmethod
    def detect_language(text_content: List[Dict]) -> str:
        """Simple language detection based on character patterns."""
        all_text = ' '.join([item.get('text', '') for item in text_content])
        
        if not all_text.strip():
            return 'unknown'
        
        latin_chars = sum(1 for c in all_text if c.isascii() and c.isalpha())
        total_chars = sum(1 for c in all_text if c.isalpha())
        
        if total_chars == 0:
            return 'unknown'
        
        latin_ratio = latin_chars / total_chars
        
        if latin_ratio > 0.9:
            return 'en'
        elif latin_ratio > 0.7:
            return 'mixed-latin'
        else:
            return 'non-latin'