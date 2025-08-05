"""
OCR Document Processor

A comprehensive document processing pipeline that combines OCR, layout analysis, and 
structured output generation. Designed for academic documents with emphasis on 
mathematical content, table of contents extraction, and knowledge structure preservation.

Features:
- OCR with text and math detection
- Layout analysis and element classification
- Table of contents extraction and parsing
- Structured JSON output for downstream NLP
- Modular design for easy extension and customization
"""

from .core.main_processor import OCRDocumentProcessor
from .models.data_structures import TOCEntry, BoundingBox, TextBlock, ParagraphBlock, NLPOutputs
from .models.enums import BlockType

__version__ = "2.0.0"
__all__ = [
    "OCRDocumentProcessor",
    "TOCEntry",
    "BoundingBox", 
    "TextBlock",
    "ParagraphBlock",
    "NLPOutputs",
    "BlockType"
]