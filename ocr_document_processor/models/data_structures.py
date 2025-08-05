"""Data structures for OCR document processing."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from .enums import BlockType


@dataclass
class TOCEntry:
    """Table of Contents entry with hierarchical information"""
    number: str
    title: str
    page: str
    level: int
    full_text: str


@dataclass
class BoundingBox:
    """Normalized bounding box coordinates"""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    
    @property
    def width(self) -> float:
        return self.x_max - self.x_min
    
    @property
    def height(self) -> float:
        return self.y_max - self.y_min
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)


@dataclass
class NLPOutputs:
    """Placeholder for NLP pipeline outputs"""
    entities: List[Dict[str, Any]] = field(default_factory=list)
    keyphrases: List[str] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    sentence_spans: List[Tuple[int, int]] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    concept_links: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TextBlock:
    """Individual text block with positioning and semantic info"""
    text: str
    original_text: str
    bbox: BoundingBox
    polygon: List[List[float]]
    block_id: str
    block_type: BlockType = BlockType.UNKNOWN
    confidence: Optional[float] = None
    font_size: Optional[float] = None
    font_style: Optional[str] = None
    reading_order: Optional[int] = None
    clean_text: Optional[str] = None
    tokens: List[str] = field(default_factory=list)
    equation_latex: Optional[str] = None
    toc_level: Optional[int] = None
    chapter_id: Optional[str] = None
    section_id: Optional[str] = None
    nlp_outputs: NLPOutputs = field(default_factory=NLPOutputs)


@dataclass
class ParagraphBlock:
    """Grouped text lines forming a logical paragraph"""
    paragraph_id: str
    text_lines: List[TextBlock]
    combined_text: str
    bbox: BoundingBox
    line_spacing: float
    alignment: str
    confidence: float
    block_type: BlockType = BlockType.PARAGRAPH
    clean_text: Optional[str] = None
    toc_level: Optional[int] = None
    chapter_id: Optional[str] = None
    section_id: Optional[str] = None
    nlp_outputs: NLPOutputs = field(default_factory=NLPOutputs)