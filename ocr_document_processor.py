#!/usr/bin/env python3
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

import os
import json
import re
import html
import unicodedata
import datetime
import hashlib
from typing import List, Optional, Dict, Any, Tuple, Union
from PIL import Image
import fitz
import pymupdf as fitz
from dataclasses import dataclass, asdict, field
from pathlib import Path
from enum import Enum


# ============================================================================
# Data Classes and Enums
# ============================================================================

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


class BlockType(Enum):
    """Enumeration of semantic block types"""
    HEADING = "heading"
    PARAGRAPH = "paragraph" 
    EQUATION = "equation"
    CAPTION = "caption"
    TOC_ENTRY = "toc_entry"
    LIST_ITEM = "list_item"
    FOOTER = "footer"
    HEADER = "header"
    FIGURE_LABEL = "figure_label"
    TABLE_CELL = "table_cell"
    UNKNOWN = "unknown"


@dataclass
class DetectedConcept:
    """Detected concept with metadata"""
    name: str
    uri: Optional[str] = None
    confidence: Optional[float] = None
    context: Optional[str] = None
    category: Optional[str] = None


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
    detected_concepts: List[DetectedConcept] = field(default_factory=list)
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
    detected_concepts: List[DetectedConcept] = field(default_factory=list)
    toc_level: Optional[int] = None
    chapter_id: Optional[str] = None
    section_id: Optional[str] = None
    nlp_outputs: NLPOutputs = field(default_factory=NLPOutputs)


# ============================================================================
# TOC Extraction Module
# ============================================================================

class TOCExtractor:
    """Extract and parse table of contents entries from text"""
    
    def __init__(self, min_title_length=8, patterns=None, keywords=None):
        self.patterns = patterns or [
            r'([A-Z]?\d*\.?\d*)\s+([^.]{10,}?)\s*\.{3,}\s*-?(\d+)',
            r'([A-Z]?\d*\.?\d*)\s+([^0-9]{10,}?)\s+(\d+)(?:\s|$)',
            r'^([^.]{10,}?)\s*\.{3,}\s*-?(\d+)',
            r'([A-Z]?\d*\.?\d*)\s*([^.]{8,}?)\s*\.{2,}\s*-?(\d+)'
        ]
        
        self.meaningful_keywords = keywords or {
            'introduction', 'induction', 'proof', 'theorem', 'definition', 'property',
            'formula', 'equation', 'method', 'algorithm', 'structure', 'analysis',
            'proposition', 'statement', 'logic', 'logical', 'mathematical', 'basic', 'advanced',
            'theory', 'concept', 'principle', 'rule', 'law', 'function', 'set',
            'number', 'system', 'operation', 'relation', 'graph', 'geometry',
            'algebra', 'calculus', 'statistics', 'probability', 'matrix', 'vector',
            'chapter', 'section', 'appendix', 'conclusion', 'summary', 'exercise',
            'problem', 'solution', 'example', 'application', 'properties', 'types'
        }
        self.min_title_length = min_title_length
        
    def extract_toc_entries(self, text: str) -> List[TOCEntry]:
        """Extract meaningful TOC entries from text."""
        entries = []
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        for line in lines:
            line_entries = self._extract_from_line(line)
            meaningful_entries = [entry for entry in line_entries if self._is_meaningful_entry(entry)]
            entries.extend(meaningful_entries)
        
        return self._post_process_entries(entries)
    
    def _extract_from_line(self, line: str) -> List[TOCEntry]:
        """Extract TOC entries from a single line."""
        entries = []
        page_matches = list(re.finditer(r'-?\d{1,3}(?:\s|$)', line))
        
        if len(page_matches) > 1:
            entries.extend(self._split_multi_entry_line(line, page_matches))
        else:
            for pattern in self.patterns:
                match = re.search(pattern, line)
                if match:
                    groups = match.groups()
                    number = groups[0].strip() if len(groups) == 3 else ""
                    title = groups[1].strip() if len(groups) == 3 else groups[0].strip()
                    page = groups[2].strip() if len(groups) == 3 else groups[1].strip()
                    
                    title = self._clean_title(title)

                    entries.append(TOCEntry(
                        number=number,
                        title=title,
                        page=page,
                        level=0,
                        full_text=line
                    ))
                    break
        return entries
    
    def _split_multi_entry_line(self, line: str, page_matches: List) -> List[TOCEntry]:
        segments = []
        last_end = 0
        
        for i, match in enumerate(page_matches):
            if i == 0:
                segments.append(line[last_end:match.end()])
            else:
                segments.append(line[last_end:match.end()])
            last_end = match.start()
        
        entries = []
        for segment in segments:
            segment = segment.strip()
            for pattern in self.patterns:
                match = re.search(pattern, segment)
                if match:
                    groups = match.groups()
                    number = groups[0].strip() if len(groups) == 3 else ""
                    title = groups[1].strip() if len(groups) == 3 else groups[0].strip()
                    page = groups[2].strip() if len(groups) == 3 else groups[1].strip()
                    
                    title = self._clean_title(title)

                    entries.append(TOCEntry(
                        number=number,
                        title=title,
                        page=page,
                        level=0,
                        full_text=segment
                    ))
                    break
        return entries
    
    def _clean_title(self, title: str) -> str:
        """Clean TOC title by removing dots and other artifacts."""
        title = re.sub(r'<[^>]+>', '', title).strip()
        title = re.sub(r'\s+', ' ', title).strip()
        title = re.sub(r'(?:\.\s*){2,}', '', title).strip()
        title = re.sub(r'(\\[a-zA-Z]+)', '', title)
        title = re.sub(r'[^A-Za-z0-9"\' ]+$', '', title)
        title = re.sub(r'^[A-Z](\s|(?=\b))', '', title).strip() if re.match(r'^[A-Z]\s', title) else title
        return title.strip()
    
    def _is_meaningful_entry(self, entry: TOCEntry) -> bool:
        """Check if a TOC entry is meaningful."""
        title = entry.title.lower().strip()
        return (len(title) >= self.min_title_length and 
                not re.match(r'^[a-z]\d*\.?\d*$', title) and
                len(re.sub(r'[.\s-]', '', title)) >= 3 and
                any(keyword in title for keyword in self.meaningful_keywords))
    
    def _post_process_entries(self, entries: List[TOCEntry]) -> List[TOCEntry]:
        """Apply post-processing to entries."""
        entries = self._merge_fragmented_entries(entries)
        entries = self._assign_hierarchy_levels(entries)
        return entries
    
    def _merge_fragmented_entries(self, entries: List[TOCEntry]) -> List[TOCEntry]:
        merged = []
        i = 0
        
        while i < len(entries):
            current = entries[i]
            if i + 1 < len(entries) and self._could_be_continuation(current, entries[i + 1]):
                next_entry = entries[i + 1]
                merged_entry = TOCEntry(
                    number=current.number or next_entry.number,
                    title=f"{current.title} {next_entry.title}".strip(),
                    page=next_entry.page or current.page,
                    level=0,
                    full_text=f"{current.full_text} {next_entry.full_text}"
                )
                merged.append(merged_entry)
                i += 2
                continue
            merged.append(current)
            i += 1
        return merged
    
    def _could_be_continuation(self, entry1: TOCEntry, entry2: TOCEntry) -> bool:
        """Check if entry2 could be a continuation of entry1."""
        title1_ends_abruptly = len(entry1.title.split()) < 3
        title2_starts_without_number = not entry2.number.strip()
        
        try:
            page1 = int(entry1.page) if entry1.page.isdigit() else 0
            page2 = int(entry2.page) if entry2.page.isdigit() else 0
            similar_pages = abs(page1 - page2) <= 2
        except:
            similar_pages = True
        
        return title1_ends_abruptly and title2_starts_without_number and similar_pages
    
    def _assign_hierarchy_levels(self, entries: List[TOCEntry]) -> List[TOCEntry]:
        """Assign hierarchy levels based on numbering schemes."""
        for entry in entries:
            number = entry.number.strip()
            if not number:
                entry.level = 1
            elif re.match(r'^[A-Z]$', number):
                entry.level = 1
            else:
                dot_count = number.count('.')
                entry.level = min(dot_count + 1, 4)
        return entries


# ============================================================================
# Text Processing and Structure Detection
# ============================================================================

class TextProcessor:
    """Text processing utilities for OCR output"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for NLP processing"""
        text = TextProcessor._strip_html_tags(text)
        text = html.unescape(text)
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'\.{3,}', ' ', text)
        text = re.sub(r'\s*-\s*$', '', text)
        text = re.sub(r'^\s*-\s*', '', text)
        return text
    
    @staticmethod
    def _strip_html_tags(text: str) -> str:
        """Strip HTML tags while preserving math content"""
        math_matches = re.findall(r'<math[^>]*>(.*?)</math>', text, re.DOTALL)
        math_placeholders = {}
        
        for i, math_content in enumerate(math_matches):
            placeholder = f"__MATH_PLACEHOLDER_{i}__"
            math_placeholders[placeholder] = math_content
            text = text.replace(f'<math>{math_content}</math>', placeholder, 1)
        
        text = re.sub(r'<[^>]+>', '', text)
        
        for placeholder, math_content in math_placeholders.items():
            text = text.replace(placeholder, math_content)
        
        return text
    
    @staticmethod
    def extract_font_styles(text: str) -> Dict:
        """Extract font style information and clean text."""
        cleaned_text = text
        font_styles = {
            'has_bold': False,
            'has_italic': False,
            'has_underline': False,
            'has_strikethrough': False,
            'has_superscript': False,
            'has_subscript': False,
            'html_tags_removed': [],
            'markdown_formatting': []
        }
        
        html_patterns = {
            'bold': [r'<b>(.*?)</b>', r'<strong>(.*?)</strong>'],
            'italic': [r'<i>(.*?)</i>', r'<em>(.*?)</em>'],
            'underline': [r'<u>(.*?)</u>'],
            'strikethrough': [r'<s>(.*?)</s>', r'<strike>(.*?)</strike>'],
            'superscript': [r'<sup>(.*?)</sup>'],
            'subscript': [r'<sub>(.*?)</sub>']
        }
        
        markdown_patterns = {
            'bold': [r'\*\*(.*?)\*\*', r'__(.*?)__'],
            'italic': [r'\*(.*?)\*', r'_(.*?)_'],
            'strikethrough': [r'~~(.*?)~~']
        }
        
        for style, patterns in html_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    font_styles[f'has_{style}'] = True
                    font_styles['html_tags_removed'].extend(matches)
                    cleaned_text = re.sub(pattern, r'\1', cleaned_text, flags=re.IGNORECASE)
        
        for style, patterns in markdown_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, cleaned_text)
                if matches:
                    font_styles[f'has_{style}'] = True
                    font_styles['markdown_formatting'].extend(matches)
                    cleaned_text = re.sub(pattern, r'\1', cleaned_text)
        
        if re.search(r'[²³⁴⁵⁶⁷⁸⁹⁰]', text):
            font_styles['has_superscript'] = True
        
        if re.search(r'[₀₁₂₃₄₅₆₇₈₉]', text):
            font_styles['has_subscript'] = True
        
        font_styles['original_text'] = text
        font_styles['cleaned_text'] = cleaned_text.strip()
        
        return font_styles


# ============================================================================
# Main OCR Document Processor
# ============================================================================

class OCRDocumentProcessor:
    """
    Comprehensive OCR document processing pipeline that combines:
    - OCR text extraction with layout analysis
    - Table of contents detection and parsing
    - Structured JSON output generation
    - Mathematical content preservation
    """
    
    def __init__(self, containment_threshold=0.5, iou_threshold=0.1, dpi=300, toc_extractor=None):
        self.containment_threshold = containment_threshold
        self.iou_threshold = iou_threshold
        self.dpi = dpi
        self.toc_extractor = toc_extractor or TOCExtractor()
        self.text_processor = TextProcessor()
        
        # Initialize predictors
        print("Initializing Surya predictors...")
        try:
            from surya.recognition import RecognitionPredictor
            from surya.detection import DetectionPredictor
            from surya.layout import LayoutPredictor
            
            self.recognition_predictor = RecognitionPredictor()
            self.detection_predictor = DetectionPredictor()
            self.layout_predictor = LayoutPredictor()
            print("Predictors initialized successfully!")
        except ImportError as e:
            print(f"Error importing Surya: {e}")
            print("Please install Surya: pip install surya-ocr")
            raise
    
    # ========================================================================
    # PDF Processing and Image Extraction
    # ========================================================================
    
    def pdf_to_images(self, pdf_path: str, page_numbers: Optional[List[int]] = None, dpi: int = None) -> List[Image.Image]:
        """Convert PDF pages to PIL Images."""
        dpi = dpi or self.dpi
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
            
            import io
            images.append(Image.open(io.BytesIO(img_data)))
        
        doc.close()
        return images
    
    def save_image_segments(self, image: Image.Image, layout_predictions, 
                           page_num: int, output_dir: str) -> List[Dict]:
        """Extract and save image segments."""
        image_segments = []
        image_labels = {'Picture', 'Figure', 'Formula', 'Table'}
        bboxes = getattr(layout_predictions, 'bboxes', [])
        
        for i, bbox_info in enumerate(bboxes):
            label = getattr(bbox_info, 'label', '')
            if label not in image_labels:
                continue
                
            bbox = getattr(bbox_info, 'bbox', [])
            if len(bbox) < 4:
                continue
                
            x1, y1, x2, y2 = bbox[:4]
            cropped_image = image.crop((x1, y1, x2, y2))
            filename = f"page_{page_num+1}_image_{i}_{label.lower()}.jpg"
            filepath = os.path.join(output_dir, filename)
            cropped_image.save(filepath, 'JPEG', quality=95)
            
            image_segments.append({
                'filename': filename,
                'filepath': filepath,
                'label': label,
                'bbox': bbox,
                'polygon': getattr(bbox_info, 'polygon', []),
                'position': getattr(bbox_info, 'position', 0),
                'confidence': getattr(bbox_info, 'confidence', 0.0)
            })
        
        return image_segments
    
    # ========================================================================
    # TOC Processing
    # ========================================================================
    
    def is_toc_content(self, text_content: List[Dict], layout_elements: List[Dict]) -> bool:
        """Determine if page contains table of contents."""
        toc_labels = {'tableofcontents', 'table-of-contents', 'toc'}
        normalize = lambda x: x.lower().replace('_', '-').replace(' ', '-')
        return (any(normalize(elem.get('label', '')) in toc_labels for elem in layout_elements) or
                any(normalize(item.get('layout_type', '')) in toc_labels for item in text_content))
    
    def extract_toc_elements(self, text_content: List[Dict]) -> List[Dict]:
        """Extract text elements associated with TOC layout."""
        toc_labels = {'tableofcontents', 'table-of-contents', 'toc'}
        normalize = lambda x: x.lower().replace('_', '-').replace(' ', '-')
        return [item for item in text_content if normalize(item.get('layout_type', '')) in toc_labels]
    
    def clean_and_extract_toc_entries(self, toc_text_elements: List[Dict]) -> List[TOCEntry]:
        """Clean TOC text elements and extract meaningful entries."""
        if not toc_text_elements:
            return []
        return self._deduplicate_entries(
            self._extract_from_individual_elements(toc_text_elements) +
            self._extract_from_combined_text(toc_text_elements)
        )
    
    def _extract_from_individual_elements(self, toc_text_elements: List[Dict]) -> List[TOCEntry]:
        """Extract TOC entries from individual text elements."""
        sorted_elements = sorted(toc_text_elements, 
                            key=lambda x: (x.get('bbox', [0, 0, 0, 0])[1], x.get('bbox', [0, 0, 0, 0])[0]))
        
        entries = []
        current = {}
        
        for element in sorted_elements:
            text = re.sub(r'<[^>]+>', '', element.get('text', '')).strip()
            text = re.sub(r'\.{2,}', '', text)
            text = re.sub(r'(?:\.\s*){2,}', '', text)
            text = re.sub(r'[\.\s]+$', '', text)
            text = re.sub(r'^[\.\s]+', '', text)
            text = re.sub(r'[^ ]*\\\\\s*', '', text)
            text = text.strip()

            if len(text) < 2 or text.lower() in {'table of contents', 'contents', 'toc', 'index'}:
                continue
                
            if self._is_valid_page_number(text):
                if current and 'title' in current:
                    entry = self._create_toc_entry(
                        current.get('number', ''),
                        current.get('title', ''),
                        text,
                        f"{current.get('full_text', '')} {text}"
                    )
                    if entry:
                        entries.append(entry)
                    current = {}
                continue
            
            match = re.match(r'^([A-Z]?\d+\.?\d*\.?\d*)\s+(.+)', text)
            if match and len(match.group(2)) > 2:
                current = {'number': match.group(1), 'title': match.group(2).strip()}
            elif self._is_meaningful_title(text):
                if current and 'title' in current:
                    entry = self._create_toc_entry(
                        current.get('number', ''),
                        current.get('title', ''),
                        '',
                        current.get('full_text', '')
                    )
                    if entry:
                        entries.append(entry)
                current = {'number': '', 'title': text}
        
        if current and 'title' in current:
            entry = self._create_toc_entry(
                current.get('number', ''),
                current.get('title', ''),
                '',
                current.get('full_text', '')
            )
            if entry:
                entries.append(entry)
        
        return entries
    
    def _extract_from_combined_text(self, toc_text_elements: List[Dict]) -> List[TOCEntry]:
        """Extract TOC entries from combined text lines."""
        lines = self._group_elements_by_line(toc_text_elements)
        entries = []
        
        for line_elements in lines:
            line_text = " ".join(re.sub(r'<[^>]+>', '', elem.get('text', '')).strip() 
                         for elem in line_elements).strip()
            
            if len(line_text) < 3 or line_text.lower() in {'table of contents', 'contents', 'toc'}:
                continue
                
            entries.extend(self.toc_extractor._extract_from_line(line_text))
        
        return entries
    
    def _create_toc_entry(self, number: str, title: str, page: str, full_text: str) -> Optional[TOCEntry]:
        title = self.toc_extractor._clean_title(title)
        if not self._is_meaningful_title(title):
            return None
        return TOCEntry(
            number=number,
            title=title,
            page=page,
            level=self._determine_entry_level(number),
            full_text=full_text
        )

    def _is_meaningful_title(self, title: str) -> bool:
        return (len(title) >= 5 and re.search(r'[a-zA-Z]', title) and
                not re.match(r'^[A-Z](\s|\d?$)', title) and
                any(len(word) >= 3 and re.match(r'^[A-Za-z]+', word) for word in title.split()))
  
    def _determine_entry_level(self, number: str) -> int:
        """Determine hierarchy level from section number."""
        return min(number.count('.') + 1, 4) if number else 1
    
    def _deduplicate_entries(self, entries: List[TOCEntry]) -> List[TOCEntry]:
        """Remove duplicate entries based on title."""
        seen = set()
        unique = []
        for entry in entries:
            norm_title = re.sub(r'\s+', ' ', entry.title.lower().strip())
            if norm_title not in seen:
                unique.append(entry)
                seen.add(norm_title)
        return unique
    
    def _is_valid_page_number(self, page_str: str) -> bool:
        """Validate page number."""
        page_str = (page_str or '').strip()
        if not page_str:
            return False
        
        if re.match(r'^\d{1,4}$', page_str):
            try:
                return 1 <= int(page_str) <= 9999
            except ValueError:
                return False
        
        roman_numerals = {'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x',
                         'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx',
                         'xxi', 'xxii', 'xxiii', 'xxiv', 'xxv', 'xxx', 'xl', 'l', 'lx', 'lxx', 'lxxx', 'xc', 'c'}
        return page_str.lower() in roman_numerals
    
    def _group_elements_by_line(self, elements: List[Dict]) -> List[List[Dict]]:
        """Group elements by similar Y coordinates."""
        if not elements:
            return []
        
        sorted_elements = sorted(elements, key=lambda x: x.get('bbox', [0, 0, 0, 0])[1])
        lines = []
        current_line = [sorted_elements[0]]
        
        for elem in sorted_elements[1:]:
            current_y = elem.get('bbox', [0, 0, 0, 0])[1]
            last_y = current_line[-1].get('bbox', [0, 0, 0, 0])[1]
            
            if abs(current_y - last_y) <= 20:
                current_line.append(elem)
            else:
                lines.append(current_line)
                current_line = [elem]
        
        if current_line:
            lines.append(current_line)
        
        for line in lines:
            line.sort(key=lambda x: x.get('bbox', [0, 0, 0, 0])[0])
        
        return lines
    
    # ========================================================================
    # Layout and Text Processing
    # ========================================================================
    
    def bbox_overlap(self, bbox1: List[float], bbox2: List[float], metric: str = 'containment') -> float:
        """Calculate bbox overlap using specified metric."""
        if len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0
        
        x1_1, y1_1, x2_1, y2_1 = bbox1[:4]
        x1_2, y1_2, x2_2, y2_2 = bbox2[:4]
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        if metric == 'iou':
            union = area1 + area2 - intersection
            return intersection / union if union > 0 else 0.0
        else:
            return intersection / area1 if area1 > 0 else 0.0
    
    def associate_text_with_layout(self, text_content: List[Dict], 
                                 layout_elements: List[Dict],
                                 containment_threshold: float = None,
                                 iou_threshold: float = None) -> List[Dict]:
        """Associate text with layout elements and add semantic labeling."""
        containment_threshold = containment_threshold or self.containment_threshold
        iou_threshold = iou_threshold or self.iou_threshold
        enhanced = []
        hierarchy_map = {
            'Title': 0, 'Heading': 1, 'Subheading': 2, 'List-item': 3,
            'Text': 4, 'Caption': 5, 'Footnote': 6, 'Page-header': 7,
            'Page-footer': 7, 'unassigned': 8, 'unknown': 9
        }
        
        for text_item in text_content:
            text_bbox = text_item.get('bbox', [])
            best_match = None
            best_score = 0.0
            best_metric = None
            
            for layout_item in layout_elements:
                layout_bbox = layout_item.get('bbox', [])
                containment = self.bbox_overlap(text_bbox, layout_bbox, 'containment')
                iou = self.bbox_overlap(text_bbox, layout_bbox, 'iou')
                
                if containment >= containment_threshold and containment > best_score:
                    best_match = layout_item
                    best_score = containment
                    best_metric = 'containment'
                elif containment < containment_threshold and iou >= iou_threshold and iou > best_score:
                    best_match = layout_item
                    best_score = iou
                    best_metric = 'iou'
            
            enhanced_item = text_item.copy()
            if best_match:
                layout_type = best_match.get('label', 'unknown')
                enhanced_item.update({
                    'layout_type': layout_type,
                    'layout_confidence': best_match.get('confidence', 0.0),
                    'layout_position': best_match.get('position', 0),
                    'association_score': best_score,
                    'association_method': best_metric,
                    'hierarchy_level': hierarchy_map.get(layout_type, 5)
                })
            else:
                enhanced_item.update({
                    'layout_type': 'unassigned',
                    'layout_confidence': 0.0,
                    'layout_position': -1,
                    'association_score': 0.0,
                    'association_method': 'none',
                    'hierarchy_level': 8
                })
            
            semantic_info = self.add_semantic_labeling(enhanced_item)
            enhanced_item.update(semantic_info)
            
            enhanced.append(enhanced_item)
        
        return enhanced
    
    def add_semantic_labeling(self, text_item: Dict) -> Dict:
        """Add semantic type and role based on keywords and layout."""
        text = text_item.get('text', '').strip().lower()
        layout_type = text_item.get('layout_type', '')
        
        semantic_type = 'text'
        semantic_role = 'content'
        
        if any(keyword in text for keyword in ['figure', 'fig.', 'image', 'diagram']):
            semantic_type = 'figure_reference'
            semantic_role = 'caption'
        elif any(keyword in text for keyword in ['table', 'tab.', 'chart']):
            semantic_type = 'table_reference'
            semantic_role = 'caption'
        elif any(keyword in text for keyword in ['equation', 'formula', 'eq.']):
            semantic_type = 'equation_reference'
            semantic_role = 'label'
        elif text_item.get('type') == 'mathematical':
            semantic_type = 'equation'
            semantic_role = 'mathematical_content'
        elif layout_type in ['Title', 'Heading', 'Subheading']:
            semantic_type = 'heading'
            semantic_role = 'structural'
        elif layout_type == 'Caption':
            semantic_type = 'caption'
            semantic_role = 'descriptive'
        elif layout_type == 'List-item':
            semantic_type = 'list'
            semantic_role = 'enumeration'
        elif layout_type in ['Page-header', 'Page-footer']:
            semantic_type = 'metadata'
            semantic_role = 'navigation'
        elif layout_type == 'Footnote':
            semantic_type = 'reference'
            semantic_role = 'annotation'
        
        keywords_found = self._extract_keywords(text)
        
        return {
            'semantic_type': semantic_type,
            'semantic_role': semantic_role,
            'keywords': keywords_found,
            'is_structural': semantic_role in ['structural', 'navigation'],
            'is_content': semantic_role in ['content', 'mathematical_content'],
            'is_metadata': semantic_role in ['annotation', 'navigation', 'descriptive']
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text."""
        academic_keywords = {
            'definition', 'theorem', 'proof', 'proposition', 'lemma', 'corollary',
            'example', 'problem', 'solution', 'exercise', 'note', 'remark',
            'chapter', 'section', 'subsection', 'introduction', 'conclusion',
            'figure', 'table', 'equation', 'formula', 'algorithm', 'method'
        }
        
        found_keywords = []
        words = text.lower().split()
        
        for word in words:
            cleaned_word = re.sub(r'[^\w]', '', word)
            if cleaned_word in academic_keywords:
                found_keywords.append(cleaned_word)
        
        return found_keywords
    
    def process_text_content(self, ocr_predictions) -> List[Dict]:
        """Process OCR predictions to extract content."""
        text_content = []
        math_indicators = {
            '∫', '∑', '∏', '√', '∞', '±', '≤', '≥', '≠', '≈', '≡',
            'α', 'β', 'γ', 'δ', 'ε', 'θ', 'λ', 'μ', 'π', 'σ', 'φ', 'ψ', 'ω',
            '²', '³', '₁', '₂', '₃', '₄', '₅',
            '\\frac', '\\int', '\\sum', '\\sqrt', '\\alpha', '\\beta',
            '$', '\\(', '\\)', '\\[', '\\]'
        }
        
        for line in getattr(ocr_predictions, 'text_lines', []):
            text = getattr(line, 'text', '')
            original_words = self._convert_to_dict_list(getattr(line, 'words', []))
            chars = self._convert_to_dict_list(getattr(line, 'chars', []))
            
            grouped_words = self.group_chars_into_words(chars) if not original_words else original_words
            font_style_info = self.text_processor.extract_font_styles(text)
            
            text_content.append({
                'text': font_style_info['cleaned_text'],
                'original_text': text,
                'type': 'mathematical' if any(ind in text for ind in math_indicators) else 'text',
                'bbox': getattr(line, 'bbox', []),
                'polygon': getattr(line, 'polygon', []),
                'confidence': getattr(line, 'confidence', 0.0),
                'words': grouped_words,
                'chars': chars,
                'font_styles': font_style_info
            })
        
        return text_content
    
    def group_chars_into_words(self, chars: List[Dict]) -> List[Dict]:
        """Group character arrays into words for NLP processing."""
        if not chars:
            return []
        
        words = []
        current_word = {'chars': [], 'text': '', 'bbox': None}
        word_gap_threshold = 20
        
        sorted_chars = sorted(chars, key=lambda c: (c.get('bbox', [0, 0, 0, 0])[1], c.get('bbox', [0, 0, 0, 0])[0]))
        
        for char in sorted_chars:
            char_bbox = char.get('bbox', [])
            char_text = char.get('text', '').strip()
            
            if not char_text or len(char_bbox) < 4:
                continue
                
            if current_word['chars']:
                last_char_bbox = current_word['chars'][-1].get('bbox', [])
                if len(last_char_bbox) >= 4:
                    horizontal_gap = char_bbox[0] - last_char_bbox[2]
                    vertical_gap = abs(char_bbox[1] - last_char_bbox[1])
                    
                    if horizontal_gap > word_gap_threshold or vertical_gap > 10:
                        if current_word['text'].strip():
                            words.append(self._finalize_word(current_word))
                        current_word = {'chars': [], 'text': '', 'bbox': None}
            
            current_word['chars'].append(char)
            current_word['text'] += char_text
            current_word['bbox'] = self._merge_bboxes(current_word['bbox'], char_bbox)
        
        if current_word['text'].strip():
            words.append(self._finalize_word(current_word))
        
        return words
    
    def _finalize_word(self, word: Dict) -> Dict:
        """Finalize a word with proper attributes."""
        return {
            'text': word['text'].strip(),
            'bbox': word['bbox'],
            'chars': word['chars'],
            'confidence': sum(c.get('confidence', 0) for c in word['chars']) / len(word['chars']) if word['chars'] else 0,
            'char_count': len(word['chars']),
            'polygon': self._chars_to_polygon(word['chars'])
        }
    
    def _merge_bboxes(self, bbox1: List[float], bbox2: List[float]) -> List[float]:
        """Merge two bounding boxes."""
        if not bbox1:
            return bbox2[:]
        if not bbox2:
            return bbox1[:]
        
        return [
            min(bbox1[0], bbox2[0]),
            min(bbox1[1], bbox2[1]),
            max(bbox1[2], bbox2[2]),
            max(bbox1[3], bbox2[3])
        ]
    
    def _chars_to_polygon(self, chars: List[Dict]) -> List[List[float]]:
        """Convert chars to polygon representation."""
        if not chars:
            return []
        
        all_points = []
        for char in chars:
            polygon = char.get('polygon', [])
            if polygon:
                all_points.extend(polygon)
        
        if not all_points:
            return []
        
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        return [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]
    
    def _convert_to_dict_list(self, items) -> List[Dict]:
        """Convert list of objects to dictionaries."""
        return [
            {attr: getattr(item, attr) for attr in ['text', 'bbox', 'polygon', 'confidence', 'bbox_valid'] 
             if hasattr(item, attr)}
            if hasattr(item, '__dict__') else item
            for item in items
        ]
    
    def _convert_layout_to_dict(self, layout_predictions) -> List[Dict]:
        return [
            {
                'label': getattr(bbox_info, 'label', ''),
                'bbox': getattr(bbox_info, 'bbox', []),
                'polygon': getattr(bbox_info, 'polygon', []),
                'position': getattr(bbox_info, 'position', 0),
                'confidence': getattr(bbox_info, 'confidence', 0.0)
            }
            for bbox_info in getattr(layout_predictions, 'bboxes', [])
        ]
    
    # ========================================================================
    # Paragraph Grouping and Hierarchical Structure
    # ========================================================================
    
    def group_text_into_paragraphs(self, text_content: List[Dict]) -> List[ParagraphBlock]:
        """Group text blocks into logical paragraphs for NLP processing."""
        if not text_content:
            return []
        
        # Filter out structural elements and focus on content
        content_blocks = [item for item in text_content 
                         if item.get('semantic_role') in ['content', 'mathematical_content'] 
                         and item.get('layout_type') not in ['Page-header', 'Page-footer', 'Caption', 'Footnote']]
        
        if not content_blocks:
            return []
        
        # Sort by reading order (top to bottom, left to right)
        sorted_blocks = sorted(content_blocks, 
                             key=lambda x: (x.get('bbox', [0, 0, 0, 0])[1], 
                                           x.get('bbox', [0, 0, 0, 0])[0]))
        
        paragraphs = []
        current_paragraph = []
        
        for i, block in enumerate(sorted_blocks):
            if not current_paragraph:
                current_paragraph = [block]
                continue
            
            # Check if this block should start a new paragraph
            if self._should_start_new_paragraph(current_paragraph[-1], block, sorted_blocks[max(0, i-2):i]):
                # Finalize current paragraph
                if current_paragraph:
                    paragraph = self._create_paragraph_block(current_paragraph, len(paragraphs))
                    if paragraph:
                        paragraphs.append(paragraph)
                current_paragraph = [block]
            else:
                current_paragraph.append(block)
        
        # Add the last paragraph
        if current_paragraph:
            paragraph = self._create_paragraph_block(current_paragraph, len(paragraphs))
            if paragraph:
                paragraphs.append(paragraph)
        
        return paragraphs
    
    def _should_start_new_paragraph(self, prev_block: Dict, current_block: Dict, context_blocks: List[Dict]) -> bool:
        """Determine if current block should start a new paragraph."""
        prev_bbox = prev_block.get('bbox', [0, 0, 0, 0])
        curr_bbox = current_block.get('bbox', [0, 0, 0, 0])
        
        # Calculate vertical gap
        vertical_gap = curr_bbox[1] - prev_bbox[3]
        
        # Calculate average line height from context
        avg_line_height = self._calculate_average_line_height(context_blocks + [prev_block, current_block])
        
        # Rules for paragraph breaks
        # 1. Large vertical gap (more than 1.5x average line height)
        if vertical_gap > avg_line_height * 1.5:
            return True
        
        # 2. Significant horizontal misalignment (different column or indentation)
        horizontal_offset = abs(curr_bbox[0] - prev_bbox[0])
        if horizontal_offset > 50:  # Significant indentation change
            return True
        
        # 3. Different layout types that indicate paragraph boundaries
        prev_layout = prev_block.get('layout_type', '')
        curr_layout = current_block.get('layout_type', '')
        
        layout_boundary_pairs = [
            ('Text', 'Heading'), ('Text', 'Subheading'),
            ('Heading', 'Text'), ('Subheading', 'Text'),
            ('List-item', 'Text'), ('Text', 'List-item'),
            ('mathematical', 'text'), ('text', 'mathematical')
        ]
        
        if (prev_layout, curr_layout) in layout_boundary_pairs or (curr_layout, prev_layout) in layout_boundary_pairs:
            return True
        
        # 4. Mathematical content boundaries
        prev_type = prev_block.get('type', '')
        curr_type = current_block.get('type', '')
        if prev_type != curr_type and 'mathematical' in [prev_type, curr_type]:
            return True
        
        # 5. Semantic context changes
        prev_keywords = set(prev_block.get('keywords', []))
        curr_keywords = set(current_block.get('keywords', []))
        
        # If there's a significant topic shift (no common keywords and both have keywords)
        if (prev_keywords and curr_keywords and 
            len(prev_keywords.intersection(curr_keywords)) == 0 and 
            len(prev_keywords) > 1 and len(curr_keywords) > 1):
            return True
        
        # 6. Text that starts with paragraph indicators
        curr_text = current_block.get('text', '').strip()
        paragraph_starters = [
            r'^(Definition|Theorem|Proof|Proposition|Lemma|Corollary|Example|Note|Remark)[\s\.:]',
            r'^\d+\.',  # Numbered items
            r'^[A-Z][a-z]*\s+\d+',  # "Chapter 1", "Section 2", etc.
            r'^[A-Z]{2,}[\s\.:]',  # All caps words (headings)
        ]
        
        for pattern in paragraph_starters:
            if re.match(pattern, curr_text, re.IGNORECASE):
                return True
        
        return False
    
    def _calculate_average_line_height(self, blocks: List[Dict]) -> float:
        """Calculate average line height from text blocks."""
        heights = []
        for block in blocks:
            bbox = block.get('bbox', [])
            if len(bbox) >= 4:
                height = bbox[3] - bbox[1]
                if height > 0:
                    heights.append(height)
        
        return sum(heights) / len(heights) if heights else 30.0  # Default fallback
    
    def _create_paragraph_block(self, text_blocks: List[Dict], paragraph_id: int) -> Optional[ParagraphBlock]:
        """Create a ParagraphBlock from grouped text blocks."""
        if not text_blocks:
            return None
        
        # Convert text dicts to TextBlock objects
        text_block_objects = []
        for block in text_blocks:
            bbox = BoundingBox(
                x_min=block.get('bbox', [0, 0, 0, 0])[0],
                y_min=block.get('bbox', [0, 0, 0, 0])[1],
                x_max=block.get('bbox', [0, 0, 0, 0])[2] if len(block.get('bbox', [])) > 2 else 0,
                y_max=block.get('bbox', [0, 0, 0, 0])[3] if len(block.get('bbox', [])) > 3 else 0
            )
            
            text_block = TextBlock(
                text=block.get('text', ''),
                original_text=block.get('original_text', block.get('text', '')),
                bbox=bbox,
                polygon=block.get('polygon', []),
                block_id=f"block_{paragraph_id}_{len(text_block_objects)}",
                block_type=self._map_layout_to_block_type(block.get('layout_type', '')),
                confidence=block.get('confidence', 0.0),
                clean_text=self.text_processor.normalize_text(block.get('text', '')),
                tokens=self._tokenize_text(block.get('text', '')),
                detected_concepts=self._extract_concepts(block.get('text', '')),
                chapter_id=block.get('chapter_id'),
                section_id=block.get('section_id')
            )
            text_block_objects.append(text_block)
        
        # Calculate combined bounding box
        all_bboxes = [block.bbox for block in text_block_objects]
        combined_bbox = self._merge_multiple_bboxes(all_bboxes)
        
        # Combine text with proper spacing
        combined_text = self._combine_text_intelligently(text_block_objects)
        
        # Calculate paragraph-level metrics
        line_spacing = self._calculate_line_spacing(text_block_objects)
        alignment = self._determine_alignment(text_block_objects)
        confidence = sum(block.confidence or 0 for block in text_block_objects) / len(text_block_objects)
        
        # Determine paragraph type
        paragraph_type = self._determine_paragraph_type(text_block_objects)
        
        return ParagraphBlock(
            paragraph_id=f"paragraph_{paragraph_id}",
            text_lines=text_block_objects,
            combined_text=combined_text,
            bbox=combined_bbox,
            line_spacing=line_spacing,
            alignment=alignment,
            confidence=confidence,
            block_type=paragraph_type,
            clean_text=self.text_processor.normalize_text(combined_text),
            detected_concepts=self._extract_concepts(combined_text)
        )
    
    def _map_layout_to_block_type(self, layout_type: str) -> BlockType:
        """Map layout type to BlockType enum."""
        mapping = {
            'Title': BlockType.HEADING,
            'Heading': BlockType.HEADING,
            'Subheading': BlockType.HEADING,
            'Text': BlockType.PARAGRAPH,
            'Caption': BlockType.CAPTION,
            'List-item': BlockType.LIST_ITEM,
            'Footnote': BlockType.FOOTER,
            'Page-header': BlockType.HEADER,
            'Page-footer': BlockType.FOOTER,
            'tableofcontents': BlockType.TOC_ENTRY,
            'table-of-contents': BlockType.TOC_ENTRY,
            'toc': BlockType.TOC_ENTRY
        }
        return mapping.get(layout_type, BlockType.UNKNOWN)
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Simple tokenization for NLP readiness."""
        normalized = self.text_processor.normalize_text(text)
        # Basic tokenization - can be enhanced with proper NLP libraries
        tokens = re.findall(r'\b\w+\b', normalized.lower())
        return tokens
    
    def _extract_concepts(self, text: str) -> List[DetectedConcept]:
        """Extract mathematical and academic concepts."""
        concepts = []
        
        # Mathematical concepts
        math_patterns = {
            'equation': r'(?:equation|formula|expression)',
            'function': r'(?:function|mapping|transformation)',
            'set': r'(?:set|collection|family)',
            'proof': r'(?:proof|demonstration|argument)',
            'theorem': r'(?:theorem|proposition|lemma|corollary)',
            'definition': r'(?:definition|define|defined as)',
            'property': r'(?:property|characteristic|attribute)',
            'relation': r'(?:relation|relationship|correspondence)'
        }
        
        text_lower = text.lower()
        for concept_name, pattern in math_patterns.items():
            if re.search(pattern, text_lower):
                concepts.append(DetectedConcept(
                    name=concept_name,
                    confidence=0.8,
                    context=text[:100] + '...' if len(text) > 100 else text,
                    category='mathematical'
                ))
        
        return concepts
    
    def _merge_multiple_bboxes(self, bboxes: List[BoundingBox]) -> BoundingBox:
        """Merge multiple bounding boxes."""
        if not bboxes:
            return BoundingBox(0, 0, 0, 0)
        
        min_x = min(bbox.x_min for bbox in bboxes)
        min_y = min(bbox.y_min for bbox in bboxes)
        max_x = max(bbox.x_max for bbox in bboxes)
        max_y = max(bbox.y_max for bbox in bboxes)
        
        return BoundingBox(min_x, min_y, max_x, max_y)
    
    def _combine_text_intelligently(self, text_blocks: List[TextBlock]) -> str:
        """Combine text blocks with intelligent spacing and punctuation."""
        if not text_blocks:
            return ""
        
        combined_parts = []
        
        for i, block in enumerate(text_blocks):
            text = block.text.strip()
            if not text:
                continue
            
            # Add spacing based on context
            if i > 0 and combined_parts:
                prev_text = combined_parts[-1]
                
                # Check if we need a space or new sentence
                if not prev_text.endswith(('.', '!', '?', ':', ';')):
                    if not text[0].isupper() and not prev_text.endswith('-'):
                        text = ' ' + text
                    elif text[0].isupper():
                        # Likely start of new sentence
                        text = '. ' + text
                elif prev_text.endswith('-') and text[0].islower():
                    # Hyphenated word continuation
                    pass
                else:
                    text = ' ' + text
            
            combined_parts.append(text)
        
        return ''.join(combined_parts)
    
    def _calculate_line_spacing(self, text_blocks: List[TextBlock]) -> float:
        """Calculate average line spacing in the paragraph."""
        if len(text_blocks) < 2:
            return 0.0
        
        spacings = []
        for i in range(1, len(text_blocks)):
            prev_block = text_blocks[i-1]
            curr_block = text_blocks[i]
            
            spacing = curr_block.bbox.y_min - prev_block.bbox.y_max
            if spacing >= 0:
                spacings.append(spacing)
        
        return sum(spacings) / len(spacings) if spacings else 0.0
    
    def _determine_alignment(self, text_blocks: List[TextBlock]) -> str:
        """Determine text alignment (left, center, right, justified)."""
        if not text_blocks:
            return 'unknown'
        
        left_margins = [block.bbox.x_min for block in text_blocks]
        right_margins = [block.bbox.x_max for block in text_blocks]
        
        left_variance = max(left_margins) - min(left_margins)
        right_variance = max(right_margins) - min(right_margins)
        
        if left_variance < 10 and right_variance < 10:
            return 'justified'
        elif left_variance < 10:
            return 'left'
        elif right_variance < 10:
            return 'right'
        elif abs(sum(left_margins)/len(left_margins) - sum(right_margins)/len(right_margins)) < 50:
            return 'center'
        else:
            return 'irregular'
    
    def _determine_paragraph_type(self, text_blocks: List[TextBlock]) -> BlockType:
        """Determine the semantic type of the paragraph."""
        if not text_blocks:
            return BlockType.UNKNOWN
        
        # Check for mathematical content
        math_count = sum(1 for block in text_blocks if 'mathematical' in str(block.detected_concepts))
        if math_count > len(text_blocks) * 0.3:
            return BlockType.EQUATION
        
        # Check for heading patterns
        first_block_text = text_blocks[0].text.strip()
        if (len(text_blocks) == 1 and 
            (first_block_text.isupper() or 
             re.match(r'^(Chapter|Section|\d+\.)', first_block_text))):
            return BlockType.HEADING
        
        # Check for list items
        if any(re.match(r'^[\d\w]+[\.\)]\s', block.text.strip()) for block in text_blocks):
            return BlockType.LIST_ITEM
        
        return BlockType.PARAGRAPH
    
    def create_hierarchical_structure(self, paragraphs: List[ParagraphBlock]) -> Dict[str, Any]:
        """Create hierarchical document structure for NLP processing."""
        if not paragraphs:
            return {'sections': [], 'metadata': {}}
        
        structure = {
            'document_hierarchy': [],
            'sections': [],
            'concepts': [],
            'metadata': {
                'total_paragraphs': len(paragraphs),
                'paragraph_types': {},
                'concept_categories': {},
                'readiness_score': 0.0
            }
        }
        
        current_section = None
        section_id = 0
        
        for paragraph in paragraphs:
            # Update paragraph type statistics
            ptype = paragraph.block_type.value
            structure['metadata']['paragraph_types'][ptype] = \
                structure['metadata']['paragraph_types'].get(ptype, 0) + 1
            
            # Check if this paragraph starts a new section
            if paragraph.block_type == BlockType.HEADING:
                # Finalize previous section
                if current_section:
                    structure['sections'].append(current_section)
                
                # Start new section
                section_id += 1
                current_section = {
                    'section_id': f'section_{section_id}',
                    'title': paragraph.combined_text,
                    'paragraphs': [],
                    'concepts': [],
                    'bbox': asdict(paragraph.bbox),
                    'hierarchy_level': self._determine_hierarchy_level(paragraph.combined_text)
                }
                
                structure['document_hierarchy'].append({
                    'type': 'section',
                    'title': paragraph.combined_text,
                    'level': current_section['hierarchy_level'],
                    'paragraph_id': paragraph.paragraph_id
                })
            
            # Add paragraph to current section
            if not current_section:
                section_id += 1
                current_section = {
                    'section_id': f'section_{section_id}',
                    'title': 'Introduction',
                    'paragraphs': [],
                    'concepts': [],
                    'bbox': asdict(paragraph.bbox),
                    'hierarchy_level': 1
                }
            
            # Convert paragraph to serializable format
            paragraph_dict = {
                'paragraph_id': paragraph.paragraph_id,
                'combined_text': paragraph.combined_text,
                'clean_text': paragraph.clean_text,
                'block_type': paragraph.block_type.value,
                'bbox': asdict(paragraph.bbox),
                'confidence': paragraph.confidence,
                'line_spacing': paragraph.line_spacing,
                'alignment': paragraph.alignment,
                'text_lines': [
                    {
                        'text': block.text,
                        'clean_text': block.clean_text,
                        'tokens': block.tokens,
                        'bbox': asdict(block.bbox),
                        'confidence': block.confidence,
                        'block_type': block.block_type.value,
                        'concepts': [asdict(concept) for concept in block.detected_concepts]
                    }
                    for block in paragraph.text_lines
                ],
                'detected_concepts': [asdict(concept) for concept in paragraph.detected_concepts]
            }
            
            current_section['paragraphs'].append(paragraph_dict)
            
            # Collect concepts
            for concept in paragraph.detected_concepts:
                current_section['concepts'].append(asdict(concept))
                structure['concepts'].append(asdict(concept))
                
                category = concept.category or 'general'
                structure['metadata']['concept_categories'][category] = \
                    structure['metadata']['concept_categories'].get(category, 0) + 1
        
        # Add final section
        if current_section:
            structure['sections'].append(current_section)
        
        # Calculate readiness score
        structure['metadata']['readiness_score'] = self._calculate_nlp_readiness_score(structure)
        
        return structure
    
    def _determine_hierarchy_level(self, text: str) -> int:
        """Determine hierarchy level from text content."""
        text = text.strip()
        
        # Chapter level
        if re.match(r'^Chapter\s+\d+', text, re.IGNORECASE):
            return 1
        
        # Section patterns
        section_patterns = [
            r'^\d+\.\s',  # "1. Title"
            r'^\d+\.\d+\s',  # "1.1 Title"
            r'^[A-Z]+\s+\d+',  # "SECTION 1"
        ]
        
        for i, pattern in enumerate(section_patterns):
            if re.match(pattern, text):
                return i + 2
        
        # Default to subsection level
        return 3
    
    def _calculate_nlp_readiness_score(self, structure: Dict) -> float:
        """Calculate a score indicating how ready the text is for NLP processing."""
        score = 0.0
        max_score = 100.0
        
        # Paragraph structure quality (30 points)
        total_paragraphs = structure['metadata']['total_paragraphs']
        if total_paragraphs > 0:
            paragraph_score = min(30, total_paragraphs * 3)  # Up to 10 paragraphs
            score += paragraph_score
        
        # Hierarchical structure (25 points)
        sections_count = len(structure['sections'])
        if sections_count > 0:
            hierarchy_score = min(25, sections_count * 5)  # Up to 5 sections
            score += hierarchy_score
        
        # Concept extraction (25 points)
        concepts_count = len(structure['concepts'])
        if concepts_count > 0:
            concept_score = min(25, concepts_count * 2.5)  # Up to 10 concepts
            score += concept_score
        
        # Text cleanliness and tokenization (20 points)
        text_quality_score = 20  # Assume good quality for now
        score += text_quality_score
        
        return (score / max_score) * 100.0

    # ========================================================================
    # Structured Output Generation
    # ========================================================================
    
    def create_page_metadata(self, pdf_path: str, page_num: int) -> Dict:
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
            'dpi': self.dpi,
            'processing_parameters': {
                'containment_threshold': self.containment_threshold,
                'iou_threshold': self.iou_threshold
            }
        }
        
        return metadata
    
    def detect_language(self, text_content: List[Dict]) -> str:
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
    
    def extract_knowledge_structure(self, text_content: List[Dict]) -> Dict[str, Any]:
        """Extract hierarchical knowledge structure."""
        sorted_content = sorted(text_content, 
                              key=lambda x: (x.get('hierarchy_level', 5), 
                                           x.get('layout_position', 0)))
        
        structure = {
            'title': None,
            'headings': [],
            'sections': [],
            'metadata': {
                'total_elements': len(text_content),
                'hierarchy_distribution': {}
            }
        }
        
        current_section = None
        
        for item in sorted_content:
            layout_type = item.get('layout_type', 'unassigned')
            hierarchy_level = item.get('hierarchy_level', 5)
            text = item.get('text', '').strip()
            
            dist = structure['metadata']['hierarchy_distribution']
            dist[layout_type] = dist.get(layout_type, 0) + 1
            
            if hierarchy_level == 0:
                structure['title'] = {
                    'text': text,
                    'confidence': item.get('confidence', 0.0),
                    'bbox': item.get('bbox', [])
                }
            elif hierarchy_level == 1:
                current_section = {
                    'heading': text,
                    'content': [],
                    'confidence': item.get('confidence', 0.0),
                    'bbox': item.get('bbox', [])
                }
                structure['headings'].append(text)
                structure['sections'].append(current_section)
            elif hierarchy_level >= 2 and current_section:
                current_section['content'].append({
                    'text': text,
                    'type': layout_type,
                    'confidence': item.get('confidence', 0.0),
                    'is_mathematical': item.get('type') == 'mathematical',
                    'bbox': item.get('bbox', [])
                })
        
        return structure
    
    def combine_results(self, ocr_predictions, layout_predictions, 
                    image_segments: List[Dict], page_num: int, pdf_path: str = None) -> Dict:
        """Combine OCR and layout analysis results with enhanced features and paragraph grouping."""
        layout_elements = self._convert_layout_to_dict(layout_predictions)
        text_content = self.process_text_content(ocr_predictions)
        enhanced_text = self.associate_text_with_layout(text_content, layout_elements)
        
        page_metadata = self.create_page_metadata(pdf_path or 'unknown.pdf', page_num)
        page_metadata['language'] = self.detect_language(enhanced_text)
        
        is_toc_page = self.is_toc_content(enhanced_text, layout_elements)
        toc_text_elements = self.extract_toc_elements(enhanced_text) if is_toc_page else []
        toc_entries = self.clean_and_extract_toc_entries(toc_text_elements) if toc_text_elements else []
        
        # NEW: Group text into logical paragraphs for NLP processing
        paragraphs = self.group_text_into_paragraphs(enhanced_text)
        hierarchical_structure = self.create_hierarchical_structure(paragraphs)
        
        # Convert to structured format for JSON serialization
        structured_toc_entries = []
        for entry in toc_entries:
            structured_toc_entries.append({
                'number': entry.number,
                'title': entry.title,
                'page': entry.page,
                'level': entry.level,
                'full_text': entry.full_text
            })
        
        # Convert paragraphs to serializable format
        structured_paragraphs = []
        for paragraph in paragraphs:
            structured_paragraphs.append({
                'paragraph_id': paragraph.paragraph_id,
                'combined_text': paragraph.combined_text,
                'clean_text': paragraph.clean_text,
                'block_type': paragraph.block_type.value,
                'bbox': asdict(paragraph.bbox),
                'confidence': paragraph.confidence,
                'line_spacing': paragraph.line_spacing,
                'alignment': paragraph.alignment,
                'text_lines': [
                    {
                        'text': block.text,
                        'clean_text': block.clean_text,
                        'tokens': block.tokens,
                        'bbox': asdict(block.bbox),
                        'confidence': block.confidence,
                        'block_type': block.block_type.value,
                        'concepts': [asdict(concept) for concept in block.detected_concepts]
                    }
                    for block in paragraph.text_lines
                ],
                'detected_concepts': [asdict(concept) for concept in paragraph.detected_concepts]
            })
        
        return {
            'metadata': page_metadata,
            'page_number': page_num,
            'image_bbox': getattr(ocr_predictions, 'image_bbox', []),
            'text_content': enhanced_text,
            'paragraphs': structured_paragraphs,
            'hierarchical_structure': hierarchical_structure,
            'image_segments': image_segments,
            'layout_elements': layout_elements,
            'knowledge_structure': self.extract_knowledge_structure(enhanced_text),
            'is_toc_page': is_toc_page,
            'toc_text_elements': [{
                'text': elem.get('text', ''),
                'bbox': elem.get('bbox', []),
                'confidence': elem.get('confidence', 0.0),
                'layout_type': elem.get('layout_type', ''),
                'association_score': elem.get('association_score', 0.0)
            } for elem in toc_text_elements],
            'toc_entries': structured_toc_entries,
            'statistics': {
                'total_text_elements': len(enhanced_text),
                'total_paragraphs': len(paragraphs),
                'total_layout_elements': len(layout_elements),
                'total_image_segments': len(image_segments),
                'total_toc_entries': len(toc_entries),
                'total_toc_text_elements': len(toc_text_elements),
                'association_success_rate': len([t for t in enhanced_text 
                                                if t.get('layout_type') != 'unassigned']) / len(enhanced_text) 
                                                if enhanced_text else 0,
                'nlp_readiness_score': hierarchical_structure.get('metadata', {}).get('readiness_score', 0.0),
                'paragraph_types': hierarchical_structure.get('metadata', {}).get('paragraph_types', {}),
                'concept_categories': hierarchical_structure.get('metadata', {}).get('concept_categories', {})
            }
        }
    
    # ========================================================================
    # Document Processing Pipeline
    # ========================================================================
    
    def process_document(self, pdf_path: str, output_dir: str, 
                        page_numbers: Optional[List[int]] = None) -> Dict[str, Any]:
        """Process entire document with comprehensive analysis."""
        os.makedirs(output_dir, exist_ok=True)
        images = self.pdf_to_images(pdf_path, page_numbers)
        if not images:
            raise ValueError("No valid pages found in PDF")
        
        results = {}
        actual_page_numbers = page_numbers or list(range(len(images)))
        
        for image, page_num in zip(images, actual_page_numbers):
            print(f"Processing page {page_num + 1}...")
            
            ocr_predictions = self.recognition_predictor([image], det_predictor=self.detection_predictor)[0] or None
            layout_predictions = self.layout_predictor([image])[0] or None
            
            if not ocr_predictions or not layout_predictions:
                print(f"  Warning: No results for page {page_num + 1}")
                continue
            
            image_segments = self.save_image_segments(image, layout_predictions, page_num, output_dir)
            page_result = self.combine_results(ocr_predictions, layout_predictions, image_segments, page_num + 1, pdf_path)
            results[f"page_{page_num + 1}"] = page_result
        
        page_str = '_'.join(str(n+1) for n in actual_page_numbers)
        results_file = os.path.join(output_dir, f'page_{page_str}_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        summary = self._create_document_summary(results)
        summary_file = os.path.join(output_dir, 'knowledge_structure_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return results
    
    def _create_document_summary(self, results: Dict) -> Dict:
        """Create document summary with enhanced paragraph and NLP readiness metrics."""
        summary = {
            'document_title': None,
            'total_pages': len(results),
            'all_headings': [],
            'hierarchy_distribution': {},
            'mathematical_content_ratio': 0.0,
            'average_association_success': 0.0,
            'paragraph_summary': {
                'total_paragraphs': 0,
                'paragraph_types': {},
                'average_paragraph_length': 0.0,
                'alignment_distribution': {},
                'concept_density': 0.0
            },
            'nlp_readiness': {
                'average_readiness_score': 0.0,
                'total_concepts': 0,
                'concept_categories': {},
                'document_structure_quality': 'unknown'
            },
            'toc_summary': {
                'total_toc_pages': 0,
                'total_toc_entries': 0,
                'toc_hierarchy_levels': {},
                'toc_entries': []
            }
        }
        
        total_elements = 0
        total_math = 0
        total_success = 0.0
        total_readiness = 0.0
        total_paragraph_length = 0
        total_concepts = 0
        
        for page_data in results.values():
            ks = page_data.get('knowledge_structure', {})
            
            if not summary['document_title'] and ks.get('title'):
                summary['document_title'] = ks['title']['text']
            
            summary['all_headings'].extend(ks.get('headings', []))
            
            for k, v in ks.get('metadata', {}).get('hierarchy_distribution', {}).items():
                summary['hierarchy_distribution'][k] = summary['hierarchy_distribution'].get(k, 0) + v
            
            text_content = page_data.get('text_content', [])
            page_total = len(text_content)
            page_math = len([t for t in text_content if t.get('type') == 'mathematical'])
            total_elements += page_total
            total_math += page_math
            
            total_success += page_data.get('statistics', {}).get('association_success_rate', 0.0)
            
            # Process paragraph information
            paragraphs = page_data.get('paragraphs', [])
            summary['paragraph_summary']['total_paragraphs'] += len(paragraphs)
            
            for paragraph in paragraphs:
                # Paragraph type distribution
                ptype = paragraph.get('block_type', 'unknown')
                summary['paragraph_summary']['paragraph_types'][ptype] = \
                    summary['paragraph_summary']['paragraph_types'].get(ptype, 0) + 1
                
                # Alignment distribution
                alignment = paragraph.get('alignment', 'unknown')
                summary['paragraph_summary']['alignment_distribution'][alignment] = \
                    summary['paragraph_summary']['alignment_distribution'].get(alignment, 0) + 1
                
                # Track paragraph length for average
                para_length = len(paragraph.get('combined_text', ''))
                total_paragraph_length += para_length
                
                # Count concepts
                concepts = paragraph.get('detected_concepts', [])
                total_concepts += len(concepts)
            
            # Process hierarchical structure and NLP readiness
            hierarchical = page_data.get('hierarchical_structure', {})
            readiness_score = hierarchical.get('metadata', {}).get('readiness_score', 0.0)
            total_readiness += readiness_score
            
            # Concept categories
            concept_categories = hierarchical.get('metadata', {}).get('concept_categories', {})
            for category, count in concept_categories.items():
                summary['nlp_readiness']['concept_categories'][category] = \
                    summary['nlp_readiness']['concept_categories'].get(category, 0) + count
            
            # TOC processing
            if page_data.get('is_toc_page', False):
                summary['toc_summary']['total_toc_pages'] += 1
                toc_entries = page_data.get('toc_entries', [])
                summary['toc_summary']['total_toc_entries'] += len(toc_entries)
                
                for entry in toc_entries:
                    level = entry.get('level', 1)
                    summary['toc_summary']['toc_hierarchy_levels'][level] = \
                        summary['toc_summary']['toc_hierarchy_levels'].get(level, 0) + 1
                
                summary['toc_summary']['toc_entries'].extend(toc_entries)
        
        # Calculate averages and ratios
        if total_elements > 0:
            summary['mathematical_content_ratio'] = total_math / total_elements
        if results:
            summary['average_association_success'] = total_success / len(results)
            summary['nlp_readiness']['average_readiness_score'] = total_readiness / len(results)
        
        if summary['paragraph_summary']['total_paragraphs'] > 0:
            summary['paragraph_summary']['average_paragraph_length'] = \
                total_paragraph_length / summary['paragraph_summary']['total_paragraphs']
            summary['paragraph_summary']['concept_density'] = \
                total_concepts / summary['paragraph_summary']['total_paragraphs']
        
        summary['nlp_readiness']['total_concepts'] = total_concepts
        
        # Determine document structure quality
        readiness_score = summary['nlp_readiness']['average_readiness_score']
        if readiness_score >= 80:
            summary['nlp_readiness']['document_structure_quality'] = 'excellent'
        elif readiness_score >= 60:
            summary['nlp_readiness']['document_structure_quality'] = 'good'
        elif readiness_score >= 40:
            summary['nlp_readiness']['document_structure_quality'] = 'fair'
        else:
            summary['nlp_readiness']['document_structure_quality'] = 'poor'
        
        return summary


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    import argparse
    
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
        avg_nlp_readiness = sum(p['statistics'].get('nlp_readiness_score', 0) for p in results.values()) / total_pages if total_pages > 0 else 0
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
        print(f"NLP readiness score: {avg_nlp_readiness:.1f}/100")
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