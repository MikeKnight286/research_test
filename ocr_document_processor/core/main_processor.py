"""Main OCR Document Processor using modular components."""

import os
import json
import re
from typing import List, Optional, Dict, Any
from dataclasses import asdict

from ..models.data_structures import TOCEntry, BoundingBox, TextBlock, ParagraphBlock
from ..models.enums import BlockType
from ..extractors.toc_extractor import TOCExtractor
from ..processors.text_processor import TextProcessor
from ..processors.layout_analyzer import LayoutAnalyzer
from ..processors.paragraph_grouper import ParagraphGrouper
from ..utils.pdf_utils import PDFUtils
from ..utils.image_utils import ImageUtils
from ..output.json_generator import JSONGenerator
from ..output.summary_generator import SummaryGenerator


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
        
        # Initialize modular components
        self.toc_extractor = toc_extractor or TOCExtractor()
        self.text_processor = TextProcessor()
        self.layout_analyzer = LayoutAnalyzer(containment_threshold, iou_threshold)
        self.paragraph_grouper = ParagraphGrouper()
        self.pdf_utils = PDFUtils()
        self.image_utils = ImageUtils()
        self.json_generator = JSONGenerator()
        self.summary_generator = SummaryGenerator()
        
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
    
    def pdf_to_images(self, pdf_path: str, page_numbers: Optional[List[int]] = None, dpi: int = None) -> List:
        """Convert PDF pages to PIL Images."""
        dpi = dpi or self.dpi
        return self.pdf_utils.pdf_to_images(pdf_path, page_numbers, dpi)
    
    def save_image_segments(self, image, layout_predictions, page_num: int, output_dir: str) -> List[Dict]:
        """Extract and save image segments."""
        return self.image_utils.save_image_segments(image, layout_predictions, page_num, output_dir)
    
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
    # Text Processing
    # ========================================================================
    
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
    # Structured Output Generation
    # ========================================================================
    
    def combine_results(self, ocr_predictions, layout_predictions, 
                    image_segments: List[Dict], page_num: int, pdf_path: str = None) -> Dict:
        """Combine OCR and layout analysis results with enhanced features and paragraph grouping."""
        layout_elements = self._convert_layout_to_dict(layout_predictions)
        text_content = self.process_text_content(ocr_predictions)
        enhanced_text = self.layout_analyzer.associate_text_with_layout(text_content, layout_elements)
        
        page_metadata = self.json_generator.create_page_metadata(
            pdf_path or 'unknown.pdf', page_num, self.dpi, 
            self.containment_threshold, self.iou_threshold
        )
        page_metadata['language'] = self.json_generator.detect_language(enhanced_text)
        
        is_toc_page = self.is_toc_content(enhanced_text, layout_elements)
        toc_text_elements = self.extract_toc_elements(enhanced_text) if is_toc_page else []
        toc_entries = self.clean_and_extract_toc_entries(toc_text_elements) if toc_text_elements else []
        
        # NEW: Group text into logical paragraphs for NLP processing
        paragraphs = self.paragraph_grouper.group_text_into_paragraphs(enhanced_text)
        hierarchical_structure = self.paragraph_grouper.create_hierarchical_structure(paragraphs)
        
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
                    }
                    for block in paragraph.text_lines
                ],
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
            'knowledge_structure': self.layout_analyzer.extract_knowledge_structure(enhanced_text),
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
        
        summary = self.summary_generator.create_document_summary(results)
        summary_file = os.path.join(output_dir, 'knowledge_structure_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return results