"""Paragraph grouping and hierarchical structure creation."""

import re
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import asdict

from ..models.data_structures import TextBlock, ParagraphBlock, BoundingBox, NLPOutputs
from ..models.enums import BlockType
from .text_processor import TextProcessor


class ParagraphGrouper:
    """Paragraph grouping and hierarchical structure creation functionality"""
    
    def __init__(self):
        self.text_processor = TextProcessor()
    
    def group_text_into_paragraphs(self, text_content: List[Dict]) -> List[ParagraphBlock]:
        """Group text blocks into logical paragraphs for NLP processing with improved bbox-based grouping."""
        if not text_content:
            return []
        
        # Filter out structural elements and focus on content, but be more inclusive
        content_blocks = [item for item in text_content 
                         if (item.get('semantic_role') in ['content', 'mathematical_content'] or
                             item.get('layout_type') in ['Text', 'List-item']) and
                         item.get('layout_type') not in ['Page-header', 'Page-footer', 'Footnote'] and
                         len(item.get('text', '').strip()) > 0]  # Ensure non-empty text
        
        if not content_blocks:
            return []
        
        # Enhanced sorting by reading order with better spatial awareness
        def reading_order_key(block):
            bbox = block.get('bbox', [0, 0, 0, 0])
            # Primary sort by Y position (top to bottom)
            # Secondary sort by X position (left to right)  
            # Tertiary sort by layout hierarchy to handle edge cases
            hierarchy_level = block.get('hierarchy_level', 5)
            return (bbox[1], bbox[0], hierarchy_level)
        
        sorted_blocks = sorted(content_blocks, key=reading_order_key)
        
        paragraphs = []
        current_paragraph = []
        
        for i, block in enumerate(sorted_blocks):
            if not current_paragraph:
                current_paragraph = [block]
                continue
            
            # Get broader context for better decision making
            context_start = max(0, i - 5)  # Look at more context
            context_end = min(len(sorted_blocks), i)
            context_blocks = sorted_blocks[context_start:context_end]
            
            # Check if this block should start a new paragraph
            should_break = self._should_start_new_paragraph(
                current_paragraph[-1], 
                block, 
                context_blocks
            )
            
            if should_break:
                # Before finalizing, check if we should include any skipped intermediate blocks
                intermediate_blocks = self._find_intermediate_blocks(
                    current_paragraph, block, sorted_blocks[max(0, i-10):i]
                )
                
                if intermediate_blocks:
                    current_paragraph.extend(intermediate_blocks)
                
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
        """Determine if current block should start a new paragraph with improved bbox-based text continuity detection."""
        prev_bbox = prev_block.get('bbox', [0, 0, 0, 0])
        curr_bbox = current_block.get('bbox', [0, 0, 0, 0])
        prev_text = prev_block.get('text', '').strip()
        curr_text = current_block.get('text', '').strip()
        
        # Calculate bbox-based metrics more accurately
        prev_height = prev_bbox[3] - prev_bbox[1]
        curr_height = curr_bbox[3] - curr_bbox[1]
        
        # Vertical gap should account for potential overlap
        vertical_gap = curr_bbox[1] - prev_bbox[3]
        
        # Check for vertical overlap (indicates same line or very close lines)
        vertical_overlap = max(0, min(prev_bbox[3], curr_bbox[3]) - max(prev_bbox[1], curr_bbox[1]))
        
        # Horizontal alignment and offset
        horizontal_offset = abs(curr_bbox[0] - prev_bbox[0])
        horizontal_overlap = max(0, min(prev_bbox[2], curr_bbox[2]) - max(prev_bbox[0], curr_bbox[0]))
        
        # Calculate line spacing relative to text height
        avg_line_height = self._calculate_average_line_height(context_blocks + [prev_block, current_block])
        relative_line_spacing = vertical_gap / avg_line_height if avg_line_height > 0 else 0
        
        # RULE 1: Strong bbox-based continuity (PRIORITY)
        # If text blocks have significant vertical overlap or very small gap, they should continue
        if vertical_overlap > min(prev_height, curr_height) * 0.3:
            return False  # Overlapping or adjacent lines - continue paragraph
        
        # Very small vertical gaps relative to line height indicate continuation
        if vertical_gap >= 0 and relative_line_spacing < 0.3:
            return False  # Very close lines - continue paragraph
        
        # RULE 2: Text continuity analysis (secondary check)
        # Check if current text appears to continue previous text
        text_continues = self._is_text_continuation(prev_text, curr_text)
        if text_continues and vertical_gap < avg_line_height * 1.5:
            return False  # Text continues and gap is reasonable
        
        # RULE 3: Strong paragraph break indicators
        # Large vertical gap indicates clear paragraph break
        if relative_line_spacing > 1.5:
            return True
        
        # Significant indentation change for new paragraph
        # But allow for minor alignment variations in same paragraph
        if horizontal_offset > avg_line_height * 2:  # Scale with text size
            return True
        
        # RULE 4: Layout-based boundaries
        prev_layout = prev_block.get('layout_type', '')
        curr_layout = current_block.get('layout_type', '')
        
        # Strong layout boundaries
        strong_boundaries = [
            ('Text', 'Heading'), ('Text', 'Subheading'), ('Text', 'Title'),
            ('Heading', 'Text'), ('Subheading', 'Text'), ('Title', 'Text'),
            ('List-item', 'Text'), ('Text', 'List-item'),
            ('Caption', 'Text'), ('Text', 'Caption'),
            ('Footnote', 'Text'), ('Text', 'Footnote')
        ]
        
        if (prev_layout, curr_layout) in strong_boundaries:
            return True
        
        # RULE 5: Structural text indicators
        structural_patterns = [
            r'^(Chapter|Section|Subsection|Part)\s+\d+',  # Chapter/Section headers
            r'^(Definition|Theorem|Proof|Proposition|Lemma|Corollary|Example)[\s\d\.:]',  # Mathematical structures
            r'^(Figure|Table|Equation)\s+\d+',  # Reference labels
            r'^\d+\.\s',  # Numbered lists
            r'^[A-Z]{3,}[\s\.:]',  # All caps headings
        ]
        
        for pattern in structural_patterns:
            if re.match(pattern, curr_text, re.IGNORECASE):
                return True
        
        # RULE 6: Medium gap decision with improved logic
        # Break on medium gaps if bbox and text analysis suggest separation
        if (relative_line_spacing > 0.8 and 
            not self._is_likely_same_paragraph(prev_text, curr_text) and
            not text_continues):
            return True
        
        # RULE 7: Reading order consistency check
        # Ensure we don't skip lines that are spatially between current paragraph lines
        if self._check_reading_order_continuity(prev_block, current_block, context_blocks):
            return False
        
        return False
    
    def _find_intermediate_blocks(self, current_paragraph: List[Dict], next_block: Dict, candidate_blocks: List[Dict]) -> List[Dict]:
        """Find blocks that should be included in the current paragraph before moving to next_block."""
        if not current_paragraph or not candidate_blocks:
            return []
        
        last_block = current_paragraph[-1]
        last_bbox = last_block.get('bbox', [0, 0, 0, 0])
        next_bbox = next_block.get('bbox', [0, 0, 0, 0])
        
        # Find blocks that are spatially between the last block of current paragraph and next block
        intermediate_blocks = []
        
        for candidate in candidate_blocks:
            # Skip if already in current paragraph
            if candidate in current_paragraph:
                continue
                
            candidate_bbox = candidate.get('bbox', [0, 0, 0, 0])
            
            # Check if candidate is vertically between last block and next block
            if (last_bbox[1] <= candidate_bbox[1] <= next_bbox[1] or
                last_bbox[3] <= candidate_bbox[3] <= next_bbox[3]):
                
                # Check horizontal alignment - should overlap with paragraph content
                paragraph_left = min(block.get('bbox', [0,0,0,0])[0] for block in current_paragraph)
                paragraph_right = max(block.get('bbox', [0,0,0,0])[2] for block in current_paragraph)
                
                candidate_left = candidate_bbox[0]
                candidate_right = candidate_bbox[2]
                
                # Check if candidate has reasonable horizontal overlap with paragraph
                horizontal_overlap = max(0, min(paragraph_right, candidate_right) - max(paragraph_left, candidate_left))
                paragraph_width = paragraph_right - paragraph_left
                
                if horizontal_overlap > paragraph_width * 0.1:  # At least 10% overlap
                    # Additional content check
                    candidate_text = candidate.get('text', '').strip()
                    if (len(candidate_text) > 3 and 
                        candidate.get('semantic_role') in ['content', 'mathematical_content']):
                        intermediate_blocks.append(candidate)
        
        # Sort intermediate blocks by vertical position
        intermediate_blocks.sort(key=lambda x: x.get('bbox', [0,0,0,0])[1])
        return intermediate_blocks
    
    def _check_reading_order_continuity(self, prev_block: Dict, current_block: Dict, context_blocks: List[Dict]) -> bool:
        """Check if there are intermediate text blocks that should be included in the current paragraph."""
        if not context_blocks:
            return False
        
        prev_bbox = prev_block.get('bbox', [0, 0, 0, 0])
        curr_bbox = current_block.get('bbox', [0, 0, 0, 0])
        
        # Find blocks that are spatially between prev and current blocks
        intermediate_blocks = []
        for block in context_blocks:
            block_bbox = block.get('bbox', [0, 0, 0, 0])
            
            # Check if block is vertically between prev and current
            if (prev_bbox[1] < block_bbox[1] < curr_bbox[1] or  # Y position between
                prev_bbox[3] < block_bbox[3] < curr_bbox[3]):   # Bottom Y between
                
                # Check if it's also horizontally aligned (within reasonable range)
                horizontal_overlap_with_prev = max(0, min(prev_bbox[2], block_bbox[2]) - max(prev_bbox[0], block_bbox[0]))
                horizontal_overlap_with_curr = max(0, min(curr_bbox[2], block_bbox[2]) - max(curr_bbox[0], block_bbox[0]))
                
                # If the intermediate block has significant horizontal overlap with either prev or current
                if (horizontal_overlap_with_prev > 0 or horizontal_overlap_with_curr > 0):
                    intermediate_blocks.append(block)
        
        # If there are intermediate blocks, we should continue the paragraph to include them
        if intermediate_blocks:
            # Additional check: make sure intermediate blocks contain meaningful content
            meaningful_intermediates = [
                block for block in intermediate_blocks 
                if (len(block.get('text', '').strip()) > 3 and 
                    block.get('semantic_role') in ['content', 'mathematical_content'])
            ]
            return len(meaningful_intermediates) > 0
        
        return False
    
    def _is_text_continuation(self, prev_text: str, curr_text: str) -> bool:
        """Check if current text continues the previous text (same sentence/thought)."""
        if not prev_text or not curr_text:
            return False
        
        prev_text = prev_text.strip()
        curr_text = curr_text.strip()
        
        # Check if previous text ends mid-sentence and current continues
        # Previous text doesn't end with sentence terminators
        if not prev_text[-1] in '.!?':
            # Current text starts with lowercase (strong continuation indicator)
            if curr_text[0].islower():
                return True
            
            # Previous text ends with connecting words/phrases
            continuation_endings = [
                'the', 'a', 'an', 'and', 'or', 'but', 'of', 'in', 'on', 'at', 'to', 
                'for', 'with', 'by', 'that', 'which', 'this', 'these', 'those'
            ]
            
            last_word = prev_text.split()[-1].lower().rstrip('.,;:')
            if last_word in continuation_endings:
                return True
        
        # Additional check: if previous text ends with a period but current starts lowercase,
        # it might be a continuation of the same paragraph (OCR artifact)
        if prev_text[-1] == '.' and curr_text[0].islower():
            return True
        
        return False
    
    def _is_likely_same_paragraph(self, prev_text: str, curr_text: str) -> bool:
        """Check if two text blocks are likely part of the same paragraph."""
        if not prev_text or not curr_text:
            return False
        
        prev_text = prev_text.strip()
        curr_text = curr_text.strip()
        
        # Strong indicators of paragraph continuation
        # 1. Previous text ends mid-sentence
        if not prev_text[-1] in '.!?':
            return True
        
        # 2. Current text starts with lowercase
        if curr_text[0].islower():
            return True
        
        # 3. Both texts are short and likely fragmented
        if len(prev_text) < 200 and len(curr_text) < 200:
            # Look for common academic/mathematical terms that suggest continuity
            common_words = self._get_common_content_words(prev_text, curr_text)
            if len(common_words) > 1:  # Lowered threshold
                return True
        
        # 4. Academic text patterns that suggest continuity
        academic_continuity_patterns = [
            # Current starts with connecting words
            r'^(also|however|furthermore|moreover|therefore|thus|hence|consequently|additionally|similarly|likewise|nevertheless|nonetheless)',
            # Current continues a thought
            r'^(this|that|these|those|such|it|they|which|who|where|when)',
            # Previous ends with introductory phrases
            r'(such as|for example|including|namely|specifically|particularly|especially)\.?$'
        ]
        
        for pattern in academic_continuity_patterns[:2]:  # Check current text patterns
            if re.match(pattern, curr_text, re.IGNORECASE):
                return True
        
        # Check if previous text ends with introductory phrases
        if re.search(academic_continuity_patterns[2], prev_text, re.IGNORECASE):
            return True
        
        # 5. Mathematical/academic context indicators
        math_academic_keywords = [
            'proof', 'theorem', 'equation', 'formula', 'figure', 'property', 'definition',
            'example', 'problem', 'solution', 'method', 'calculation', 'relationship',
            'principle', 'concept', 'idea', 'approach', 'technique', 'result'
        ]
        
        prev_lower = prev_text.lower()
        curr_lower = curr_text.lower()
        
        prev_has_math = any(keyword in prev_lower for keyword in math_academic_keywords)
        curr_has_math = any(keyword in curr_lower for keyword in math_academic_keywords)
        
        if prev_has_math and curr_has_math:
            return True
        
        # 6. Sequential references or numbering
        sequential_patterns = [
            (r'\b(\d+)\b', r'\b(\d+)\b'),  # Numbers
            (r'\b([a-z])\)', r'\b([a-z])\)'),  # Lettered items
        ]
        
        for prev_pattern, curr_pattern in sequential_patterns:
            prev_matches = re.findall(prev_pattern, prev_text.lower())
            curr_matches = re.findall(curr_pattern, curr_text.lower())
            
            if prev_matches and curr_matches:
                try:
                    if int(prev_matches[-1]) + 1 == int(curr_matches[0]):
                        return True
                except (ValueError, IndexError):
                    continue
        
        return False
    
    def _get_common_content_words(self, text1: str, text2: str) -> set:
        """Get common meaningful words between two texts."""
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        words1 = set(word.lower() for word in re.findall(r'\b\w{3,}\b', text1) if word.lower() not in stopwords)
        words2 = set(word.lower() for word in re.findall(r'\b\w{3,}\b', text2) if word.lower() not in stopwords)
        
        return words1.intersection(words2)
    
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
                
                # More sophisticated text joining logic
                if prev_text.endswith('-') and text[0].islower():
                    # Hyphenated word continuation - no space needed
                    pass
                elif not prev_text[-1] in '.!?:;' and text[0].islower():
                    # Previous text doesn't end sentence, current starts lowercase - add space
                    text = ' ' + text
                elif prev_text[-1] in '.!?' and text[0].isupper():
                    # Sentence boundary - add space
                    text = ' ' + text
                elif prev_text[-1] in ',;:' and text[0].islower():
                    # Clause continuation - add space
                    text = ' ' + text
                elif prev_text.split()[-1].lower() in ['the', 'a', 'an', 'and', 'or', 'but', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'that', 'which', 'this', 'these', 'those']:
                    # Previous ends with connecting word - add space
                    text = ' ' + text
                elif not prev_text[-1] in '.!?:;' and text[0].isupper():
                    # Previous doesn't end sentence but current starts capital - likely new sentence
                    # Add period if missing
                    text = '. ' + text
                else:
                    # Default case - add space
                    text = ' ' + text
            
            combined_parts.append(text)
        
        # Post-process the combined text for better readability
        result = ''.join(combined_parts)
        
        # Fix common OCR artifacts
        result = re.sub(r'\s+', ' ', result)  # Multiple spaces to single
        result = re.sub(r'\s+([.,:;!?])', r'\1', result)  # Remove space before punctuation
        result = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', result)  # Ensure space after sentence end
        result = re.sub(r'([.!?]){2,}', r'\1', result)  # Remove repeated punctuation
        
        return result.strip()
    
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
        
        # Check for mathematical content based on text content
        math_indicators = ['=', '+', '-', '*', '/', '^', '∫', '∑', '∏', '√', 'α', 'β', 'γ', 'θ', 'π']
        combined_text = ' '.join(block.text for block in text_blocks).lower()
        math_count = sum(1 for indicator in math_indicators if indicator in combined_text)
        if math_count > 3:  # Threshold for mathematical content
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
                    }
                    for block in paragraph.text_lines
                ],
            }
            
            current_section['paragraphs'].append(paragraph_dict)
            
        
        # Add final section
        if current_section:
            structure['sections'].append(current_section)
        
        
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