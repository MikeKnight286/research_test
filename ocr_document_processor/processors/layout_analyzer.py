"""Layout analysis and text-layout association utilities."""

import re
from typing import List, Dict, Any


class LayoutAnalyzer:
    """Layout analysis and text-layout association functionality"""
    
    def __init__(self, containment_threshold=0.5, iou_threshold=0.1):
        self.containment_threshold = containment_threshold
        self.iou_threshold = iou_threshold
    
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