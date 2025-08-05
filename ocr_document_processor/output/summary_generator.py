"""Document summary generation utilities."""

from typing import Dict, Any


class SummaryGenerator:
    """Document summary generation utilities"""
    
    @staticmethod
    def create_document_summary(results: Dict) -> Dict:
        """Create document summary with enhanced paragraph metrics."""
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
        
        if summary['paragraph_summary']['total_paragraphs'] > 0:
            summary['paragraph_summary']['average_paragraph_length'] = \
                total_paragraph_length / summary['paragraph_summary']['total_paragraphs']
            summary['paragraph_summary']['concept_density'] = \
                total_concepts / summary['paragraph_summary']['total_paragraphs']
        
        
        return summary