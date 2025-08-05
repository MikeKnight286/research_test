"""Text processing utilities for OCR output."""

import re
import html
import unicodedata
from typing import Dict


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