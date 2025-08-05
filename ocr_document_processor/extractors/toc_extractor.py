"""Table of Contents extraction module."""

import re
from typing import List, Optional, Dict, Any
from ..models.data_structures import TOCEntry


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
            # No justification for selection of meaningful_keywords yet
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