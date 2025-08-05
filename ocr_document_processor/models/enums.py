"""Enumerations for OCR document processing."""

from enum import Enum


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