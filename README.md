# OCR Document Processor

A modular OCR document processing system built on top of [Surya OCR](https://github.com/datalab-to/surya) for research.

## 🚀 Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
ocr_document_processor/
├── __init__.py                 # Package exports
├── models/                     # Data structures & enums
│   ├── __init__.py
│   ├── data_structures.py      # TOCEntry, BoundingBox, TextBlock, ParagraphBlock, NLPOutputs
│   └── enums.py               # BlockType enum
├── core/                      # Main processing logic
│   ├── __init__.py
│   └── main_processor.py      # OCRDocumentProcessor class
├── extractors/                # Content extraction
│   ├── __init__.py
│   └── toc_extractor.py       # TOCExtractor class
├── processors/                # Content processing
│   ├── __init__.py
│   ├── text_processor.py      # TextProcessor class
│   ├── layout_analyzer.py     # LayoutAnalyzer class
│   └── paragraph_grouper.py   # ParagraphGrouper class
├── utils/                     # Helper utilities
│   ├── __init__.py
│   ├── pdf_utils.py          # PDF processing utilities
│   └── image_utils.py        # Image processing helpers
├── output/                    # Result generation
│   ├── __init__.py
│   ├── json_generator.py     # JSON output generation
│   └── summary_generator.py  # Document summary creation
└── cli.py                    # Command-line interface
```

## 🎯 Features

- **OCR Processing**: Extract text from PDF documents
- **Table of Contents Detection**: Automatically identify and extract TOC entries
- **Layout Analysis**: Analyze document structure and identify text blocks
- **Paragraph Grouping**: Group related text blocks into coherent paragraphs
- **JSON Output**: Generate structured JSON results with document analysis
- **Document Summaries**: Create comprehensive document summaries

## 🔄 Usage Examples

### Basic Usage
```python
from ocr_document_processor import OCRDocumentProcessor

processor = OCRDocumentProcessor()
results = processor.process_document("document.pdf", "output/")
```

### Advanced Component Usage
```python
from ocr_document_processor.extractors import TOCExtractor
from ocr_document_processor.processors import TextProcessor
from ocr_document_processor.models import BlockType

# Use individual components
toc_extractor = TOCExtractor()
entries = toc_extractor.extract_toc_entries(text)

text_processor = TextProcessor()
clean_text = text_processor.normalize_text(raw_text)
```

### Custom Component Integration
```python
from ocr_document_processor.core import OCRDocumentProcessor
from my_custom_processors import CustomTOCExtractor

# Replace components with custom implementations
processor = OCRDocumentProcessor(toc_extractor=CustomTOCExtractor())
```

## 🛠️ Command Line Interface

```bash
# Process single page
python -m ocr_document_processor.cli document.pdf --pages 1

# Process multiple pages
python -m ocr_document_processor.cli document.pdf --pages 1,2,3 --output_dir results/

# Using the demo script
python demo_ocr.py
```

## 🎪 Demo

Run the demo to see the processor in action:

```bash
python demo_ocr.py
```

## 📊 Module Overview

| Module | Responsibility |
|--------|----------------|
| `models/` | Data structures and type definitions |
| `extractors/` | TOC extraction and text extraction |
| `processors/` | Text, layout, and paragraph processing |
| `utils/` | PDF/image utilities and helpers |
| `output/` | JSON generation and summaries |
| `core/` | Main orchestration and integration |
| `cli.py` | Command-line interface |


## 🙏 Acknowledgments

This project is built on top of [Surya OCR](https://github.com/datalab-to/surya).