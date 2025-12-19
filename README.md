# Local OCR Demo

A document processing pipeline that uses NVIDIA Vision-Language Models for OCR and Nemotron LLM for entity extraction.

## Architecture

```
PDF → Image (Poppler) → Vision OCR (Llama 3.2 90B) → Entity Extraction (Nemotron 30B) → Structured Output
```

## Features

- **Multi-Page Processing**: Processes ALL pages in PDF documents
- **Vision-Based OCR**: Uses Llama 3.2 90B Vision model for accurate text extraction
- **Checkbox Detection**: Identifies checked/unchecked boxes in forms
- **Handwritten Text**: Extracts and marks handwritten content
- **Confidence Scores**: Each extracted entity includes a confidence level
- **Page Source Tracking**: Know which page each entity came from
- **Business Document Optimized**: Designed for invoices, forms, applications

## Prerequisites

- Python 3.12
- Poppler (for PDF to image conversion)
- NVIDIA API Key from [build.nvidia.com](https://build.nvidia.com)

## Quick Start

### 1. Install Poppler (macOS)

```bash
brew install poppler
```

### 2. Clone and Setup

```bash
git clone https://github.com/nileshai/local_ocr_demo.git
cd local_ocr_demo

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key

```bash
# Create .env file
echo "NVIDIA_API_KEY=your-api-key-here" > .env
```

### 4. Run the App

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### 5. Open in Browser

Navigate to `http://localhost:8501`

## Pipeline Stages

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | File Input | Upload PDF or image |
| 2 | Image Conversion | PDF to images via Poppler (150 DPI) |
| 3 | Vision OCR | Llama 3.2 90B Vision extracts text per page |
| 4 | Entity Extraction | Nemotron 30B structures entities with confidence |
| 5 | Results | Markdown table with page source |

## Output Format

```markdown
| Page | Field | Value | Confidence |
|------|-------|-------|------------|
| 1 | Company Name | Emerald Tech Limited | High |
| 1 | Registration No | CRO/2025/IE/004512 | High |
| 2 | Signatory Name | Mark Ryan | High |
| 2 | Signature | (handwritten) | Medium |
```

## Models Used

| Model | Purpose | Provider |
|-------|---------|----------|
| Llama 3.2 90B Vision | OCR / Text Extraction | NVIDIA API |
| Nemotron 3 Nano 30B | Entity Extraction | NVIDIA API |

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `NVIDIA_API_KEY` | API key from build.nvidia.com | Yes |
| `NGC_API_KEY` | Alternative API key name | No |

### Supported File Types

- PDF (multi-page supported)
- PNG
- JPG / JPEG

## Project Structure

```
local_ocr_demo/
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies
├── .env               # API key (create this)
├── .gitignore
└── README.md
```

## License

MIT License

## Acknowledgments

- [NVIDIA Build](https://build.nvidia.com) - API Platform
- [Llama 3.2 Vision](https://ai.meta.com/llama/) - Vision Language Model
- [Streamlit](https://streamlit.io) - Web UI Framework
