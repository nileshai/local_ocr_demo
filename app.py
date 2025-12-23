import json
import os
import io
import base64
import time
from typing import Any, Dict, Tuple, List

import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# NVIDIA API Key - from .env, environment, or Streamlit secrets
def get_api_key():
    # Try environment variables first
    key = os.getenv("NGC_API_KEY", os.getenv("NVIDIA_API_KEY", ""))
    if key:
        return key
    # Try Streamlit secrets (for Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'NVIDIA_API_KEY' in st.secrets:
            return st.secrets['NVIDIA_API_KEY']
    except:
        pass
    return ""

NVIDIA_API_KEY = get_api_key()

# API Endpoints
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
VISION_API_URL = "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct/chat/completions"

# OCR Model Options
OCR_MODEL_OPTIONS = {
    "Llama 3.2 90B Vision (Default)": {
        "name": "Llama 3.2 90B Vision",
        "model": "meta/llama-3.2-90b-vision-instruct",
        "url": VISION_API_URL,
        "type": "vision_llm",
        "description": "High accuracy Vision LLM • Best for forms & handwriting",
    },
    "Llama 3.2 11B Vision (Faster)": {
        "name": "Llama 3.2 11B Vision",
        "model": "meta/llama-3.2-11b-vision-instruct",
        "url": "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-11b-vision-instruct/chat/completions",
        "type": "vision_llm",
        "description": "Faster inference • Good for simple documents",
    },
}

# LLM models for entity extraction - Nemotron 3 Nano 30B is default
MODEL_OPTIONS = {
    "Nemotron 3 Nano 30B (Default)": {
        "url": NVIDIA_API_URL,
        "model": "nvidia/nemotron-3-nano-30b-a3b",
        "description": "30B model • Great balance of speed & quality",
    },
    "Nemotron Super 49B (Accurate)": {
        "url": NVIDIA_API_URL,
        "model": "nvidia/llama-3.3-nemotron-super-49b-v1",
        "description": "High accuracy • Best for complex docs",
    },
    "Llama 3.1 8B (Fast)": {
        "url": NVIDIA_API_URL,
        "model": "meta/llama-3.1-8b-instruct",
        "description": "Fast inference • Quick results",
    },
}

TIMEOUT = int(os.getenv("NIM_TIMEOUT", "180"))

# NVIDIA Brand Colors
NVIDIA_GREEN = "#76B900"
NVIDIA_DARK = "#1A1A1A"
NVIDIA_GRAY = "#2D2D2D"

# Optimized prompts for business documents
VISION_OCR_PROMPT = """Extract ALL text from this document page. Include:
- All form fields and their values
- Checkbox states: [X] if checked, [ ] if unchecked
- Handwritten text (mark as "(handwritten)")
- Signatures (mark as "[SIGNATURE]")
- All names, dates, ID numbers, amounts
- Tables with all rows and columns
- Section headers and labels

Extract everything visible on this page:"""

ENTITY_EXTRACTION_PROMPT = """You are a document data extraction system for business automation. Extract ALL information as entity pairs with confidence scores.

RULES:
1. Extract ONLY what is present - do NOT invent fields
2. Use EXACT field names as they appear in the document
3. Preserve EXACT values (numbers, dates, text, formatting)
4. For checkboxes: "✓ Yes" if checked, "✗ No" if unchecked
5. For handwritten text: Include value with "(handwritten)" note
6. For signatures: Mark as "[Signature Present]"
7. Include PAGE NUMBER for each field
8. Include CONFIDENCE SCORE (0.0 to 1.0) for each extraction

CONFIDENCE SCORING:
- 0.95-1.00: Printed text, clearly legible, standard format
- 0.85-0.94: Clear but with minor formatting variations
- 0.70-0.84: Readable with some ambiguity
- 0.50-0.69: Partially legible, handwritten, or uncertain
- Below 0.50: Low confidence, needs manual review

OUTPUT FORMAT:

| Page | Field | Value | Confidence |
|------|-------|-------|------------|
| 1 | [field name] | [value] | 0.95 |
| 1 | [checkbox] | ✓ Yes | 0.98 |
| 2 | [handwritten field] | [value] (handwritten) | 0.72 |

For tables/line items:
| Page | Field | Value | Confidence |
|------|-------|-------|------------|
| 1 | Item 1 - Description | [value] | 0.97 |
| 1 | Item 1 - Amount | $X.XX | 0.98 |

EXTRACTION SUMMARY:
```
Total Fields: [count]
By Page: Page 1: [n], Page 2: [n]
By Confidence:
  - High (≥0.90): [count] fields
  - Medium (0.70-0.89): [count] fields  
  - Low (<0.70): [count] fields
Average Confidence: [0.XX]
Overall Reliability: [High/Medium/Low]
```

DOCUMENT TEXT:
"""


def inject_custom_css():
    """Inject custom CSS for NVIDIA branding."""
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .stApp {{
            background: linear-gradient(135deg, {NVIDIA_DARK} 0%, #0D0D0D 50%, {NVIDIA_GRAY} 100%);
            font-family: 'Inter', sans-serif;
        }}
        
        .main-header {{
            background: linear-gradient(90deg, {NVIDIA_GREEN}22 0%, transparent 100%);
            border-left: 4px solid {NVIDIA_GREEN};
            padding: 1.5rem 2rem;
            margin-bottom: 2rem;
            border-radius: 0 12px 12px 0;
        }}
        
        .main-title {{ color: white; font-size: 2.2rem; font-weight: 700; margin: 0; }}
        .nvidia-badge {{
            background: {NVIDIA_GREEN};
            color: black;
            font-size: 0.7rem;
            font-weight: 600;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            margin-left: 1rem;
        }}
        
        .main-subtitle {{ color: #E0E0E0; font-size: 1rem; margin-top: 0.5rem; }}
        
        .pipeline-container {{
            background: linear-gradient(135deg, {NVIDIA_GRAY}88 0%, {NVIDIA_DARK}88 100%);
            border: 1px solid #404040;
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1rem 0;
        }}
        
        .pipeline-title {{ color: {NVIDIA_GREEN}; font-size: 1rem; font-weight: 600; margin-bottom: 1rem; }}
        
        .pipeline-flow {{
            display: flex;
            align-items: center;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}
        
        .pipeline-step {{
            background: linear-gradient(135deg, #333 0%, #222 100%);
            border: 1px solid #444;
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
            min-width: 100px;
        }}
        
        .step-icon {{ font-size: 1.5rem; margin-bottom: 0.3rem; }}
        .step-name {{ color: white; font-weight: 600; font-size: 0.8rem; }}
        .step-tech {{ color: {NVIDIA_GREEN}; font-size: 0.7rem; }}
        .pipeline-arrow {{ color: {NVIDIA_GREEN}; font-size: 1.2rem; }}
        
        .status-item {{
            background: #222;
            border-radius: 8px;
            padding: 0.8rem;
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin: 0.3rem 0;
        }}
        
        .status-dot {{ width: 8px; height: 8px; border-radius: 50%; }}
        .status-dot.online {{ background: {NVIDIA_GREEN}; box-shadow: 0 0 6px {NVIDIA_GREEN}; }}
        .status-dot.offline {{ background: #FF4444; }}
        
        .results-header {{
            background: linear-gradient(90deg, {NVIDIA_GREEN} 0%, {NVIDIA_GREEN}88 100%);
            color: black;
            padding: 0.8rem 1.2rem;
            border-radius: 10px 10px 0 0;
            font-weight: 600;
        }}
        
        .results-body {{
            background: #1E1E1E;
            border: 1px solid #333;
            border-top: none;
            border-radius: 0 0 10px 10px;
            padding: 1.2rem;
        }}
        
        .stButton > button {{
            background: linear-gradient(135deg, {NVIDIA_GREEN} 0%, #5A9000 100%);
            color: black;
            font-weight: 600;
            border: none;
            padding: 0.6rem 1.5rem;
            border-radius: 8px;
            width: 100%;
        }}
        
        .stButton > button:hover {{
            background: linear-gradient(135deg, #8BD000 0%, {NVIDIA_GREEN} 100%);
        }}
        
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {NVIDIA_DARK} 0%, #0A0A0A 100%);
            border-right: 1px solid #333;
        }}
        
        .stMarkdown, .stMarkdown p {{ color: #E0E0E0 !important; }}
        h1, h2, h3, h4, h5, h6 {{ color: #FFFFFF !important; }}
        
        .stage-box {{
            background: #1a1a2e;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 4px solid #444;
        }}
        
        .stage-success {{ border-left-color: {NVIDIA_GREEN}; }}
        .stage-error {{ border-left-color: #ff4444; }}
        .stage-warning {{ border-left-color: #ffaa00; }}
        .stage-info {{ border-left-color: #4488ff; }}
        
        .confidence-high {{ color: {NVIDIA_GREEN}; font-weight: bold; }}
        .confidence-medium {{ color: #ffaa00; font-weight: bold; }}
        .confidence-low {{ color: #ff6666; font-weight: bold; }}
        
        #MainMenu, footer, header {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)


def render_header():
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">
            Document Intelligence
            <span class="nvidia-badge">NVIDIA Vision + LLM</span>
        </h1>
        <p class="main-subtitle">
            Multi-Page Vision OCR + Entity Extraction with Confidence Scores
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_pipeline_diagram():
    st.markdown("""
    <div class="pipeline-container">
        <div class="pipeline-title">🔄 Multi-Page Vision OCR Pipeline</div>
        <div class="pipeline-flow">
            <div class="pipeline-step">
                <div class="step-icon">📄</div>
                <div class="step-name">PDF</div>
                <div class="step-tech">All Pages</div>
            </div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step">
                <div class="step-icon">🖼️</div>
                <div class="step-name">Images</div>
                <div class="step-tech">Per Page</div>
            </div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step">
                <div class="step-icon">👁️</div>
                <div class="step-name">Vision OCR</div>
                <div class="step-tech">Llama 3.2 90B</div>
            </div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step">
                <div class="step-icon">🧠</div>
                <div class="step-name">Extract</div>
                <div class="step-tech">Nemotron 30B</div>
            </div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step">
                <div class="step-icon">📊</div>
                <div class="step-name">Entities</div>
                <div class="step-tech">+ Confidence</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_stage(name: str, status: str, message: str, data: Any = None):
    """Display a processing stage with status."""
    icons = {"success": "✅", "error": "❌", "warning": "⚠️", "info": "ℹ️", "processing": "⏳"}
    icon = icons.get(status, "•")
    css_class = f"stage-{status}" if status in ["success", "error", "warning", "info"] else ""
    
    st.markdown(f"""
    <div class="stage-box {css_class}">
        <strong>{icon} {name}</strong><br>
        <span style="color: #aaa; font-size: 0.9rem;">{message}</span>
    </div>
    """, unsafe_allow_html=True)
    
    if data is not None:
        with st.expander(f"📋 {name} - Details"):
            if isinstance(data, (dict, list)):
                st.json(data)
            else:
                st.code(str(data)[:5000])


def pdf_to_images(file_bytes: bytes, dpi: int = 150) -> Tuple[List[Image.Image], Dict]:
    """Convert ALL PDF pages to images using pdf2image."""
    from pdf2image import convert_from_bytes
    
    info = {"method": "pdf2image", "dpi": dpi}
    
    try:
        images = convert_from_bytes(file_bytes, dpi=dpi)
        info["pages"] = len(images)
        info["success"] = True
        return images, info
    except Exception as e:
        info["error"] = str(e)
        info["success"] = False
        return [], info


def image_to_base64(image: Image.Image, max_dim: int = 1024) -> Tuple[str, Dict]:
    """Convert PIL Image to base64, resizing if needed."""
    info = {"original_size": f"{image.width}x{image.height}"}
    
    # Resize if too large
    if image.width > max_dim or image.height > max_dim:
        ratio = min(max_dim / image.width, max_dim / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        info["resized_to"] = f"{image.width}x{image.height}"
    
    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    info["base64_length"] = len(b64)
    return b64, info


def call_vision_ocr(
    image_b64: str, 
    api_key: str, 
    page_num: int = 1, 
    total_pages: int = 1,
    ocr_config: dict = None
) -> Tuple[str, str, Dict]:
    """Call Vision model for OCR on a single page."""
    
    # Default to Llama 3.2 90B Vision if no config provided
    if ocr_config is None:
        ocr_config = OCR_MODEL_OPTIONS["Llama 3.2 90B Vision (Default)"]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    # Simple, direct prompt that works well
    prompt = VISION_OCR_PROMPT
    
    payload = {
        "model": ocr_config["model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                    }
                ]
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.1,
    }
    
    details = {
        "model": ocr_config["model"],
        "model_name": ocr_config["name"],
        "page": page_num,
        "total_pages": total_pages,
    }
    
    try:
        response = requests.post(ocr_config["url"], headers=headers, json=payload, timeout=180)
        details["status_code"] = response.status_code
        
        if response.status_code == 200:
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            details["usage"] = data.get("usage", {})
            details["output_length"] = len(content)
            return content, "success", details
        else:
            details["error"] = response.text[:500]
            return "", f"HTTP {response.status_code}", details
            
    except requests.Timeout:
        details["error"] = "Request timed out (180s)"
        return "", "timeout", details
    except Exception as e:
        details["error"] = str(e)
        return "", "error", details


def process_all_pages(
    images: List[Image.Image], 
    api_key: str, 
    progress_callback=None,
    status_container=None,
    ocr_config: dict = None
) -> Tuple[str, List[Dict]]:
    """Process ALL pages with Vision OCR and combine results."""
    
    all_text = []
    page_details = []
    total_pages = len(images)
    
    for i, img in enumerate(images):
        page_num = i + 1
        
        if progress_callback:
            progress_callback(f"Processing page {page_num}/{total_pages}...")
        
        if status_container:
            status_container.info(f"👁️ OCR processing page {page_num} of {total_pages}...")
        
        # Convert to base64
        image_b64, b64_info = image_to_base64(img, max_dim=1024)
        
        # Call Vision OCR for this page
        text, status, details = call_vision_ocr(image_b64, api_key, page_num, total_pages, ocr_config)
        
        details["page"] = page_num
        details["image_info"] = b64_info
        details["status"] = status
        page_details.append(details)
        
        if status == "success" and text:
            all_text.append(f"\n{'='*60}\n📄 PAGE {page_num} OF {total_pages}\n{'='*60}\n\n{text}")
            if status_container:
                status_container.success(f"✅ Page {page_num}: Extracted {len(text):,} characters")
        else:
            error_msg = details.get("error", status)
            all_text.append(f"\n{'='*60}\n📄 PAGE {page_num} OF {total_pages} - ERROR\n{'='*60}\n\n[OCR Failed: {error_msg}]")
            if status_container:
                status_container.error(f"❌ Page {page_num}: {error_msg}")
        
        # Small delay between API calls to avoid rate limiting
        if page_num < total_pages:
            time.sleep(1)
    
    combined_text = "\n".join(all_text)
    return combined_text, page_details


def call_entity_extraction(
    text: str,
    model_config: dict,
    api_key: str,
    total_pages: int = 1,
) -> Tuple[str, str, Dict]:
    """Call LLM for entity extraction with confidence scores."""
    import re
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Use more text for multi-page documents
    max_chars = min(20000, len(text))
    full_prompt = f"{ENTITY_EXTRACTION_PROMPT}\n{text[:max_chars]}"
    
    # Add reminder about all pages
    if total_pages > 1:
        full_prompt += f"\n\nIMPORTANT: This document has {total_pages} pages. Extract entities from ALL {total_pages} pages. Make sure to include fields from every page."
    
    body = {
        "model": model_config["model"],
        "messages": [
            {
                "role": "system",
                "content": f"You are a document extraction system. Extract ALL fields from ALL {total_pages} pages. Include the page number for each field. Do not stop until you have extracted every field from every page."
            },
            {"role": "user", "content": full_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 8192,
    }
    
    details = {
        "model": model_config["model"],
        "prompt_length": len(full_prompt),
    }
    
    try:
        resp = requests.post(model_config["url"], headers=headers, json=body, timeout=TIMEOUT)
        details["status_code"] = resp.status_code
        
        if resp.status_code != 200:
            details["error"] = resp.text[:500]
            return "", f"HTTP {resp.status_code}", details
        
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Clean thinking tags
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        details["usage"] = data.get("usage", {})
        details["output_length"] = len(content)
        
        return content, "success", details
        
    except requests.Timeout:
        details["error"] = "Request timed out"
        return "", "timeout", details
    except Exception as e:
        details["error"] = str(e)
        return "", "error", details


def process_image_file(file_bytes: bytes, filename: str) -> Tuple[List[Image.Image], Dict]:
    """Load an image file directly."""
    info = {"method": "direct_image", "filename": filename}
    try:
        img = Image.open(io.BytesIO(file_bytes))
        info["size"] = f"{img.width}x{img.height}"
        info["format"] = img.format
        info["pages"] = 1
        info["success"] = True
        return [img], info
    except Exception as e:
        info["error"] = str(e)
        info["success"] = False
        return [], info


def main():
    st.set_page_config(
        page_title="NVIDIA Document Intelligence",
        page_icon="🟢",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    inject_custom_css()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🧠 Entity Extraction Model")
        selected_model = st.selectbox(
            "Choose Model",
            options=list(MODEL_OPTIONS.keys()),
            index=0
        )
        model_config = MODEL_OPTIONS[selected_model]
        st.caption(model_config["description"])
        
        st.markdown("### 👁️ OCR Model")
        selected_ocr = st.selectbox(
            "Choose OCR Model",
            options=list(OCR_MODEL_OPTIONS.keys()),
            index=0,
            key="ocr_model"
        )
        ocr_config = OCR_MODEL_OPTIONS[selected_ocr]
        st.caption(ocr_config["description"])
        
        st.markdown("### 🔑 NVIDIA API Key")
        api_key = st.text_input(
            "API Key",
            value=NVIDIA_API_KEY,
            type="password",
            help="From build.nvidia.com"
        )
        
        if api_key and len(api_key) > 20:
            st.markdown(f"""
            <div class="status-item" style="background: #1a3d1a;">
                <div class="status-dot online"></div>
                <div class="status-value">API Key Configured</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Enter API key")
        
        st.markdown("### ℹ️ Features")
        st.markdown("""
        - ✅ **Multi-page processing**
        - ✅ **Checkbox detection**
        - ✅ **Confidence scores**
        - ✅ **Business document optimized**
        """)
        
        st.markdown("---")
        st.caption("📄 Supports: PDF (all pages), PNG, JPG")
    
    # Main content
    render_header()
    render_pipeline_diagram()
    
    st.markdown("---")
    st.markdown("### 📤 Upload Document")
    
    uploaded = st.file_uploader(
        "Drop your PDF or image file here",
        type=["pdf", "png", "jpg", "jpeg"],
        help="PDF files: ALL pages will be processed"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_button = st.button(
            "🚀 Process Document",
            type="primary",
            use_container_width=True
        )
    
    if uploaded and run_button:
        if not api_key or len(api_key) < 20:
            st.error("❌ Please enter a valid NVIDIA API key in the sidebar.")
            return
        
        st.markdown("---")
        st.markdown("### ⚡ Processing Pipeline")
        
        progress = st.progress(0, text="Starting...")
        status_text = st.empty()
        file_bytes = uploaded.read()
        ext = os.path.splitext(uploaded.name)[1].lower()
        
        # ===== STAGE 1: File Input =====
        st.markdown("#### 📁 Stage 1: Document Input")
        show_stage(
            "File Received",
            "success",
            f"**{uploaded.name}** | {len(file_bytes):,} bytes | Type: {ext}"
        )
        progress.progress(10, text="File received...")
        
        # ===== STAGE 2: Convert to Images (ALL PAGES) =====
        st.markdown("#### 🖼️ Stage 2: Image Conversion (All Pages)")
        progress.progress(15, text="Converting to images...")
        
        if ext == ".pdf":
            images, conv_info = pdf_to_images(file_bytes, dpi=150)
            if images:
                show_stage(
                    "PDF → Images",
                    "success",
                    f"Converted **{len(images)} page(s)** at 150 DPI",
                    conv_info
                )
            else:
                show_stage("PDF Conversion", "error", f"Failed: {conv_info.get('error')}", conv_info)
                st.error("❌ Could not convert PDF. Install poppler: `brew install poppler`")
                return
        else:
            images, conv_info = process_image_file(file_bytes, uploaded.name)
            if images:
                show_stage(
                    "Image Loaded",
                    "success",
                    f"Size: {conv_info.get('size')} | Format: {conv_info.get('format')}",
                    conv_info
                )
            else:
                show_stage("Image Load", "error", f"Failed: {conv_info.get('error')}", conv_info)
                return
        
        total_pages = len(images)
        progress.progress(20, text=f"Processing {total_pages} page(s)...")
        
        # ===== STAGE 3: Vision OCR (ALL PAGES) =====
        st.markdown(f"#### 👁️ Stage 3: OCR with {ocr_config['name']} ({total_pages} pages)")
        
        # Create a container to show per-page status
        page_status_container = st.container()
        
        def update_progress(msg):
            status_text.text(msg)
        
        ocr_text, page_details = process_all_pages(
            images, 
            api_key, 
            progress_callback=update_progress,
            status_container=page_status_container,
            ocr_config=ocr_config
        )
        
        # Calculate stats
        successful_pages = sum(1 for d in page_details if d.get("output_length", 0) > 0)
        total_chars = len(ocr_text)
        
        progress.progress(60, text="OCR complete...")
        status_text.empty()
        
        if successful_pages > 0:
            show_stage(
                "Vision OCR Complete",
                "success",
                f"Processed **{successful_pages}/{total_pages}** pages | "
                f"Total: {total_chars:,} characters",
                {"pages": page_details}
            )
            
            # Show per-page OCR summary
            st.markdown("**📄 Per-Page OCR Results:**")
            for detail in page_details:
                page_num = detail.get("page", "?")
                chars = detail.get("output_length", 0)
                status = detail.get("status", "unknown")
                if status == "success":
                    st.success(f"Page {page_num}: ✅ {chars:,} characters extracted")
                else:
                    st.error(f"Page {page_num}: ❌ {detail.get('error', status)}")
            
            # Show OCR result with page markers visible
            with st.expander("📝 Complete OCR Text (All Pages)", expanded=False):
                st.text_area(
                    "Extracted Text",
                    value=ocr_text,
                    height=400,
                    disabled=True
                )
                st.caption(f"Total: {len(ocr_text):,} characters from {total_pages} pages")
        else:
            show_stage("Vision OCR", "error", "Failed to process any pages", {"pages": page_details})
            return
        
        # ===== STAGE 4: Entity Extraction with Confidence =====
        st.markdown(f"#### 🧠 Stage 4: Entity Extraction with Confidence Scores")
        progress.progress(70, text="Extracting entities...")
        
        # Verify we have content from all pages
        pages_in_text = ocr_text.count("📄 PAGE")
        if pages_in_text < total_pages:
            st.warning(f"⚠️ OCR text contains {pages_in_text} page markers but expected {total_pages}. Some pages may not have been processed.")
        
        show_stage(
            "LLM Analysis",
            "info",
            f"Model: {model_config['model']} | Input: {len(ocr_text):,} chars | "
            f"Pages detected: {pages_in_text} | Extracting ALL entities..."
        )
        
        with st.spinner(f"🧠 Analyzing {total_pages} page(s) with {selected_model}..."):
            entities, entity_status, entity_details = call_entity_extraction(
                ocr_text, model_config, api_key, total_pages
            )
        
        progress.progress(90, text="Entity extraction complete...")
        
        if entity_status == "success" and entities:
            show_stage(
                "Entity Extraction Complete",
                "success",
                f"Output: {len(entities):,} characters | "
                f"Tokens: {entity_details.get('usage', {}).get('total_tokens', 'N/A')}",
                entity_details
            )
        else:
            show_stage("Entity Extraction", "error", f"Status: {entity_status}", entity_details)
        
        # ===== STAGE 5: Results with Confidence =====
        st.markdown("#### 📊 Stage 5: Extracted Entities with Confidence")
        progress.progress(100, text="Complete!")
        
        if entities:
            st.markdown("""<div class="results-header">📊 Extracted Entities with Confidence Scores</div>""", 
                       unsafe_allow_html=True)
            st.markdown("""<div class="results-body">""", unsafe_allow_html=True)
            st.markdown(entities)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.success(f"✅ Document processing complete! ({total_pages} pages processed)")
            
            # Download options
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    "📥 Entities (MD)",
                    entities,
                    file_name=f"{uploaded.name}_entities.md",
                    mime="text/markdown"
                )
            with col2:
                st.download_button(
                    "📥 OCR Text",
                    ocr_text,
                    file_name=f"{uploaded.name}_ocr.txt",
                    mime="text/plain"
                )
            with col3:
                # Create JSON export
                export_data = {
                    "filename": uploaded.name,
                    "pages_processed": total_pages,
                    "ocr_text": ocr_text,
                    "entities_markdown": entities,
                    "page_details": page_details,
                    "model_used": model_config["model"]
                }
                st.download_button(
                    "📥 Full JSON",
                    json.dumps(export_data, indent=2),
                    file_name=f"{uploaded.name}_full.json",
                    mime="application/json"
                )
        else:
            st.error("❌ No entities extracted. Check error details above.")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #666; border-top: 1px solid #333; margin-top: 2rem;">
        Powered by <span style="color: #76B900;">NVIDIA</span> • 
        Multi-Page Vision OCR + Entity Extraction with Confidence Scores
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
