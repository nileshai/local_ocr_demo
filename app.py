import json
import os
import time
import uuid
from typing import Any, Dict

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# NV Ingest endpoint
INGEST_URL = os.getenv("INGEST_URL", "http://localhost:8080/v1/convert")
STATUS_URL = os.getenv("STATUS_URL", "http://localhost:8080/v1/status")

# LLM endpoints
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:8000/v1/chat/completions")
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_API_KEY = os.getenv("NGC_API_KEY", "")

# Model options
MODEL_OPTIONS = {
    "Local: Nemotron Nano 9B (Fast)": {
        "url": LOCAL_LLM_URL,
        "model": "nvidia/nvidia-nemotron-nano-9b-v2",
        "is_local": True,
        "description": "Locally deployed on A100 GPU • Low latency • Best for real-time applications",
    },
    "API: Nemotron 3 Nano 30B (Accurate)": {
        "url": NVIDIA_API_URL,
        "model": "nvidia/nemotron-3-nano-30b-a3b",
        "is_local": False,
        "description": "NVIDIA Cloud API • Higher accuracy • Best for complex documents",
    },
}

TIMEOUT = int(os.getenv("NIM_TIMEOUT", "300"))

# NVIDIA Brand Colors
NVIDIA_GREEN = "#76B900"
NVIDIA_DARK = "#1A1A1A"
NVIDIA_GRAY = "#2D2D2D"


def inject_custom_css():
    """Inject custom CSS for NVIDIA branding."""
    st.markdown(f"""
    <style>
        /* Import fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global styles */
        .stApp {{
            background: linear-gradient(135deg, {NVIDIA_DARK} 0%, #0D0D0D 50%, {NVIDIA_GRAY} 100%);
            font-family: 'Inter', sans-serif;
        }}
        
        /* Header styling */
        .main-header {{
            background: linear-gradient(90deg, {NVIDIA_GREEN}22 0%, transparent 100%);
            border-left: 4px solid {NVIDIA_GREEN};
            padding: 1.5rem 2rem;
            margin-bottom: 2rem;
            border-radius: 0 12px 12px 0;
        }}
        
        .main-title {{
            color: white;
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        
        .nvidia-badge {{
            background: {NVIDIA_GREEN};
            color: black;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .main-subtitle {{
            color: #E0E0E0;
            font-size: 1.1rem;
            margin-top: 0.5rem;
            font-weight: 300;
        }}
        
        /* Pipeline visualization */
        .pipeline-container {{
            background: linear-gradient(135deg, {NVIDIA_GRAY}88 0%, {NVIDIA_DARK}88 100%);
            border: 1px solid #404040;
            border-radius: 16px;
            padding: 2rem;
            margin: 1.5rem 0;
        }}
        
        .pipeline-title {{
            color: {NVIDIA_GREEN};
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .pipeline-flow {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 1rem;
        }}
        
        .pipeline-step {{
            background: linear-gradient(135deg, #333 0%, #222 100%);
            border: 1px solid #444;
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            flex: 1;
            min-width: 140px;
            transition: all 0.3s ease;
        }}
        
        .pipeline-step:hover {{
            border-color: {NVIDIA_GREEN};
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(118, 185, 0, 0.15);
        }}
        
        .step-icon {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }}
        
        .step-name {{
            color: white;
            font-weight: 600;
            font-size: 0.9rem;
            margin-bottom: 0.25rem;
        }}
        
        .step-tech {{
            color: {NVIDIA_GREEN};
            font-size: 0.75rem;
            font-weight: 500;
        }}
        
        .pipeline-arrow {{
            color: {NVIDIA_GREEN};
            font-size: 1.5rem;
            font-weight: bold;
        }}
        
        /* Card styling */
        .info-card {{
            background: linear-gradient(135deg, {NVIDIA_GRAY}66 0%, {NVIDIA_DARK}66 100%);
            border: 1px solid #404040;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
        }}
        
        .card-title {{
            color: {NVIDIA_GREEN};
            font-size: 0.9rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.75rem;
        }}
        
        .card-content {{
            color: #E0E0E0;
            font-size: 0.95rem;
            line-height: 1.6;
        }}
        
        /* Status indicators */
        .status-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        
        .status-item {{
            background: #222;
            border-radius: 8px;
            padding: 1rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}
        
        .status-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }}
        
        .status-dot.online {{
            background: {NVIDIA_GREEN};
            box-shadow: 0 0 8px {NVIDIA_GREEN};
        }}
        
        .status-dot.offline {{
            background: #FF4444;
            box-shadow: 0 0 8px #FF4444;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        
        .status-label {{
            color: #E0E0E0;
            font-size: 0.85rem;
        }}
        
        .status-value {{
            color: white;
            font-weight: 500;
        }}
        
        /* Results section */
        .results-header {{
            background: linear-gradient(90deg, {NVIDIA_GREEN} 0%, {NVIDIA_GREEN}88 100%);
            color: black;
            padding: 1rem 1.5rem;
            border-radius: 12px 12px 0 0;
            font-weight: 600;
            font-size: 1.1rem;
        }}
        
        .results-body {{
            background: #1E1E1E;
            border: 1px solid #333;
            border-top: none;
            border-radius: 0 0 12px 12px;
            padding: 1.5rem;
        }}
        
        /* Streamlit overrides */
        .stButton > button {{
            background: linear-gradient(135deg, {NVIDIA_GREEN} 0%, #5A9000 100%);
            color: black;
            font-weight: 600;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 8px;
            font-size: 1rem;
            transition: all 0.3s ease;
            width: 100%;
        }}
        
        .stButton > button:hover {{
            background: linear-gradient(135deg, #8BD000 0%, {NVIDIA_GREEN} 100%);
            box-shadow: 0 4px 16px rgba(118, 185, 0, 0.4);
            transform: translateY(-1px);
        }}
        
        .stSelectbox label, .stTextArea label, .stTextInput label {{
            color: #E0E0E0 !important;
            font-weight: 500;
        }}
        
        .stSelectbox > div > div {{
            background: #2A2A2A;
            border-color: #444;
            color: #FFFFFF !important;
        }}
        
        .stSelectbox > div > div > div {{
            color: #FFFFFF !important;
        }}
        
        div[data-baseweb="select"] span {{
            color: #FFFFFF !important;
        }}
        
        .stTextArea textarea {{
            background: #2A2A2A;
            border-color: #444;
            color: white;
        }}
        
        .stExpander {{
            background: #222;
            border: 1px solid #333;
            border-radius: 8px;
        }}
        
        div[data-testid="stExpander"] details summary p {{
            color: #E0E0E0;
        }}
        
        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {NVIDIA_DARK} 0%, #0A0A0A 100%);
            border-right: 1px solid #333;
        }}
        
        section[data-testid="stSidebar"] .stMarkdown h3 {{
            color: {NVIDIA_GREEN};
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        section[data-testid="stSidebar"] .stMarkdown p {{
            color: #E0E0E0 !important;
        }}
        
        section[data-testid="stSidebar"] .stCaption {{
            color: #CCCCCC !important;
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding: 2rem;
            color: #999;
            font-size: 0.85rem;
            border-top: 1px solid #333;
            margin-top: 3rem;
        }}
        
        .footer a {{
            color: {NVIDIA_GREEN};
            text-decoration: none;
        }}
        
        /* Progress steps */
        .step-indicator {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.75rem 1rem;
            background: #222;
            border-radius: 8px;
            margin: 0.5rem 0;
        }}
        
        .step-number {{
            background: {NVIDIA_GREEN};
            color: black;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 0.8rem;
        }}
        
        .step-text {{
            color: white;
            font-weight: 500;
        }}
        
        /* Fix all text visibility */
        .stMarkdown, .stMarkdown p, .stMarkdown li {{
            color: #E0E0E0 !important;
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            color: #FFFFFF !important;
        }}
        
        .stCaption, span[data-testid="stCaption"] {{
            color: #CCCCCC !important;
        }}
        
        /* Info text */
        .stAlert p {{
            color: #E0E0E0 !important;
        }}
        
        /* Hide default elements */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render the main header with NVIDIA branding."""
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="#76B900">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
            </svg>
            Intelligent Document Processing
            <span class="nvidia-badge">Powered by NVIDIA</span>
        </h1>
        <p class="main-subtitle">
            Enterprise-grade OCR and Entity Extraction using NVIDIA's AI Infrastructure
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_pipeline_diagram():
    """Render the pipeline visualization."""
    st.markdown("""
    <div class="pipeline-container">
        <div class="pipeline-title">🔄 Processing Pipeline Architecture</div>
        <div class="pipeline-flow">
            <div class="pipeline-step">
                <div class="step-icon">📄</div>
                <div class="step-name">Document Input</div>
                <div class="step-tech">PDF / Image</div>
            </div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step">
                <div class="step-icon">🔍</div>
                <div class="step-name">NV-Ingest</div>
                <div class="step-tech">OCR & Extraction</div>
            </div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step">
                <div class="step-icon">🧠</div>
                <div class="step-name">Nemotron LLM</div>
                <div class="step-tech">Entity Analysis</div>
            </div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step">
                <div class="step-icon">📊</div>
                <div class="step-name">Structured Output</div>
                <div class="step-tech">Key-Value Pairs</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_info_section():
    """Render information about the pipeline."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="card-title">🚀 NV-Ingest OCR Engine</div>
            <div class="card-content">
                NVIDIA's high-performance document processing microservice extracts text, 
                tables, and structured content from PDFs and images using GPU-accelerated 
                computer vision models. Optimized for enterprise throughput.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="card-title">🤖 Nemotron Language Model</div>
            <div class="card-content">
                NVIDIA's Nemotron LLMs provide state-of-the-art natural language 
                understanding for entity extraction, document comprehension, and structured 
                data generation. Deployed as NVIDIA NIM for optimized inference.
            </div>
        </div>
        """, unsafe_allow_html=True)


def check_service_status():
    """Check and return service status."""
    services = {}
    try:
        r = requests.get("http://localhost:8080/v1/health/ready", timeout=3)
        services["nv_ingest"] = r.status_code == 200
    except:
        services["nv_ingest"] = False
    
    try:
        r = requests.get("http://localhost:8000/v1/health/ready", timeout=3)
        services["local_llm"] = r.status_code == 200
    except:
        services["local_llm"] = False
    
    return services


def render_status_indicators(services: dict):
    """Render service status indicators."""
    nv_status = "online" if services.get("nv_ingest") else "offline"
    llm_status = "online" if services.get("local_llm") else "offline"
    
    st.markdown(f"""
    <div class="status-grid">
        <div class="status-item">
            <div class="status-dot {nv_status}"></div>
            <div>
                <div class="status-label">NV-Ingest OCR</div>
                <div class="status-value">{'Ready' if services.get('nv_ingest') else 'Offline'}</div>
            </div>
        </div>
        <div class="status-item">
            <div class="status-dot {llm_status}"></div>
            <div>
                <div class="status-label">Nemotron LLM</div>
                <div class="status-value">{'Ready' if services.get('local_llm') else 'Offline'}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def call_ingest(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Use NV Ingest /v1/convert endpoint to process document."""
    job_id = str(uuid.uuid4())
    
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        mime_type = "application/pdf"
    elif ext in [".png", ".jpg", ".jpeg"]:
        mime_type = f"image/{ext.lstrip('.')}"
    else:
        mime_type = "application/pdf"
    
    files = [("files", (filename, file_bytes, mime_type))]
    data = {
        "job_id": job_id,
        "extract_text": "true",
        "extract_images": "true",
        "extract_tables": "true",
    }
    
    resp = requests.post(INGEST_URL, files=files, data=data, timeout=TIMEOUT)
    resp.raise_for_status()
    result = resp.json()
    
    if result.get("status") == "processing":
        task_id = result.get("task_id", job_id)
        return poll_for_result(task_id)
    
    return result


def poll_for_result(task_id: str, max_wait: int = 120) -> Dict[str, Any]:
    """Poll for job completion."""
    start = time.time()
    while time.time() - start < max_wait:
        resp = requests.get(f"{STATUS_URL}/{task_id}", timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "completed":
                return data
        time.sleep(2)
    raise TimeoutError(f"Job {task_id} did not complete in {max_wait}s")


def call_llm(prompt: str, model_config: dict, api_key: str) -> str:
    """Call LLM for inference (local or API)."""
    import re
    
    headers = {"Content-Type": "application/json"}
    
    if not model_config["is_local"] and api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    body = {
        "model": model_config["model"],
        "messages": [
            {"role": "system", "content": "You are an assistant that extracts and formats key-value entities from document text. Be concise and structured. Output only the final result."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 4096,
    }
    
    resp = requests.post(model_config["url"], headers=headers, json=body, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    
    return content


def extract_text_from_result(result: Any) -> str:
    """Extract text content from nv-ingest result."""
    if isinstance(result, dict):
        if "result" in result and isinstance(result["result"], list):
            texts = []
            for item in result["result"]:
                if isinstance(item, dict) and "content" in item:
                    texts.append(item["content"])
            if texts:
                return "\n".join(texts)
        
        for key in ["text", "content", "extracted_text", "data"]:
            if key in result:
                val = result[key]
                if isinstance(val, str):
                    return val
                return extract_text_from_result(val)
        
        texts = []
        for v in result.values():
            t = extract_text_from_result(v)
            if t:
                texts.append(t)
        return "\n".join(texts)
    elif isinstance(result, list):
        texts = []
        for item in result:
            t = extract_text_from_result(item)
            if t:
                texts.append(t)
        return "\n".join(texts)
    elif isinstance(result, str):
        return result
    return ""


def main() -> None:
    st.set_page_config(
        page_title="NVIDIA Document Intelligence",
        page_icon="🟢",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    inject_custom_css()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <svg width="120" height="30" viewBox="0 0 120 30">
                <text x="0" y="22" fill="#76B900" font-family="Inter, sans-serif" font-weight="700" font-size="18">NVIDIA</text>
            </svg>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🤖 Model Selection")
        selected_model = st.selectbox(
            "Choose LLM",
            options=list(MODEL_OPTIONS.keys()),
            index=0,
            help="Select the language model for entity extraction"
        )
        model_config = MODEL_OPTIONS[selected_model]
        
        st.caption(model_config["description"])
        
        api_key = ""
        if not model_config["is_local"]:
            st.markdown("### 🔑 API Configuration")
            api_key = st.text_input("NVIDIA API Key", value=NVIDIA_API_KEY, type="password")
            if not api_key:
                st.warning("⚠️ API key required")
        
        st.markdown("### 📝 Extraction Prompt")
        custom_prompt = st.text_area(
            "Customize the extraction prompt",
            value="Extract all key-value entities from the following document. Format as a clean markdown table with columns: Field, Value. Include all relevant information such as names, dates, amounts, addresses, and identifiers.",
            height=120,
            label_visibility="collapsed"
        )
        
        st.markdown("### 📡 System Status")
        services = check_service_status()
        render_status_indicators(services)
    
    # Main content
    render_header()
    render_pipeline_diagram()
    render_info_section()
    
    # Upload section
    st.markdown("---")
    st.markdown("### 📤 Upload Document")
    
    uploaded = st.file_uploader(
        "Drop your PDF or image file here",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Supported formats: PDF, PNG, JPG, JPEG"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_button = st.button("🚀 Process Document", type="primary", use_container_width=True)
    
    if uploaded and run_button:
        services = check_service_status()
        
        if not services.get("nv_ingest"):
            st.error("❌ NV-Ingest service is offline. Please ensure the service is running.")
            return
        
        if not model_config["is_local"] and not api_key:
            st.error("❌ API key required for cloud model. Please enter your NVIDIA API key.")
            return
        
        if model_config["is_local"] and not services.get("local_llm"):
            st.error("❌ Local LLM service is offline. Please ensure the service is running or switch to API model.")
            return
        
        # Processing
        st.markdown("---")
        st.markdown("### ⚡ Processing Results")
        
        progress_container = st.container()
        
        with progress_container:
            # Step 1: OCR
            st.markdown("""
            <div class="step-indicator">
                <div class="step-number">1</div>
                <div class="step-text">Extracting text with NV-Ingest OCR...</div>
            </div>
            """, unsafe_allow_html=True)
            
            try:
                with st.spinner(""):
                    ingest_result = call_ingest(uploaded.read(), uploaded.name)
                st.success("✅ Document processed successfully!")
                
                with st.expander("📋 View Raw OCR Output"):
                    st.json(ingest_result)
                
                extracted_text = extract_text_from_result(ingest_result)
                if not extracted_text.strip():
                    extracted_text = json.dumps(ingest_result, indent=2)
                
                # Step 2: Show extracted text
                st.markdown("""
                <div class="step-indicator">
                    <div class="step-number">2</div>
                    <div class="step-text">Extracted Document Content</div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📄 View Extracted Text", expanded=False):
                    st.text_area("", value=extracted_text, height=200, disabled=True, label_visibility="collapsed")
                
                # Step 3: LLM Analysis
                st.markdown(f"""
                <div class="step-indicator">
                    <div class="step-number">3</div>
                    <div class="step-text">Analyzing with {model_config['model'].split('/')[-1]}...</div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.spinner(""):
                    full_prompt = f"{custom_prompt}\n\nDocument content:\n{extracted_text[:8000]}"
                    llm_response = call_llm(full_prompt, model_config, api_key)
                
                st.success("✅ Entity extraction complete!")
                
                # Results display
                st.markdown("""
                <div class="results-header">
                    📊 Extracted Entities
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="results-body">
                """, unsafe_allow_html=True)
                
                st.markdown(llm_response)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            except requests.HTTPError as e:
                st.error(f"❌ HTTP Error: {e.response.status_code}")
                with st.expander("Error Details"):
                    st.code(e.response.text[:1000])
            except TimeoutError as e:
                st.error(f"❌ Timeout: {e}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Built with <span style="color: #76B900;">NVIDIA</span> • 
        <a href="https://build.nvidia.com" target="_blank">NVIDIA AI Platform</a> • 
        <a href="https://github.com/NVIDIA/nv-ingest" target="_blank">NV-Ingest</a></p>
        <p style="font-size: 0.75rem; margin-top: 0.5rem;">
            Enterprise Document Intelligence Solution
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
