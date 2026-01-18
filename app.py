import streamlit as st
import pandas as pd
import numpy as np
import os
import boto3
import requests
import faiss
import io
import re
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# --- SECRETS & CLOUD CONFIG ---
# These must be set in your Streamlit Cloud "Secrets" or your local environment
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
AWS_KEY = st.secrets.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET = st.secrets.get("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = st.secrets.get("AWS_REGION", "us-east-1")
BUCKET_NAME = st.secrets.get("S3_BUCKET_NAME", "")
MODEL_ID = "gemini-2.5-flash-preview-09-2025"

# --- AWS CLIENT INITIALIZATION ---
try:
    s3_client = boto3.client(
        's3', 
        aws_access_key_id=AWS_KEY, 
        aws_secret_access_key=AWS_SECRET, 
        region_name=AWS_REGION
    )
    textract_client = boto3.client(
        'textract', 
        aws_access_key_id=AWS_KEY, 
        aws_secret_access_key=AWS_SECRET, 
        region_name=AWS_REGION
    )
    AWS_CONNECTED = True if BUCKET_NAME else False
except Exception:
    AWS_CONNECTED = False

# --- UI TRANSLATIONS ---
TRANSLATIONS = {
    "ar": {
        "title": "⚖️ مساعد الاكتشاف القانوني",
        "subtitle": "البحث المعرفي والتحليل في الوثائق القانونية",
        "search_placeholder": "ما الذي تبحث عنه؟ (مثال: شروط إنهاء العقد)",
        "sidebar_header": "📁 إدارة الملفات",
        "upload_label": "تحميل وثائق قانونية (PDF)",
        "status_connected": "📦 متصل بـ AWS S3",
        "status_local": "⚠️ وضع المعاينة المحلية",
        "answer_header": "🤖 الخلاصة القانونية الذكية",
        "sources_header": "📚 المصادر والأدلة",
        "page_label": "صفحة",
        "thinking": "جاري التحليل واستخراج الإجابة...",
        "no_docs": "يرجى تحميل وثائق لبدء التحليل."
    },
    "en": {
        "title": "⚖️ Legal Knowledge Assistant",
        "subtitle": "Cognitive Search and Analysis for Legal Documents",
        "search_placeholder": "What are you looking for? (e.g., Termination clauses)",
        "sidebar_header": "📁 Document Management",
        "upload_label": "Upload Legal Documents (PDF)",
        "status_connected": "📦 Connected to AWS S3",
        "status_local": "⚠️ Local Preview Mode",
        "answer_header": "🤖 Smart Legal Summary",
        "sources_header": "📚 Evidence & Sources",
        "page_label": "Page",
        "thinking": "Analyzing documents and generating response...",
        "no_docs": "Please upload documents to begin analysis."
    }
}

# --- STYLING & RESPONSIVE DESIGN ---
def apply_custom_css(lang):
    direction = "rtl" if lang == "ar" else "ltr"
    align = "right" if lang == "ar" else "left"
    
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@400;700&display=swap');
        
        html, body, [class*="css"] {{
            font-family: 'Noto Sans Arabic', sans-serif;
            direction: {direction};
            text-align: {align};
        }}

        .main-container {{ max-width: 900px; margin: auto; }}
        
        /* Legal Cards */
        .legal-card {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
        }}
        
        .legal-answer {{
            line-height: 1.8;
            font-size: 1.1rem;
            color: #1e293b;
        }}

        .evidence-card {{
            border-{align}: 5px solid #b5935e;
            padding: 15px;
            background: #f8fafc;
            border-radius: 8px;
            margin-bottom: 12px;
        }}

        .evidence-content {{
            font-size: 0.95rem;
            color: #475569;
            font-style: italic;
        }}

        /* Mobile Adjustments */
        @media (max-width: 768px) {{
            .stApp {{ padding: 10px; }}
            .legal-card {{ padding: 15px; }}
        }}
        </style>
    """, unsafe_allow_html=True)

# --- CORE LOGIC: OCR, STORAGE, SEARCH ---

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def get_textract_text(file_bytes):
    """Fallback OCR using AWS Textract for scanned documents."""
    if not AWS_CONNECTED: return ""
    try:
        response = textract_client.detect_document_text(Document={'Bytes': file_bytes})
        return " ".join([b['Text'] for b in response['Blocks'] if b['BlockType'] == 'LINE'])
    except Exception:
        return ""

def process_pdf(uploaded_file):
    """Extracts text, uses Textract if page is scanned, and prepares metadata."""
    file_bytes = uploaded_file.read()
    
    # Optional: Upload to S3 if connected
    if AWS_CONNECTED:
        s3_client.put_object(Bucket=BUCKET_NAME, Key=f"vault/{uploaded_file.name}", Body=file_bytes)

    reader = PdfReader(io.BytesIO(file_bytes))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # Heuristic: If text is very short, it's likely a scan
        if len(text.strip()) < 100 and AWS_CONNECTED:
            # Note: In production MVP, we'd send only this specific page to Textract
            # For this PoC, we send the whole file if a scan is detected
            text = get_textract_text(file_bytes)
            
        if text.strip():
            docs.append({
                "source": uploaded_file.name,
                "page": i + 1,
                "content": text.strip()
            })
    return docs

def call_gemini(prompt, context, lang):
    """Calls Gemini with the RAG context and system instructions."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent?key={GEMINI_API_KEY}"
    
    sys_instruction = "You are a bilingual Legal AI. Answer strictly using context. Cite source and page."
    if lang == "ar":
        sys_instruction = "أنت خبير قانوني. أجب فقط باستخدام المراجع المقدمة وذكر اسم الملف ورقم الصفحة."

    payload = {
        "contents": [{"parts": [{"text": f"Context:\n{context}\n\nQuery: {prompt}"}]}],
        "systemInstruction": {"parts": [{"text": sys_instruction}]}
    }
    
    try:
        response = requests.post(url, json=payload, timeout=25)
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception:
        return "⚠️ Error connecting to AI Service. Please check your API key."

# --- APP EXECUTION ---

def main():
    if "lang" not in st.session_state: st.session_state.lang = "ar"
    if "corpus" not in st.session_state: st.session_state.corpus = []
    
    t = TRANSLATIONS[st.session_state.lang]
    apply_custom_css(st.session_state.lang)

    # Sidebar
    with st.sidebar:
        st.markdown(f"### {t['sidebar_header']}")
        st.info(t["status_connected"] if AWS_CONNECTED else t["status_local"])
        
        # Language Toggle
        if st.button("English" if st.session_state.lang == "ar" else "العربية"):
            st.session_state.lang = "en" if st.session_state.lang == "ar" else "ar"
            st.rerun()
            
        st.divider()
        files = st.file_uploader(t["upload_label"], type="pdf", accept_multiple_files=True)
        if st.button("Ingest Documents"):
            if files:
                with st.spinner("Indexing..."):
                    new_docs = []
                    for f in files:
                        new_docs.extend(process_pdf(f))
                    st.session_state.corpus = new_docs
                    st.success(f"Loaded {len(new_docs)} pages.")

    # Main Area
    st.title(t["title"])
    st.markdown(f"#### {t['subtitle']}")

    query = st.text_input("", placeholder=t["search_placeholder"])

    if query and st.session_state.corpus:
        with st.spinner(t["thinking"]):
            # 1. Load Model & Create Index
            model = load_embedding_model()
            content_list = [d['content'] for d in st.session_state.corpus]
            embeddings = model.encode(content_list)
            
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings).astype('float32'))
            
            # 2. Search
            q_vec = model.encode([query])
            D, I = index.search(np.array(q_vec).astype('float32'), k=3)
            matches = [st.session_state.corpus[idx] for idx in I[0]]
            
            # 3. Generate Answer
            ctx_str = "\n\n".join([f"Source: {m['source']} (P.{m['page']})\n{m['content']}" for m in matches])
            answer = call_gemini(query, ctx_str, st.session_state.lang)
            
            # 4. Render UI
            st.markdown(f"""
                <div class="legal-card">
                    <div style="font-weight: bold; color: #64748b; margin-bottom: 10px;">{t['answer_header']}</div>
                    <div class="legal-answer">{answer}</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"### {t['sources_header']}")
            for m in matches:
                st.markdown(f"""
                    <div class="evidence-card">
                        <div style="font-weight: bold; font-size: 0.85rem; margin-bottom: 5px;">
                            📄 {m['source']} | {t['page_label']} {m['page']}
                        </div>
                        <div class="evidence-content">{m['content'][:400]}...</div>
                    </div>
                """, unsafe_allow_html=True)
    elif not query:
        st.markdown(f"<div style='text-align: center; margin-top: 50px; color: #94a3b8;'>{t['no_docs'] if not st.session_state.corpus else ''}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
