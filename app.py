import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import faiss
import io
import re
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# --- SECRETS & CONFIG ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
GDRIVE_FOLDER_ID = st.secrets.get("GDRIVE_FOLDER_ID", "")
MODEL_ID = "gemini-2.5-flash-preview-09-2025"

# --- GOOGLE DRIVE AUTH ---
@st.cache_resource
def get_gdrive_service():
    try:
        if "gcp_service_account" not in st.secrets:
            return None
        info = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(info)
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        st.error(f"Google Drive Auth Error: {str(e)}")
        return None

# --- TRANSLATIONS ---
TRANSLATIONS = {
    "ar": {
        "title": "⚖️ مساعد الاكتشاف القانوني",
        "subtitle": "البحث المعرفي والتحليل في الوثائق القانونية",
        "badge": "مساعد ثنائي اللغة",
        "sidebar_header": "إعدادات النظام",
        "sync_btn": "🔄 مزامنة من Google Drive",
        "status_ready": "✅ النظام جاهز ومتصل",
        "search_label": "بحث قانوني ذكي",
        "search_placeholder": "اطرح سؤالك القانوني (مثال: شروط إنهاء العقد)",
        "spinner": "جاري التحليل القانوني المستفيض...",
        "answer_header": "🤖 ملخص التقرير القانوني",
        "sources_header": "📚 الأدلة والمستندات المسترجعة",
        "page_label": "الصفحة:",
        "footer": "منصة تجريبية - Aadil Illyas © 2026",
        "no_query": "بانتظار استفسارك لبدء المسح التحليلي...",
        "no_docs": "لا توجد ملفات. يرجى إضافة ملفات PDF للمجلد ومزامنتها."
    },
    "en": {
        "title": "⚖️ Legal Discovery Assistant",
        "subtitle": "Cognitive Search and Analysis for Legal Documents",
        "badge": "Bilingual AI Expert",
        "sidebar_header": "System Settings",
        "sync_btn": "🔄 Sync from Google Drive",
        "status_ready": "✅ System Ready & Connected",
        "search_label": "Smart Legal Search",
        "search_placeholder": "Ask your legal question (e.g., termination clauses)",
        "spinner": "Performing deep legal analysis...",
        "answer_header": "🤖 Legal Summary Report",
        "sources_header": "📚 Retrieved Evidence & Citations",
        "page_label": "Page:",
        "footer": "PoC Platform - Aadil Illyas © 2026",
        "no_query": "Awaiting your query to begin semantic matching...",
        "no_docs": "No documents found. Please add PDFs to Drive and sync."
    }
}

# --- AI CORE LOGIC ---
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def download_and_process_drive_docs(service):
    query = f"'{GDRIVE_FOLDER_ID}' in parents and mimeType='application/pdf' and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get('files', [])
    extracted_docs = []
    for item in items:
        request = service.files().get_media(fileId=item['id'])
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        reader = PdfReader(fh)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and len(text.strip()) > 20:
                extracted_docs.append({"source": item['name'], "page": i + 1, "content": text.strip()})
    return extracted_docs

def call_gemini(prompt, context):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent?key={GEMINI_API_KEY}"
    
    # CRITICAL FIX 1: Explicit instruction to match query language
    sys_instr = """You are a professional Bilingual Legal AI. 
    1. Identify the language of the User's Query.
    2. Respond in the EXACT SAME LANGUAGE as the User's Query (e.g., if asked in English, reply in English).
    3. Use ONLY the provided context for your answer.
    4. Always cite the Source and Page number."""

    payload = {
        "contents": [{"parts": [{"text": f"Context:\n{context}\n\nUser Query: {prompt}"}]}],
        "systemInstruction": {"parts": [{"text": sys_instr}]}
    }
    try:
        res = requests.post(url, json=payload, timeout=25)
        return res.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e: return f"Error: {str(e)}"

# --- UI DESIGN SYSTEM ---
def main():
    if "lang_code" not in st.session_state: st.session_state.lang_code = "ar"
    if "corpus" not in st.session_state: st.session_state.corpus = []
    t = TRANSLATIONS[st.session_state.lang_code]
    direction = "rtl" if st.session_state.lang_code == "ar" else "ltr"
    align = "right" if st.session_state.lang_code == "ar" else "left"

    st.set_page_config(page_title="Legal Discovery Pro (POC)", layout="wide")

    # CRITICAL FIX 2: CSS Overhaul to prevent "one word per line" issue
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Noto+Sans+Arabic:wght@400;600&display=swap');
        
        [data-testid="stAppViewContainer"] {{
            font-family: 'Inter', 'Noto Sans Arabic', sans-serif;
            direction: {direction};
        }}
        
        /* Container forcing full width */
        .legal-card {{
            background-color: white;
            border-radius: 12px;
            padding: 24px;
            border: 1px solid #e2e8f0;
            margin: 1.5rem 0;
            display: block;
            width: 100%;
            min-width: 100%;
            box-sizing: border-box;
            clear: both;
        }}
        
        .legal-answer {{
            font-size: 1.15rem;
            line-height: 1.7;
            color: #1e293b;
            text-align: {align};
            white-space: normal;
            word-wrap: break-word;
            display: block;
            width: 100%;
        }}

        .evidence-card {{
            background-color: #ffffff;
            border-radius: 10px;
            padding: 1.25rem;
            border-{'right' if direction == 'rtl' else 'left'}: 6px solid #b5935e;
            border-top: 1px solid #e2e8f0;
            border-bottom: 1px solid #e2e8f0;
            border-left: 1px solid #e2e8f0;
            border-right: 1px solid #e2e8f0;
            margin-bottom: 20px;
            display: block;
            width: 100%;
            min-width: 100%;
            box-sizing: border-box;
            clear: both;
        }}

        .evidence-content {{
            font-size: 0.95rem;
            color: #475569;
            line-height: 1.6;
            text-align: initial;
            direction: auto;
            unicode-bidi: plaintext;
            white-space: normal;
            word-break: break-word;
            display: block;
            width: 100%;
        }}

        .badge {{
            background-color: #b5935e20;
            color: #b5935e;
            padding: 0.4rem 0.8rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.title(t["sidebar_header"])
        if st.button("English" if st.session_state.lang_code == "ar" else "العربية"):
            st.session_state.lang_code = "en" if st.session_state.lang_code == "ar" else "ar"
            st.rerun()
        st.divider()
        service = get_gdrive_service()
        if service and st.button(t["sync_btn"]):
            with st.spinner("Indexing Docs..."):
                st.session_state.corpus = download_and_process_drive_docs(service)
                st.success(f"Loaded {len(st.session_state.corpus)} pages.")
        if st.session_state.corpus:
            st.success(t["status_ready"])
            for doc_name in set([d['source'] for d in st.session_state.corpus]):
                st.caption(f"📄 {doc_name}")

    st.title(t["title"])
    st.markdown(f"**{t['subtitle']}** <span class='badge'>{t['badge']}</span>", unsafe_allow_html=True)

    query = st.text_input(t["search_label"], placeholder=t["search_placeholder"])

    if query and st.session_state.corpus:
        with st.spinner(t["spinner"]):
            model = get_embedding_model()
            embeddings = model.encode([d['content'] for d in st.session_state.corpus])
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings).astype('float32'))
            q_vec = model.encode([query])
            D, I = index.search(np.array(q_vec).astype('float32'), k=3)
            matches = [st.session_state.corpus[idx] for idx in I[0]]
            ctx_str = "\n\n".join([f"Source: {m['source']} P.{m['page']}\n{m['content']}" for m in matches])
            
            # Pass query to AI
            answer = call_gemini(query, ctx_str)
            
            st.markdown(f"""
                <div class="legal-card">
                    <div style="font-weight: bold; color: #64748b; margin-bottom: 12px; border-bottom: 1px solid #f1f5f9; padding-bottom: 8px;">{t['answer_header']}</div>
                    <div class="legal-answer">{answer}</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.subheader(t["sources_header"])
            for m in matches:
                st.markdown(f"""
                    <div class="evidence-card">
                        <div style="font-weight: bold; font-size: 0.85rem; margin-bottom: 8px; color: #b5935e;">
                            📄 {m['source']} | {t['page_label']} {m['page']}
                        </div>
                        <div class="evidence-content">{m['content']}</div>
                    </div>
                """, unsafe_allow_html=True)
    elif not query:
        st.markdown(f"<div style='height: 150px; display: flex; align-items: center; justify-content: center; border: 2px dashed #e2e8f0; border-radius: 12px; color: #94a3b8; margin-top: 30px;'>{t['no_query'] if st.session_state.corpus else t['no_docs']}</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='text-align: center; border-top: 1px solid #e2e8f0; padding-top: 20px; color: #94a3b8; font-size: 0.85rem; margin-top: 50px;'>{t['footer']}</div>", unsafe_allow_html=True)

if __name__ == "__main__": main()
