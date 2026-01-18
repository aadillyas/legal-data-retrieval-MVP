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
    except Exception:
        return None

# --- UI TRANSLATIONS ---
TRANSLATIONS = {
    "ar": {
        "title": "⚖️ مساعد الاكتشاف القانوني",
        "subtitle": "نظام البحث المعرفي المدعوم بالذكاء الاصطناعي",
        "sync_btn": "تحديث المستندات",
        "placeholder": "كيف يمكنني مساعدتك في البحث اليوم؟",
        "thinking": "جاري مراجعة الوثائق القانونية...",
        "sources": "المراجع:",
        "footer": "منصة Aadil Illyas القانونية © 2026",
        "view_file": "فتح الملف"
    },
    "en": {
        "title": "⚖️ Legal Discovery Assistant",
        "subtitle": "AI-Powered Cognitive Search System",
        "sync_btn": "Sync Documents",
        "placeholder": "How can I assist with your discovery today?",
        "thinking": "Reviewing legal documents...",
        "sources": "References:",
        "footer": "Aadil Illyas Legal Platform © 2026",
        "view_file": "Open File"
    }
}

# --- AI CORE LOGIC ---
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def download_and_process_drive_docs(service):
    query = f"'{GDRIVE_FOLDER_ID}' in parents and mimeType='application/pdf' and trashed=false"
    results = service.files().list(q=query, fields="files(id, name, webViewLink)").execute()
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
                extracted_docs.append({
                    "source": item['name'], 
                    "page": i + 1, 
                    "content": text.strip(),
                    "link": item['webViewLink']
                })
    return extracted_docs

def call_gemini(query, context):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent?key={GEMINI_API_KEY}"
    sys_instr = "Professional Bilingual Legal AI. Identify query language and respond in it. Be precise. Cite source and page."
    user_prompt = f"Context:\n{context}\n\nQuery: {query}\n\nRespond in the same language as the query."
    
    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": sys_instr}]}
    }
    try:
        res = requests.post(url, json=payload, timeout=25)
        return res.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception: return "Error connecting to Intelligence engine."

# --- UI DESIGN ---
def main():
    if "lang" not in st.session_state: st.session_state.lang = "ar"
    if "corpus" not in st.session_state: st.session_state.corpus = []
    if "chat_history" not in st.session_state: st.session_state.chat_history = []

    t = TRANSLATIONS[st.session_state.lang]
    rtl = st.session_state.lang == "ar"
    dir_val = "rtl" if rtl else "ltr"
    align = "right" if rtl else "left"

    st.set_page_config(page_title="LKD Pro Assistant", layout="wide")

    # CUSTOM CSS OVERHAUL
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&family=Noto+Sans+Arabic:wght@400;600;700&display=swap');
        
        /* Global Reset */
        .stApp {{
            background-color: #f8fafc;
            font-family: 'Plus Jakarta Sans', 'Noto Sans Arabic', sans-serif;
            direction: {dir_val};
        }}

        /* Centralized Layout */
        [data-testid="stVerticalBlock"] > div:has(.central-wrapper) {{
            max-width: 850px;
            margin: auto;
        }}

        /* Header UI */
        .top-nav {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 0;
            border-bottom: 1px solid #e2e8f0;
            margin-bottom: 2rem;
        }}

        /* Floating Input Container */
        .stChatFloatingInputContainer {{
            background-color: transparent !important;
            padding-bottom: 20px;
        }}

        /* Result Cards */
        .chat-bubble {{
            background: white;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            animation: fadeIn 0.5s ease-out;
            width: 100%;
            display: flex;
            flex-direction: column;
        }}
        
        .chat-text {{
            font-size: 1.1rem;
            line-height: 1.8;
            color: #1e293b;
            text-align: {align};
            white-space: normal;
        }}

        /* References / Pills */
        .reference-pill {{
            display: inline-flex;
            align-items: center;
            background: #f1f5f9;
            color: #475569;
            padding: 6px 14px;
            border-radius: 99px;
            font-size: 0.85rem;
            margin-{ 'left' if rtl else 'right' }: 8px;
            margin-top: 8px;
            border: 1px solid #e2e8f0;
            text-decoration: none;
            transition: all 0.2s;
        }}
        .reference-pill:hover {{
            background: #b5935e20;
            border-color: #b5935e;
            color: #b5935e;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* Hide Sidebar by default for fancy look */
        [data-testid="stSidebar"] {{
            background-color: white;
        }}
        
        /* Fixed spacing issues */
        .stMarkdown div p {{
            margin-bottom: 0px;
        }}
        </style>
    """, unsafe_allow_html=True)

    # Top Navigation
    with st.container():
        col_l, col_r = st.columns([4, 1])
        with col_l:
            st.markdown(f"### {t['title']}")
            st.caption(t["subtitle"])
        with col_r:
            if st.button("English" if rtl else "العربية", use_container_width=True):
                st.session_state.lang = "en" if rtl else "ar"
                st.rerun()

    # Sidebar (Document Management)
    with st.sidebar:
        st.markdown(f"### {t['sync_btn']}")
        service = get_gdrive_service()
        if st.button(t["sync_btn"], use_container_width=True):
            if service:
                with st.spinner("Processing..."):
                    st.session_state.corpus = download_and_process_drive_docs(service)
                    st.success(f"Synced {len(st.session_state.corpus)} pages.")
        
        st.divider()
        if st.session_state.corpus:
            st.markdown("**Active Knowledge Base:**")
            for name in set([d['source'] for d in st.session_state.corpus]):
                st.caption(f"• {name}")

    # Main Chat Area
    chat_container = st.container()

    # Display Chat History
    with chat_container:
        st.markdown('<div class="central-wrapper">', unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""<div style='text-align: { 'left' if rtl else 'right' }; margin-bottom: 10px; opacity: 0.7;'><strong>You:</strong> {msg['content']}</div>""", unsafe_allow_html=True)
            else:
                # Answer Card
                refs_html = "".join([
                    f'<a href="{m["link"]}" target="_blank" class="reference-pill">📄 {m["source"]} (P.{m["page"]})</a>' 
                    for m in msg.get("metadata", [])
                ])
                
                st.markdown(f"""
                    <div class="chat-bubble">
                        <div class="chat-text">{msg['content']}</div>
                        <div style="margin-top: 15px; border-top: 1px solid #f1f5f9; padding-top: 10px;">
                            <span style="font-size: 0.8rem; font-weight: bold; color: #94a3b8;">{t['sources']}</span><br>
                            {refs_html}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Floating Input
    query = st.chat_input(t["placeholder"])

    if query:
        # Add to history
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        if st.session_state.corpus:
            with st.spinner(t["thinking"]):
                # Search
                model = get_embedding_model()
                embeddings = model.encode([d['content'] for d in st.session_state.corpus])
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(np.array(embeddings).astype('float32'))
                
                q_vec = model.encode([query])
                D, I = index.search(np.array(q_vec).astype('float32'), k=3)
                matches = [st.session_state.corpus[idx] for idx in I[0]]
                
                # Context
                ctx_str = "\n\n".join([f"Source: {m['source']} P.{m['page']}\n{m['content']}" for m in matches])
                answer = call_gemini(query, ctx_str)
                
                # Append Response
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer, 
                    "metadata": matches
                })
                st.rerun()
        else:
            st.error("Please sync your Google Drive docs in the sidebar first.")

    # Footer
    st.markdown(f"<div style='text-align: center; color: #94a3b8; font-size: 0.8rem; margin-top: 50px;'>{t['footer']}</div>", unsafe_allow_html=True)

if __name__ == "__main__": main()
