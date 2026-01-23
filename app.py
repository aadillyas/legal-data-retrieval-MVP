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
        "sources": "المراجع المستندة:",
        "footer": "منصة Aadil Illyas القانونية © 2026",
        "view_file": "فتح الملف"
    },
    "en": {
        "title": "⚖️ Legal Discovery Assistant",
        "subtitle": "AI-Powered Cognitive Search System",
        "sync_btn": "Sync Documents",
        "placeholder": "How can I assist with your discovery today?",
        "thinking": "Reviewing legal documents...",
        "sources": "Referenced Sources:",
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
    sys_instr = """You are a strictly Bilingual Legal AI.
    RULE 1: Identify the language of the User Query.
    RULE 2: Respond ONLY in the language of the query.
    RULE 3: Use provided context. Translate accurately if needed.
    RULE 4: Be professional."""
    
    user_prompt = f"Context:\n{context}\n\nUser Query: {query}\n\nRespond in the language of the query."
    payload = {"contents": [{"parts": [{"text": user_prompt}]}], "systemInstruction": {"parts": [{"text": sys_instr}]}}
    try:
        res = requests.post(url, json=payload, timeout=25)
        return res.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception: return "Technical error: Intelligence engine unavailable."

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

    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&family=Noto+Sans+Arabic:wght@400;600;700&display=swap');
        
        .stApp {{
            background-color: #fcfcfd;
            font-family: 'Plus Jakarta Sans', 'Noto Sans Arabic', sans-serif;
            direction: {dir_val};
        }}

        [data-testid="stVerticalBlock"] > div:has(.main-chat-container) {{
            max-width: 850px;
            margin: auto;
        }}

        /* Fix: Show Sidebar Toggle even when header is hidden */
        [data-testid="stSidebarCollapsedControl"] {{
            background-color: #f1f5f9;
            border-radius: 0 8px 8px 0;
            top: 10px;
        }}

        .chat-bubble-assistant {{
            background: white;
            border-radius: 20px;
            padding: 24px;
            margin-bottom: 25px;
            border: 1px solid #eef2f6;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.04);
            animation: fadeIn 0.4s ease-out;
            width: 100%;
        }}
        
        .chat-bubble-user {{
            background: #f1f5f9;
            color: #334155;
            padding: 12px 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            display: inline-block;
            float: { 'right' if rtl else 'left' };
            clear: both;
            max-width: 80%;
            font-weight: 500;
        }}

        .reference-pill {{
            display: inline-flex;
            align-items: center;
            background: #ffffff;
            color: #b5935e;
            padding: 5px 12px;
            border-radius: 8px;
            font-size: 0.85rem;
            margin-{ 'left' if rtl else 'right' }: 8px;
            margin-top: 10px;
            border: 1px solid #b5935e40;
            text-decoration: none;
            font-weight: 600;
        }}

        /* UI Cleanup */
        #MainMenu, footer {{ visibility: hidden; }}
        [data-testid="stHeader"] {{ background: transparent; }}
        
        .stChatInputContainer {{
            border-radius: 30px !important;
            max-width: 800px;
            margin: auto;
        }}
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown(f"### 🛠️ Admin Control")
        if st.button("EN / عربي"):
            st.session_state.lang = "en" if rtl else "ar"
            st.rerun()
            
        service = get_gdrive_service()
        if st.button(t["sync_btn"], use_container_width=True):
            if service:
                with st.spinner("Syncing..."):
                    st.session_state.corpus = download_and_process_drive_docs(service)
                    st.success(f"Indexed {len(st.session_state.corpus)} Pages")
        
        if st.session_state.corpus:
            st.divider()
            for name in set([d['source'] for d in st.session_state.corpus]):
                st.markdown(f"<small>📄 {name}</small>", unsafe_allow_html=True)

    # Chat Area
    st.markdown('<div class="main-chat-container">', unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align: center; color: #0f172a; margin-top: 50px;'>{t['title']}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #64748b; margin-bottom: 50px;'>{t['subtitle']}</p>", unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            refs_html = "".join([f'<a href="{m["link"]}" target="_blank" class="reference-pill">📎 {m["source"]} (P.{m["page"]})</a>' for m in msg.get("metadata", [])])
            st.markdown(f'<div class="chat-bubble-assistant"><div style="text-align:{align};">{msg["content"]}</div><div style="margin-top:15px; border-top:1px solid #f8fafc; padding-top:12px;"><span style="font-size:0.75rem; font-weight:700; color:#b5935e;">{t["sources"]}</span><br>{refs_html}</div></div>', unsafe_allow_html=True)

    query = st.chat_input(t["placeholder"])
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        if st.session_state.corpus:
            with st.spinner(t["thinking"]):
                model = get_embedding_model()
                index = faiss.IndexFlatL2(model.encode(["test"]).shape[1])
                index.add(np.array(model.encode([d['content'] for d in st.session_state.corpus])).astype('float32'))
                matches = [st.session_state.corpus[idx] for idx in index.search(np.array(model.encode([query])).astype('float32'), k=3)[1][0]]
                answer = call_gemini(query, "\n\n".join([f"Source: {m['source']} P.{m['page']}\n{m['content']}" for m in matches]))
                st.session_state.chat_history.append({"role": "assistant", "content": answer, "metadata": matches})
                st.rerun()

    st.markdown(f"<div style='text-align: center; color: #94a3b8; font-size: 0.75rem; margin-top: 100px;'>{t['footer']}</div>", unsafe_allow_html=True)

if __name__ == "__main__": main()
