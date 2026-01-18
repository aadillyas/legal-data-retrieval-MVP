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
    """Authenticates using Service Account credentials from Streamlit Secrets."""
    try:
        if "gcp_service_account" not in st.secrets:
            st.error("Google Cloud secrets not found in Streamlit settings.")
            return None
        
        info = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(info)
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        st.error(f"Google Drive Auth Error: {str(e)}")
        return None

# --- UI TRANSLATIONS ---
TRANSLATIONS = {
    "ar": {
        "title": "⚖️ مساعد الاكتشاف القانوني",
        "subtitle": "بحث ذكي في ملفات Google Drive",
        "sync_btn": "🔄 مزامنة المستندات",
        "search_label": "اطرح سؤالك القانوني هنا...",
        "answer_header": "🤖 الخلاصة القانونية",
        "sources_header": "📚 المصادر المستخرجة",
        "thinking": "جاري تحليل الملفات...",
        "no_docs": "لا توجد ملفات. يرجى إضافة ملفات PDF للمجلد ومزامنتها."
    },
    "en": {
        "title": "⚖️ Legal Discovery Assistant",
        "subtitle": "Smart search through Google Drive docs",
        "sync_btn": "🔄 Sync Documents",
        "search_label": "Ask your legal question...",
        "answer_header": "🤖 Legal Summary",
        "sources_header": "📚 Referenced Sources",
        "thinking": "Analyzing documents...",
        "no_docs": "No documents found. Please add PDFs to Drive and sync."
    }
}

# --- CORE AI LOGIC ---

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def download_and_process_drive_docs(service):
    """Downloads all PDFs from the target folder and extracts text."""
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
        
        # Process PDF from memory
        reader = PdfReader(fh)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and len(text.strip()) > 20:
                extracted_docs.append({
                    "source": item['name'],
                    "page": i + 1,
                    "content": text.strip()
                })
    return extracted_docs

def call_gemini(prompt, context, lang):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent?key={GEMINI_API_KEY}"
    sys_instr = "You are a professional Legal AI. Answer based ONLY on provided context. Cite source and page."
    if lang == "ar":
        sys_instr = "أنت خبير قانوني. أجب فقط باستخدام النصوص المقدمة وذكر اسم الملف ورقم الصفحة."

    payload = {
        "contents": [{"parts": [{"text": f"Context:\n{context}\n\nQuery: {prompt}"}]}],
        "systemInstruction": {"parts": [{"text": sys_instr}]}
    }
    try:
        res = requests.post(url, json=payload, timeout=25)
        return res.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Error: {str(e)}"

# --- MAIN UI ---

def main():
    if "lang" not in st.session_state: st.session_state.lang = "ar"
    if "corpus" not in st.session_state: st.session_state.corpus = []
    
    t = TRANSLATIONS[st.session_state.lang]
    st.set_page_config(page_title="Legal AI Drive MVP", layout="wide")

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        if st.button("English" if st.session_state.lang == "ar" else "العربية"):
            st.session_state.lang = "en" if st.session_state.lang == "ar" else "ar"
            st.rerun()
        
        st.divider()
        service = get_gdrive_service()
        if service and st.button(t["sync_btn"]):
            with st.spinner("Downloading files from Drive..."):
                st.session_state.corpus = download_and_process_drive_docs(service)
                st.success(f"Synced {len(st.session_state.corpus)} pages.")

    # Application Body
    st.title(t["title"])
    st.markdown(f"#### {t['subtitle']}")

    query = st.text_input(t["search_label"])

    if query and st.session_state.corpus:
        with st.spinner(t["thinking"]):
            # Vector Search (RAG)
            model = load_embedding_model()
            embeddings = model.encode([d['content'] for d in st.session_state.corpus])
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings).astype('float32'))
            
            q_vec = model.encode([query])
            D, I = index.search(np.array(q_vec).astype('float32'), k=3)
            matches = [st.session_state.corpus[idx] for idx in I[0]]
            
            # AI Generation
            ctx_str = "\n\n".join([f"Source: {m['source']} P.{m['page']}\n{m['content']}" for m in matches])
            answer = call_gemini(query, ctx_str, st.session_state.lang)
            
            # Display
            st.subheader(t["answer_header"])
            st.info(answer)
            
            st.subheader(t["sources_header"])
            for m in matches:
                with st.expander(f"📄 {m['source']} (Page {m['page']})"):
                    st.write(m['content'])
    elif not st.session_state.corpus:
        st.info(t["no_docs"])

if __name__ == "__main__":
    main()
