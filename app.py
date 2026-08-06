"""
app.py
------
Basit Streamlit arayüzü.
Çalıştırmak için: streamlit run app.py
"""

import os
import sys
import tempfile
import streamlit as st
from dotenv import load_dotenv
from rag import RagIndex, read_docx, answer_question

# set_page_config İLK Streamlit komutu olmalı ve YALNIZCA BİR KEZ çağrılır
st.set_page_config(page_title="Ders Notu RAG Asistanı", page_icon="📚")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

st.title("📚 Ders Notu RAG Asistanı")
st.write(
    "Bir Word (.docx) ders notu belgesi yükle, sonra notlarla ilgili "
    "soru sor. Cevaplar yalnızca yüklediğin belgeye dayanarak üretilir."
)

env_api_key = os.environ.get("GROQ_API_KEY")

with st.sidebar:
    st.header("Ayarlar")
    if env_api_key:
        st.success(".env dosyasından Groq API anahtarı bulundu ✅")
        api_key = env_api_key
    else:
        api_key = st.text_input("Groq API Anahtarı", type="password")
        st.caption(
            "Anahtarını https://console.groq.com/keys adresinden "
            "ücretsiz alabilirsin."
        )
    top_k = st.slider("Kaç parça getirilsin (top_k)", min_value=1, max_value=10, value=3)

if "rag_index" not in st.session_state:
    st.session_state.rag_index = None
    st.session_state.doc_name = None

uploaded_file = st.file_uploader("Ders notunu yükle (.docx)", type=["docx"])

if uploaded_file is not None and st.session_state.doc_name != uploaded_file.name:
    with st.spinner("Belge okunuyor ve indeksleniyor... (ilk seferde embedding modeli indirilir)"):
        temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        text = read_docx(temp_path)
        index = RagIndex()
        index.index_document(text)

        st.session_state.rag_index = index
        st.session_state.doc_name = uploaded_file.name

    st.success(f"'{uploaded_file.name}' indekslendi ({len(st.session_state.rag_index.chunks)} parça).")

if st.session_state.rag_index is not None:
    question = st.text_input("Sorunu yaz:")

    if st.button("Sor") and question:
        if not api_key:
            st.error("Lütfen sol menüden Groq API anahtarını gir.")
        else:
            with st.spinner("Cevap hazırlanıyor..."):
                answer, sources = answer_question(
                    st.session_state.rag_index, question, top_k=top_k, api_key=api_key
                )

            st.markdown("### Cevap")
            st.write(answer)

            with st.expander("Kullanılan kaynak parçaları (kaynakça)"):
                for i, chunk in enumerate(sources, 1):
                    st.markdown(f"**Parça {i}:**")
                    st.write(chunk)
                    st.divider()
else:
    st.info("Önce bir .docx dosyası yükle.") bunu dener misin
