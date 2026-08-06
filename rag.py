"""
rag.py
------
Basit bir RAG (Retrieval-Augmented Generation) sisteminin çekirdek mantığı.

RAG 4 adımdan oluşur:
1) YÜKLEME     : Belgeyi okuyup düz metne çeviririz.
2) PARÇALAMA   : Metni küçük parçalara (chunk) böleriz.
3) EMBEDDING   : Her parçayı bir sayı vektörüne (embedding) çeviririz.
4) ARAMA+ÜRETİM: Kullanıcının sorusunu da vektöre çevirip, en yakın (en alakalı)
                 parçaları buluruz ve Groq üzerinden çalışan modele bağlam
                 olarak veririz.

LLM olarak Groq API kullanılıyor (ücretsiz kota, kredi kartı gerekmez,
https://console.groq.com/keys).
"""

import os
import numpy as np
from docx import Document
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def read_docx(file_path: str) -> str:
    """Bir .docx dosyasını okuyup tüm paragrafları tek bir metin olarak döner."""
    doc = Document(file_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def split_into_chunks(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Metni chunk_size karakterlik, overlap kadar üst üste binen parçalara böler."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]

import streamlit as st

@st.cache_resource
def load_embedding_model(model_name: str):
    return SentenceTransformer(model_name)

class RagIndex:
    """Belgeyi indeksler (chunk + embed) ve soru geldiğinde en alakalı parçaları bulur."""

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = load_embedding_model(embedding_model_name)
        self.chunks: List[str] = []
        self.chunk_embeddings: np.ndarray | None = None

    def index_document(self, text: str):
        self.chunks = split_into_chunks(text)
        self.chunk_embeddings = self.embedder.encode(
            self.chunks, normalize_embeddings=True
        )

    def retrieve(self, question: str, top_k: int = 3) -> list[str]:
        if self.chunk_embeddings is None:
            raise ValueError("Önce index_document() ile bir belge indekslemelisin.")
        question_embedding = self.embedder.encode([question], normalize_embeddings=True)[0]
        similarities = self.chunk_embeddings @ question_embedding
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]


def ask_groq(question: str, context_chunks: list[str], api_key: str | None = None) -> str:
    """Bağlam parçalarını ve soruyu Groq'a gönderir, bağlama dayalı cevap ister."""
    raw_key = api_key or os.environ.get("GROQ_API_KEY", "")
    # Windows'ta Türkçe klavye/otomatik düzeltme yüzünden anahtara karışabilen
    # ASCII-olmayan karakterleri temizler.
    clean_key = raw_key.encode("ascii", "ignore").decode("ascii").strip()

    client = Groq(api_key=clean_key)

    context_text = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "Sen bir ders asistanısın. Sana verilen ders notu parçalarına dayanarak "
        "öğrencinin sorusunu cevapla. Eğer cevap verilen notlarda yoksa, bunu "
        "açıkça belirt; notlarda olmayan bilgiyi uydurma."
    )

    user_message = f"""Ders notu parçaları:
{context_text}

Soru: {question}

Yukarıdaki ders notu parçalarına dayanarak soruyu cevapla."""

    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


def answer_question(rag_index: RagIndex, question: str, top_k: int = 3, api_key: str | None = None):
    """Soruyu al -> alakalı parçaları bul -> Groq'a sor -> cevabı ve kaynakları döndür."""
    relevant_chunks = rag_index.retrieve(question, top_k=top_k)
    answer = ask_groq(question, relevant_chunks, api_key=api_key)
    return answer, relevant_chunks