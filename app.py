import os
import faiss
import pickle
import numpy as np
import streamlit as st
import re
import json

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Enterprise AI Assistant", layout="wide")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("🏢 Al Nasr AI")
    st.caption("Enterprise Document Assistant")
    st.markdown("---")
    if st.button("🧹 Clear Chat"):
        st.session_state.chat = []
        st.rerun()

# =========================
# HEADER
# =========================
st.title("💬 Chat Assistant")

# =========================
# PATHS
# =========================
INDEX_PATH = "storage/index.faiss"
CHUNKS_PATH = "storage/chunks.pkl"
META_PATH = "storage/meta.json"
os.makedirs("storage", exist_ok=True)

# =========================
# TEXT CLEANING
# =========================
def clean_pdf_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r'-\s+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)
    return text.strip()

# =========================
# MODELS (CLOUD SAFE)
# =========================
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return embedder, reranker

embedder, reranker = load_models()

# =========================
# SAFE SAVE
# =========================
def safe_save(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp, path)

# =========================
# BUILD KNOWLEDGE BASE
# =========================
def build_kb():
    folder = "data/"
    if not os.path.exists(folder):
        os.makedirs(folder)
        st.error("Add PDF files inside /data folder")
        st.stop()

    texts = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder, file))
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    texts.append(clean_pdf_text(t))

    chunks = []
    for t in texts:
        chunks += [t[i:i+500] for i in range(0, len(t), 250)]

    chunks = [c for c in chunks if len(c) > 30]

    embeddings = embedder.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    safe_save(CHUNKS_PATH, chunks)

    with open(META_PATH, "w") as f:
        json.dump({"chunks": len(chunks)}, f)

    return index, chunks

# =========================
# LOAD OR BUILD
# =========================
def load_or_build():
    try:
        if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
            index = faiss.read_index(INDEX_PATH)
            with open(CHUNKS_PATH, "rb") as f:
                chunks = pickle.load(f)
            return index, chunks
    except:
        pass

    st.warning("Building knowledge base...")
    return build_kb()

index, chunks = load_or_build()

# =========================
# RETRIEVAL (FAISS ONLY)
# =========================
def retrieve(query, k=25):
    q_emb = embedder.encode([query]).astype("float32")
    _, I = index.search(q_emb, k)
    return I[0].tolist()

# =========================
# RERANK
# =========================
def rerank(query, candidates, top_k=5):
    pairs = [(query, chunks[i]) for i in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [i for i, _ in ranked[:top_k]]

# =========================
# ANSWER GENERATION (EXTRACTIVE)
# =========================
def generate_answer(context):
    if not context:
        return "No relevant information found in the documents."
    return clean_text(context[:1200])

# =========================
# CHAT STATE
# =========================
if "chat" not in st.session_state:
    st.session_state.chat = []

# =========================
# INPUT
# =========================
query = st.chat_input("Ask about Al Nasr Contracting Company...")

# =========================
# PIPELINE
# =========================
if query:
    with st.spinner("Searching documents..."):
        candidates = retrieve(query)
        top_chunks = rerank(query, candidates)

        context = "\n\n".join(clean_pdf_text(chunks[i]) for i in top_chunks)
        answer = generate_answer(context)

    st.session_state.chat.append(("user", query))
    st.session_state.chat.append(("bot", answer))

# =========================
# CHAT UI
# =========================
for role, msg in st.session_state.chat:
    if role == "user":
        with st.chat_message("user"):
            st.markdown(
                f"<div style='text-align:right;background:#DCF8C6;padding:10px;border-radius:10px'>{msg}</div>",
                unsafe_allow_html=True
            )
    else:
        with st.chat_message("assistant"):
            st.markdown(
                f"<div style='background:#F1F1F1;padding:12px;border-radius:10px'>{msg}</div>",
                unsafe_allow_html=True
            )