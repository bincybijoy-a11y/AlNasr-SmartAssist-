import os
import faiss
import pickle
import numpy as np
import streamlit as st
import re
import json

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Enterprise AI Assistant", layout="wide")

# =========================
# PATHS
# =========================
INDEX_PATH = "storage/index.faiss"
CHUNKS_PATH = "storage/chunks.pkl"
META_PATH = "storage/meta.json"
os.makedirs("storage", exist_ok=True)

# =========================
# CLEANING
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
# MODELS
# =========================
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-mpnet-base-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    return embedder, reranker

embedder, reranker = load_models()

# =========================
# FIXED LLM (STREAMLIT SAFE)
# =========================
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer
    )

llm = load_llm()

# =========================
# SAFE STORAGE HELPERS
# =========================
def safe_pickle_save(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp, path)

def safe_json_save(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)

# =========================
# BUILD KNOWLEDGE BASE
# =========================
def build_kb():
    folder = "data/"

    if not os.path.exists(folder) or len(os.listdir(folder)) == 0:
        st.error("📂 Upload PDFs inside /data folder")
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

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # SAVE
    faiss.write_index(index, INDEX_PATH)
    safe_pickle_save(CHUNKS_PATH, chunks)
    safe_json_save(META_PATH, {"chunks": len(chunks)})

    st.success("✅ Knowledge base built successfully")

    return index, chunks

# =========================
# SAFE LOAD (FIXED)
# =========================
def load_or_build():
    try:
        if not (
            os.path.exists(INDEX_PATH)
            and os.path.exists(CHUNKS_PATH)
            and os.path.exists(META_PATH)
        ):
            st.info("📦 Building knowledge base...")
            return build_kb()

        index = faiss.read_index(INDEX_PATH)

        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)

        with open(META_PATH, "r") as f:
            meta = json.load(f)

        # FAISS DIMENSION CHECK (CRITICAL FIX)
        test_vec = embedder.encode(["test"]).astype("float32")

        if test_vec.shape[1] != index.d:
            st.warning("⚠️ FAISS mismatch detected. Rebuilding...")
            return build_kb()

        if len(chunks) == 0 or meta.get("chunks", 0) == 0:
            st.warning("⚠️ Empty storage detected. Rebuilding...")
            return build_kb()

        return index, chunks

    except Exception as e:
        st.warning(f"⚠️ Corrupted storage. Rebuilding... {e}")
        return build_kb()

# INIT
index, chunks = load_or_build()

# =========================
# BM25
# =========================
@st.cache_resource
def build_bm25(chunks):
    tokenized = [c.lower().split() for c in chunks]
    return BM25Okapi(tokenized)

bm25 = build_bm25(chunks)

# =========================
# RETRIEVAL
# =========================
def retrieve(query, faiss_k=30, bm25_k=30):
    q_emb = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)

    _, faiss_ids = index.search(q_emb, faiss_k)
    faiss_candidates = set(faiss_ids[0])

    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_candidates = set(np.argsort(bm25_scores)[-bm25_k:])

    return list(faiss_candidates | bm25_candidates)

# =========================
# RERANK
# =========================
def rerank(query, candidates, top_k=5):
    if not candidates:
        return []

    pairs = [(query, chunks[i]) for i in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [i for i, _ in ranked[:top_k]]

# =========================
# LLM ANSWER
# =========================
def llm_answer(query, context):
    prompt = f"""
Answer ONLY using the context below.

Question:
{query}

Context:
{context}

Answer:
"""

    result = llm(prompt, max_new_tokens=256, do_sample=False)
    output = result[0]["generated_text"]

    return output.split("Answer:")[-1].strip()

# =========================
# CHAT STATE
# =========================
if "chat" not in st.session_state:
    st.session_state.chat = []

# =========================
# UI
# =========================
st.title("💬 Chat Assistant")

with st.sidebar:
    st.title("🏢 Al Nasr AI")
    st.caption("Enterprise Document Assistant")
    st.markdown("---")

    if st.button("🧹 Clear Chat"):
        st.session_state.chat = []
        st.rerun()

query = st.chat_input("Ask your question...")

# =========================
# PIPELINE
# =========================
if query:
    with st.spinner("Searching documents..."):
        candidates = retrieve(query)
        top_chunks = rerank(query, candidates)

        context = "\n\n".join([chunks[i] for i in top_chunks])

        answer = llm_answer(query, context)

        if not answer:
            answer = clean_text(context[:800])

    st.session_state.chat.append(("user", query))
    st.session_state.chat.append(("bot", answer))

# =========================
# CHAT DISPLAY
# =========================
for role, msg in st.session_state.chat:
    if role == "user":
        with st.chat_message("user"):
            st.markdown(msg)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg)