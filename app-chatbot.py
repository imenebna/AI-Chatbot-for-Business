import os
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Local neural embeddings
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


st.set_page_config(page_title="AI Business Chatbot", page_icon="🤖", layout="wide")


# -----------------------------
# LOCAL knowledge base (default)
# -----------------------------
KB_FILES = [
    "ai_business_guide.md",  # created next to this app by default
]

# Retrieval defaults
TOP_K = 6


@dataclass
class Chunk:
    source: str          # file name
    section: str         # markdown heading
    idx: int
    text: str


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def normalize_newlines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_markdown_sections(md: str) -> List[Tuple[str, str]]:
    """
    Returns list of (section_title, section_text).
    Splits on markdown headings (#, ##, ###...). Keeps content paragraphs.
    """
    md = normalize_newlines(md)
    lines = md.split("\n")

    sections: List[Tuple[str, List[str]]] = []
    current_title = "Introduction"
    current_buf: List[str] = []

    heading_re = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")

    for line in lines:
        m = heading_re.match(line)
        if m:
            # flush previous
            if current_buf:
                sections.append((current_title, current_buf))
            current_title = m.group(2).strip()
            current_buf = []
        else:
            current_buf.append(line)

    if current_buf:
        sections.append((current_title, current_buf))

    # Clean + join
    out: List[Tuple[str, str]] = []
    for title, buf in sections:
        text = "\n".join(buf)
        # remove horizontal rules and excess blank lines
        text = re.sub(r"(?m)^\s*---\s*$", "", text)
        text = normalize_newlines(text)
        if len(text) >= 200:
            out.append((title, text))
    return out


def split_into_paragraphs(text: str) -> List[str]:
    text = normalize_newlines(text)
    paras = [p.strip() for p in text.split("\n\n")]
    paras = [p for p in paras if len(p) >= 60]
    return paras


def build_paragraph_chunks(paras: List[str], max_chars: int = 1200, overlap_paras: int = 1) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for p in paras:
        p_len = len(p) + 2
        if current and current_len + p_len > max_chars:
            chunks.append("\n\n".join(current))
            current = current[-overlap_paras:] if overlap_paras > 0 else []
            current_len = sum(len(x) + 2 for x in current)
        current.append(p)
        current_len += p_len

    if current:
        chunks.append("\n\n".join(current))
    return chunks


@st.cache_data(show_spinner=False, ttl=3600)
def load_kb_chunks() -> Tuple[List[Chunk], List[Dict[str, Any]]]:
    chunks: List[Chunk] = []
    failures: List[Dict[str, Any]] = []

    base_dir = os.path.dirname(os.path.abspath(__file__))

    for kb in KB_FILES:
        try:
            path = os.path.join(base_dir, kb)
            raw = read_text_file(path)
            sections = parse_markdown_sections(raw)

            for section_title, section_text in sections:
                paras = split_into_paragraphs(section_text)
                para_chunks = build_paragraph_chunks(paras, max_chars=1200, overlap_paras=1)
                for i, ch in enumerate(para_chunks):
                    chunks.append(Chunk(source=kb, section=section_title, idx=i, text=ch))

        except Exception as e:
            failures.append({"file": kb, "error": str(e)})

    return chunks, failures


@st.cache_resource(show_spinner=False)
def get_embedder():
    if SentenceTransformer is None:
        return None
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts_local(embedder, texts: List[str]) -> np.ndarray:
    vecs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vecs, dtype=np.float32)


def top_k_retrieve(query_vec: np.ndarray, chunk_vecs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    sims = cosine_similarity(query_vec.reshape(1, -1), chunk_vecs).ravel()
    top_idx = sims.argsort()[-k:][::-1]
    return top_idx, sims[top_idx]


def safe_trim(text: str, max_chars: int = 1400) -> str:
    text = normalize_newlines(text)
    # soften spacing but keep paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    return text[:max_chars] + ("…" if len(text) > max_chars else "")


def format_retrieval_answer(question: str, contexts: List[Chunk]) -> str:
    blocks: List[str] = []
    blocks.append("### Retrieved passages (RAG-only)\n")
    blocks.append(f"**Question:** {question}\n")
    blocks.append("---\n")

    for i, c in enumerate(contexts, start=1):
        blocks.append(f"#### [S{i}] {c.section}")
        blocks.append(f"_Source:_ {c.source}\n")
        blocks.append(safe_trim(c.text, 1200))
        blocks.append("\n---\n")

    return "\n".join(blocks)


def diversify_contexts(chunks: List[Chunk], idxs: List[int], max_k: int) -> List[Chunk]:
    """
    Reduce 'same answer' effect by de-duplicating by section and near-identical text.
    """
    selected: List[Chunk] = []
    seen_sections = set()
    seen_signatures = set()

    for i in idxs:
        c = chunks[i]
        sig = (c.section, re.sub(r"\W+", "", c.text.lower())[:180])
        if c.section in seen_sections:
            continue
        if sig in seen_signatures:
            continue
        selected.append(c)
        seen_sections.add(c.section)
        seen_signatures.add(sig)
        if len(selected) >= max_k:
            break

    # If not enough, fill without section constraint
    if len(selected) < max_k:
        for i in idxs:
            c = chunks[i]
            sig = (c.section, re.sub(r"\W+", "", c.text.lower())[:180])
            if sig in seen_signatures:
                continue
            selected.append(c)
            seen_signatures.add(sig)
            if len(selected) >= max_k:
                break

    return selected


# Button queries + optional section filters (to force different retrieval)
SUGGESTIONS = [
    {"label": "Business model options (AI SaaS vs services)", "q": "Compare AI SaaS vs AI services/agency models and when to choose each.", "section": "Strategy: choose the right business model"},
    {"label": "Pick the right problem to solve", "q": "How do I choose a high-value problem for an AI business and avoid bad problem choices?", "section": "Problem selection: what to build (and what not to)"},
    {"label": "Data strategy for an AI product", "q": "What data do I need and how do I design a safe data strategy for an AI business?", "section": "Data: the fuel and the constraint"},
    {"label": "MVP design with RAG (no fine-tuning)", "q": "How do I build an MVP using RAG and workflow automation? What should the first version include?", "section": "Build an MVP: start with RAG and automation"},
    {"label": "Validate demand in 14 days", "q": "Give me a 14-day validation plan to prove demand for an AI product and get a paid pilot.", "section": "Validation: prove demand in 14 days"},
    {"label": "Pricing models and ROI story", "q": "How should I price an AI product? Compare subscription vs usage-based and how to build an ROI story.", "section": "Pricing: how AI products should be priced"},
    {"label": "Go-to-market for first customers", "q": "How do I get my first customers for an AI product? Give an actionable early GTM plan.", "section": "Go-to-market (GTM): getting the first customers"},
    {"label": "Risks and compliance guardrails", "q": "What are the main risks of AI products and what guardrails should I implement for compliance?", "section": "Risks, safety, and compliance"},
    {"label": "KPIs and metrics to track", "q": "What metrics should I track for an AI product (RAG) and for the business?", "section": "Metrics: what to measure"},
    {"label": "30-60-90 execution plan", "q": "Create a 30-60-90 day plan to launch and scale an AI business.", "section": "Execution plan: 30-60-90 days"},
]



def main():
    st.title("AI Business Chatbot 🤖")
    st.write("Chatbot using a local knowledge base document: AI Business Guide.")
    st.caption("Sources are local files. Results are shown as paragraph passages with section labels.")
    st.divider()

    chunks, failures = load_kb_chunks()
    if not chunks:
        st.error("Knowledge base failed to load.")
        if failures:
            st.code("\n".join([f"{f['file']} -> {f['error']}" for f in failures])[:5000])
        st.info("Ensure ai_business_guide.md is in the same folder as app-chatbot.py.")
        return

    embedder = get_embedder()
    if embedder is None:
        st.error("Missing dependency: sentence-transformers. Install requirements_chatbot.txt then rerun.")
        return

    @st.cache_resource(show_spinner=False)
    def embed_all(texts):
        return embed_texts_local(embedder, texts)

    with st.spinner("Preparing semantic index... (first run may take a bit)"):
        chunk_vecs = embed_all([c.text for c in chunks])

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask a question about building a business with AI. I will retrieve and show the most relevant passages as [S1], [S2], …"}
        ]

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    def retrieve_contexts(q, section_hint, k):
        q_vec = embed_texts_local(embedder, [q])[0]
        top_idx, _ = top_k_retrieve(q_vec, chunk_vecs, k=min(k * 3, len(chunks)))
        return diversify_contexts(chunks, top_idx.tolist(), max_k=k)

    def handle_question(q, section_hint=None):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant passages..."):
                contexts = retrieve_contexts(q, section_hint, k=TOP_K)
                answer = format_retrieval_answer(q, contexts)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # Chat input
    question = st.chat_input("Type your question and press Enter…")
    if question:
        st.session_state.pending_question = {"q": question, "section": None}

    # Process queued question
    if st.session_state.pending_question:
        pq = st.session_state.pending_question
        st.session_state.pending_question = None
        with st.chat_message("user"):
            st.markdown(pq["q"])
        handle_question(pq["q"], pq.get("section"))

    # Render history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Guided buttons
    st.subheader("Guided questions")
    c1, c2 = st.columns(2)
    for i, s in enumerate(SUGGESTIONS):
        col = c1 if i % 2 == 0 else c2
        if col.button(s["label"], use_container_width=True, key=f"sug_{i}"):
            st.session_state.pending_question = {"q": s["q"], "section": s["section"]}

    st.divider()
    st.caption("Mode: Information Retrieval. Knowledge base: ai_business_guide.md")

if __name__ == "__main__":
    main()
