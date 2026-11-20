"""
IBM TechQA Hybrid RAG Assistant
Local Windows Edition

Pipeline:
    BM25 Keyword Search
    +
    ChromaDB Semantic Search
    +
    Reciprocal Rank Fusion (RRF)
    +
    Google Gemini Answer Generation

Local requirements:
- chunks.pkl
- chroma_techqa_db/
- .env
"""

# ============================================================
# 1. ENVIRONMENT CONFIGURATION
# ============================================================

import os
import time
import pickle
import traceback
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = BASE_DIR / "chunks.pkl"
CHROMA_PATH = BASE_DIR / "chroma_techqa_db"
ENV_PATH = BASE_DIR / ".env"

# ============================================================
# 2. LOAD LOCAL ENVIRONMENT VARIABLES
# ============================================================

def load_local_env():
    if not ENV_PATH.exists():
        print("[WARNING] .env file not found.", flush=True)
        return

    print("[INIT] Loading environment variables...", flush=True)
    try:
        with open(ENV_PATH, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                    value = value[1:-1]

                if key and key not in os.environ:
                    os.environ[key] = value

        print("[INIT] Environment variables loaded.", flush=True)
    except Exception as e:
        print(f"[WARNING] Could not load .env file: {e}", flush=True)

load_local_env()

# ============================================================
# 3. IMPORT LIBRARIES
# ============================================================

print("[INIT] Importing libraries...", flush=True)

import gradio as gr
from google import genai
from google.genai import types
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

# ============================================================
# 4. CONFIGURATION
# ============================================================

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
TOP_K = 4
RRF_CONSTANT = 60
BM25_WEIGHT = 0.5
VECTOR_WEIGHT = 0.5

# ============================================================
# 5. LOAD DOCUMENT CHUNKS
# ============================================================

print("\n==============================================")
print(" IBM TECHQA HYBRID RAG ASSISTANT")
print("==============================================")
print("[INIT] Loading IBM TechQA RAG Pipeline...", flush=True)
print("[INIT] Loading document chunks...", flush=True)

if not CHUNKS_PATH.exists():
    raise FileNotFoundError(f"chunks.pkl was not found:\n{CHUNKS_PATH}")

try:
    with open(CHUNKS_PATH, "rb") as file:
        chunks = pickle.load(file)

    if not chunks:
        raise ValueError("chunks.pkl contains no documents.")

    print(f"[INIT] Loaded {len(chunks):,} document chunks.", flush=True)
except Exception as e:
    raise RuntimeError(f"Document loading failed: {e}") from e

# ============================================================
# 6. INITIALIZE BM25
# ============================================================

print("[INIT] Initializing BM25...", flush=True)
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = TOP_K
print("[INIT] BM25 ready.", flush=True)

# ============================================================
# 7. INITIALIZE EMBEDDINGS ON CPU
# ============================================================

print("[INIT] Loading HuggingFace embeddings...", flush=True)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("[INIT] CPU embeddings ready.", flush=True)

# ============================================================
# 8. CONNECT TO EXISTING CHROMADB
# ============================================================

print("[INIT] Connecting to ChromaDB...", flush=True)

if not CHROMA_PATH.exists():
    raise FileNotFoundError(f"ChromaDB folder was not found:\n{CHROMA_PATH}")

vectorstore = Chroma(
    persist_directory=str(CHROMA_PATH),
    embedding_function=embeddings,
)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

try:
    chroma_count = vectorstore._collection.count()
    print(f"[INIT] ChromaDB documents: {chroma_count:,}", flush=True)
    if chroma_count == 0:
        print("[WARNING] ChromaDB collection is empty!", flush=True)
except Exception as e:
    print(f"[WARNING] ChromaDB diagnostic failed: {e}", flush=True)

print("==============================================")
print("[INIT] RAG Pipeline Ready!")
print("==============================================\n")

# ============================================================
# 9. HYBRID RETRIEVAL — RECIPROCAL RANK FUSION
# ============================================================

def hybrid_retrieve(query, top_k=TOP_K):
    if not query or not query.strip():
        return []

    query = query.strip()

    print("[RETRIEVAL] Running BM25...", flush=True)
    bm25_docs = bm25_retriever.invoke(query)

    print("[RETRIEVAL] Running ChromaDB...", flush=True)
    try:
        vector_docs = vector_retriever.invoke(query)
    except Exception as e:
        print(f"[WARNING] ChromaDB retrieval failed: {e}", flush=True)
        vector_docs = []

    rrf_scores = {}
    doc_map = {}

    for rank, doc in enumerate(vector_docs, start=1):
        key = doc.page_content
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (VECTOR_WEIGHT / (rank + RRF_CONSTANT))
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_docs, start=1):
        key = doc.page_content
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (BM25_WEIGHT / (rank + RRF_CONSTANT))
        doc_map[key] = doc

    sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
    results = [doc_map[k] for k in sorted_keys[:top_k]]

    print(f"[RETRIEVAL] Returning {len(results)} documents.", flush=True)
    return results

# ============================================================
# 10. GEMINI CLIENT
# ============================================================

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key is missing!\n\n"
            "Create a .env file in your project folder and add:\n\n"
            "GEMINI_API_KEY=your_actual_api_key_here"
        )

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=30000),  # 30-second timeout
    )

# ============================================================
# 11. ANSWER GENERATION
# ============================================================

def answer_query(query):
    start_time = time.time()
    print(f"\n=== [QUERY] {query} ===", flush=True)

    if not query or not query.strip():
        return "Please enter a technical question.", "No sources retrieved."

    query = query.strip()

    try:
        retrieval_start = time.time()
        retrieved_docs = hybrid_retrieve(query, top_k=TOP_K)
        retrieval_time = time.time() - retrieval_start

        if not retrieved_docs:
            return (
                "I cannot answer this based on the provided IBM TechNotes.",
                "No relevant documents retrieved.",
            )

        context_parts = []
        for index, doc in enumerate(retrieved_docs, start=1):
            source = doc.metadata.get("source", "Unknown source")
            context_parts.append(
                f"DOCUMENT {index}\nSource File: {source}\nContent:\n{doc.page_content}"
            )
        context_str = "\n\n".join(context_parts)

        prompt = f"""You are an expert IBM IT Support RAG Assistant.

Your task is to answer technical troubleshooting questions using ONLY the IBM TechNotes provided below.

STRICT INSTRUCTIONS:
1. Use only the supplied IBM TechNote information.
2. Never use outside knowledge.
3. Never invent commands, configurations, troubleshooting steps, or solutions.
4. Always cite the exact source filename.
5. Explain troubleshooting steps in a clear sequence.
6. Clearly distinguish documented solutions from workarounds.
7. If the root cause is not documented, explicitly say so.
8. If the context does not contain relevant information, respond exactly:
"I cannot answer this based on the provided IBM TechNotes."
9. Do not claim a solution is officially recommended unless the provided TechNote explicitly says so.
10. If multiple documents contain relevant information, combine their information carefully and cite each source.
11. Do not assume that a command or solution from one IBM product version applies to another version.
12. Use readable formatting with headings and numbered steps where appropriate.
13. Avoid repeating the same information unnecessarily.
14. Do not fabricate citations.

============================================================
IBM TECHNOTE CONTEXT:
{context_str}
============================================================
USER QUESTION:
{query}
============================================================
ANSWER:
"""

        print("[GEMINI] Generating response...", flush=True)
        generation_start = time.time()
        client = get_gemini_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        generation_time = time.time() - generation_start

        answer = response.text or "Gemini returned an empty response."

        source_parts = []
        for index, doc in enumerate(retrieved_docs, start=1):
            source = doc.metadata.get("source", "Unknown source")
            content_preview = doc.page_content[:1200]
            if len(doc.page_content) > 1200:
                content_preview += "\n\n[Preview truncated...]"

            source_parts.append(
                f"### Document {index}\n\n**Source:** {source}\n\n**Retrieved Content:**\n\n{content_preview}"
            )
        sources_md = "\n\n---\n\n".join(source_parts)

        total_time = time.time() - start_time
        timing_info = f"""

---

### Performance
- Retrieval time: {retrieval_time:.2f} seconds
- Gemini generation: {generation_time:.2f} seconds
- Total processing time: {total_time:.2f} seconds
- Documents retrieved: {len(retrieved_docs)}
- Model: {GEMINI_MODEL}
"""
        answer += timing_info

        print(f"[SUCCESS] Retrieval: {retrieval_time:.2f}s", flush=True)
        print(f"[SUCCESS] Gemini: {generation_time:.2f}s", flush=True)
        print(f"[SUCCESS] Total: {total_time:.2f}s", flush=True)

        return answer, sources_md

    except Exception as e:
        print("\n=== [QUERY ERROR] ===", flush=True)
        print(traceback.format_exc(), flush=True)

        error_message = str(e)
        if (
            "API_KEY_INVALID" in error_message
            or "invalid api key" in error_message.lower()
            or "API key not valid" in error_message
        ):
            error_message = (
                "Your Gemini API key appears to be invalid.\n\n"
                "Check the GEMINI_API_KEY value inside your .env file."
            )
        elif "404" in error_message or "not found" in error_message.lower():
            error_message = (
                f"Gemini model '{GEMINI_MODEL}' could not be found or is unavailable.\n\n"
                "Check the GEMINI_MODEL value inside your .env file."
            )
        elif "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
            error_message = (
                "Gemini API rate limit reached.\n\n"
                "Please wait a little and try again."
            )

        return (
            f"Error processing your question:\n\n{error_message}",
            "An error occurred during processing.",
        )

# ============================================================
# 12. GRADIO INTERFACE
# ============================================================

CUSTOM_CSS = """
/* Page background - Deep Midnight Navy */
html, body, .gradio-container {
    background-color: #071426 !important;
    color: #E0F2FE !important;
    font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}

.gradio-container {
    max-width: 920px !important;
    margin: 0 auto !important;
    padding: 28px 18px !important;
}

/* Header Card - Layered Ocean Navy */
.header-box {
    text-align: center;
    background: linear-gradient(145deg, #0D213F 0%, #112C54 60%, #0A1B36 100%);
    border: 1.5px solid #2563EB;
    border-radius: 16px;
    padding: 26px 20px 22px 20px;
    margin-bottom: 18px;
    box-shadow: 0 8px 28px rgba(3, 12, 27, 0.6);
}

.header-box h1 {
    margin: 12px 0 6px 0 !important;
    color: #E0F2FE !important;
    font-size: 1.95rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    text-align: center !important;
}

.header-box p {
    margin: 4px 0 !important;
    color: #93C5FD !important;
    text-align: center !important;
    font-size: 0.98rem !important;
}

.header-badge {
    display: inline-block;
    margin-top: 12px;
    padding: 5px 15px;
    border-radius: 999px;
    background-color: #173A6E;
    border: 1px solid #3B82F6;
    color: #BAE6FD;
    font-size: 0.82rem;
    font-weight: 600;
}

/* Query & Answer Cards - Rich Navy */
.card-panel {
    background-color: #0D213F !important;
    border: 1.5px solid #1E4375 !important;
    border-radius: 14px !important;
    padding: 20px !important;
    margin-top: 16px !important;
    box-shadow: 0 6px 20px rgba(3, 12, 27, 0.45) !important;
}

/* Sources Card - Distinct Steel Blue Accent */
.sources-panel {
    background-color: #0B1D38 !important;
    border: 1.5px solid #1E4375 !important;
    border-left: 4px solid #38BDF8 !important;
    border-radius: 14px !important;
    padding: 20px !important;
    margin-top: 16px !important;
    box-shadow: 0 6px 20px rgba(3, 12, 27, 0.45) !important;
}

/* Strip inner Gradio block backgrounds so no white bleeds through */
.card-panel .block, .sources-panel .block, .form {
    background-color: transparent !important;
    border-color: #1E4375 !important;
}

/* Labels */
label span {
    color: #93C5FD !important;
    font-weight: 600 !important;
}

/* Textbox - Deep Sapphire */
textarea, input[type="text"] {
    background-color: #08172E !important;
    color: #E0F2FE !important;
    border: 1.5px solid #255391 !important;
    border-radius: 10px !important;
}
textarea::placeholder, input[type="text"]::placeholder {
    color: #60A5FA !important;
    opacity: 0.65;
}
textarea:focus, input[type="text"]:focus {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.22) !important;
}

/* Primary Button - Electric Cobalt */
button.primary-btn {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: #E0F2FE !important;
    border: 1px solid #60A5FA !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: all 0.15s ease;
}
button.primary-btn:hover {
    background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
}

/* Clear Button - Muted Steel Blue */
button.clear-btn {
    background: #16335E !important;
    color: #BAE6FD !important;
    border: 1px solid #2A528A !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}
button.clear-btn:hover {
    background: #1E4375 !important;
}

/* Example Chips */
.examples-holder button, .gallery button, table.table, table.table td {
    background-color: #102A50 !important;
    color: #BAE6FD !important;
    border: 1px solid #234B82 !important;
    border-radius: 8px !important;
}
.examples-holder button:hover, .gallery button:hover, table.table tr:hover td {
    background-color: #183C70 !important;
    border-color: #38BDF8 !important;
    color: #E0F2FE !important;
}

/* Markdown Typography - Ice & Sky Blue */
.prose, .prose *, .markdown-body, .markdown-body * {
    color: #DBEAFE !important;
}

.prose h2 {
    color: #7DD3FC !important;
    font-size: 1.25rem !important;
    margin-bottom: 10px !important;
}

.prose h3 {
    color: #60A5FA !important;
    border-bottom: 1px solid #1E4375;
    padding-bottom: 5px;
}

.prose strong {
    color: #E0F2FE !important;
}

.prose hr {
    border-color: #1E4375 !important;
}

.prose code, .prose pre {
    background-color: #061224 !important;
    color: #7DD3FC !important;
    border: 1px solid #1E4375 !important;
    padding: 2px 6px;
    border-radius: 6px;
    font-family: 'Consolas', 'Courier New', monospace !important;
}
"""

HEADER_HTML = """
<div class="header-box">
    <svg width="78" height="78" viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg" style="display:block; margin: 0 auto;">
        <rect x="6" y="6" width="108" height="108" rx="24" fill="#08172E" stroke="#2563EB" stroke-width="3"/>
        <rect x="28" y="26" width="46" height="58" rx="7" fill="#112C54" stroke="#60A5FA" stroke-width="3.5"/>
        <line x1="39" y1="42" x2="63" y2="42" stroke="#38BDF8" stroke-width="4" stroke-linecap="round"/>
        <line x1="39" y1="54" x2="59" y2="54" stroke="#60A5FA" stroke-width="4" stroke-linecap="round"/>
        <line x1="39" y1="66" x2="52" y2="66" stroke="#3B82F6" stroke-width="4" stroke-linecap="round"/>
        <circle cx="77" cy="77" r="17" fill="#2563EB" stroke="#08172E" stroke-width="4"/>
        <circle cx="75" cy="75" r="6" stroke="#E0F2FE" stroke-width="3"/>
        <line x1="80" y1="80" x2="86" y2="86" stroke="#E0F2FE" stroke-width="3" stroke-linecap="round"/>
    </svg>
    <h1>IBM TechQA Hybrid RAG Assistant</h1>
    <p>Ask technical troubleshooting questions across <strong>496 official IBM TechNotes</strong>.</p>
    <span class="header-badge">Local Retrieval &bull; Gemini AI Generation</span>
</div>
"""

with gr.Blocks(title="IBM TechQA Hybrid RAG Assistant") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Column(elem_classes="card-panel"):
        query_input = gr.Textbox(
            label="Technical Support Query",
            placeholder="Enter your IBM technical question...",
            lines=3,
            autofocus=True,
        )

        with gr.Row():
            submit_btn = gr.Button(
                "Search & Synthesize Answer",
                elem_classes="primary-btn",
            )
            clear_btn = gr.Button(
                "Clear",
                elem_classes="clear-btn",
            )

        gr.Examples(
            examples=[
                ["How do I fix .bashrc environment variables not getting picked up in IBM Streams 4.1.1.1?"],
                ["How do I resolve SQL state 08003 connection does not exist error in the database toolkit?"],
                ["How do I configure a Kubernetes ingress controller on AWS EKS?"],
                ["What causes connection timeout errors in IBM WebSphere?"],
                ["How can I troubleshoot an IBM MQ connection issue?"],
            ],
            inputs=query_input,
        )

    with gr.Column(elem_classes="card-panel"):
        gr.Markdown("## Synthesized Solution")
        answer_output = gr.Markdown()

    with gr.Column(elem_classes="sources-panel"):
        gr.Markdown("## Retrieved IBM TechNote Sources")
        sources_output = gr.Markdown()

    submit_btn.click(
        fn=answer_query,
        inputs=query_input,
        outputs=[answer_output, sources_output],
        concurrency_limit=1,
    )

    query_input.submit(
        fn=answer_query,
        inputs=query_input,
        outputs=[answer_output, sources_output],
        concurrency_limit=1,
    )

    clear_btn.click(
        fn=lambda: ("", "", ""),
        inputs=None,
        outputs=[query_input, answer_output, sources_output],
    )

# ============================================================
# 13. LAUNCH APPLICATION
# ============================================================

if __name__ == "__main__":
    print("==============================================", flush=True)
    print("[STARTUP] Launching IBM TechQA Assistant", flush=True)
    print(f"[STARTUP] Gemini model: {GEMINI_MODEL}", flush=True)
    print("==============================================", flush=True)

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False,
        show_error=True,
        ssr_mode=False,
        css=CUSTOM_CSS,
        theme=gr.themes.Base(),
    )


# ============================================================
# 13. LAUNCH APPLICATION
# ============================================================

if __name__ == "__main__":
    print("==============================================", flush=True)
    print("[STARTUP] Launching IBM TechQA Assistant", flush=True)
    print(f"[STARTUP] Gemini model: {GEMINI_MODEL}", flush=True)
    print("==============================================", flush=True)

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False,
        show_error=True,
        ssr_mode=False,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(),
    )

# ============================================================
# 13. LAUNCH APPLICATION
# ============================================================

if __name__ == "__main__":
    print("==============================================", flush=True)
    print("[STARTUP] Launching IBM TechQA Assistant", flush=True)
    print(f"[STARTUP] Gemini model: {GEMINI_MODEL}", flush=True)
    print("==============================================", flush=True)

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False,
        show_error=True,
        ssr_mode=False,
        css=CUSTOM_CSS,
        theme=gr.themes.Base(),
    )