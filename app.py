"""
AI PDF Chatbot — Retrieval Augmented Generation (RAG)
=====================================================
Upload any PDF → ask questions → get AI-powered answers grounded in the document.

Tech: Python · Streamlit · LangChain · ChromaDB · Groq / OpenAI
"""

import os
import tempfile
import hashlib
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ━━━━━━━━━━━━━━━━━━  CONSTANTS  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

SYSTEM_PROMPT = """\
You are a helpful AI assistant that answers questions about PDF documents.
Use ONLY the following context passages to answer. If the answer is not in the context, say so honestly.
Be concise, accurate, and cite page numbers when possible.

Context:
{context}
"""

# ━━━━━━━━━━━━━━━━━━  HELPERS  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _file_hash(uploaded_file) -> str:
    """Return a short SHA-256 hex digest to identify a unique PDF."""
    data = uploaded_file.getvalue()
    return hashlib.sha256(data).hexdigest()[:16]


@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Use a free, local sentence-transformer model for embeddings."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def get_llm(provider: str, api_key: str):
    """Return a LangChain chat model for the chosen provider."""
    if provider == "Groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=0.3,
            max_tokens=2048,
        )
    else:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.3,
            max_tokens=2048,
        )


def load_and_chunk_pdf(uploaded_file) -> list:
    """Save uploaded file to a temp path, load pages, split into chunks."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(pages)


def build_vectorstore(chunks, collection_name: str):
    """Create a ChromaDB collection from document chunks."""
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore


def ask_question(llm, retriever, question: str, chat_history: list) -> dict:
    """
    Custom RAG pipeline:
    1. Retrieve relevant chunks
    2. Build a prompt with context + chat history
    3. Invoke the LLM
    4. Return answer + source documents
    """
    # Retrieve relevant documents
    docs = retriever.invoke(question)

    # Build context string from retrieved docs
    context_parts = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[Page {page}]: {doc.page_content}")
    context = "\n\n".join(context_parts)

    # Build prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    # Convert chat history to LangChain messages
    lc_history = []
    for msg in chat_history[-10:]:  # Keep last 10 messages for context window
        if msg["role"] == "user":
            lc_history.append(HumanMessage(content=msg["content"]))
        else:
            lc_history.append(AIMessage(content=msg["content"]))

    # Invoke the chain
    chain = prompt | llm
    response = chain.invoke(
        {
            "context": context,
            "chat_history": lc_history,
            "question": question,
        }
    )

    return {
        "answer": response.content,
        "source_documents": docs,
    }


# ━━━━━━━━━━━━━━━━━━  PAGE CONFIG  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Global ──────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ─────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(195deg, #0f0c29, #302b63, #24243e);
}
section[data-testid="stSidebar"] * { color: #e0e0ff !important; }
section[data-testid="stSidebar"] .stFileUploader label { font-weight: 600; }

/* ── Chat bubbles ────────────────────────────────────── */
.stChatMessage[data-testid="stChatMessage-user"] {
    background: linear-gradient(135deg, #667eea33, #764ba233);
    border-left: 4px solid #667eea;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: .6rem;
}
.stChatMessage[data-testid="stChatMessage-assistant"] {
    background: linear-gradient(135deg, #0f0c2911, #302b6311);
    border-left: 4px solid #a78bfa;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: .6rem;
}

/* ── Hero banner ─────────────────────────────────────── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea, #a78bfa, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: .3rem;
}
.hero p { font-size: 1.1rem; opacity: .7; }

/* ── Feature cards ───────────────────────────────────── */
.feat-grid { display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; margin: 1.5rem 0 2rem; }
.feat-card {
    flex: 1 1 200px;
    max-width: 240px;
    background: linear-gradient(145deg, #667eea15, #a78bfa15);
    border: 1px solid #667eea30;
    border-radius: 14px;
    padding: 1.3rem;
    text-align: center;
    transition: transform .25s, box-shadow .25s;
}
.feat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 30px #667eea22;
}
.feat-icon { font-size: 2rem; margin-bottom: .5rem; }
.feat-title { font-weight: 700; margin-bottom: .25rem; }
.feat-desc { font-size: .85rem; opacity: .65; }

/* ── Status pill ─────────────────────────────────────── */
.status-pill {
    display: inline-flex; align-items: center; gap: .45rem;
    background: linear-gradient(135deg, #667eea22, #a78bfa22);
    border: 1px solid #667eea44;
    border-radius: 999px;
    padding: .35rem .9rem;
    font-size: .82rem;
    font-weight: 600;
    margin-top: .6rem;
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: #34d399; }

/* Buttons */
.stButton>button {
    border-radius: 10px;
    font-weight: 600;
    transition: transform .15s;
}
.stButton>button:hover { transform: scale(1.03); }
</style>
""",
    unsafe_allow_html=True,
)


# ━━━━━━━━━━━━━━━━━━  SIDEBAR  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    provider = st.selectbox("LLM Provider", ["Groq", "OpenAI"], index=0)

    if provider == "Groq":
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            value=os.getenv("GROQ_API_KEY", ""),
        )
    else:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            value=os.getenv("OPENAI_API_KEY", ""),
        )

    st.divider()
    st.markdown("## 📄 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Max 200 MB",
    )

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown(
        """
    <div style='text-align:center; opacity:.45; font-size:.75rem;'>
        Built with 🧠 LangChain &middot; 🗄️ ChromaDB &middot; 🎈 Streamlit
    </div>
    """,
        unsafe_allow_html=True,
    )

# ━━━━━━━━━━━━━━━━━━  MAIN AREA  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Hero / landing ───────────────────────────────────────────────────
if not uploaded_file:
    st.markdown(
        """
    <div class="hero">
        <h1>AI PDF Chatbot</h1>
        <p>Upload a PDF and ask any question — powered by RAG 🚀</p>
    </div>

    <div class="feat-grid">
        <div class="feat-card">
            <div class="feat-icon">📤</div>
            <div class="feat-title">Upload PDF</div>
            <div class="feat-desc">Drag & drop any PDF document up to 200 MB</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">🧩</div>
            <div class="feat-title">Smart Chunking</div>
            <div class="feat-desc">Automatic text splitting for optimal retrieval</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">🗄️</div>
            <div class="feat-title">Vector Search</div>
            <div class="feat-desc">ChromaDB stores embeddings for lightning-fast lookup</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">🤖</div>
            <div class="feat-title">AI Answers</div>
            <div class="feat-desc">LLM generates precise answers grounded in your document</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Validate API key ────────────────────────────────────────────────
if not api_key:
    st.warning(f"⚠️ Please enter your **{provider} API key** in the sidebar to continue.")
    st.stop()

# ── Process PDF ──────────────────────────────────────────────────────
pdf_hash = _file_hash(uploaded_file)
collection_name = f"pdf_{pdf_hash}"

if "current_pdf" not in st.session_state or st.session_state.current_pdf != pdf_hash:
    with st.spinner("📄 Parsing PDF & building vector index…"):
        chunks = load_and_chunk_pdf(uploaded_file)
        vectorstore = build_vectorstore(chunks, collection_name)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        llm = get_llm(provider, api_key)

        st.session_state.current_pdf = pdf_hash
        st.session_state.retriever = retriever
        st.session_state.llm = llm
        st.session_state.messages = []
        st.session_state.chunk_count = len(chunks)

    st.success(
        f"✅ **{uploaded_file.name}** processed — {st.session_state.chunk_count} chunks indexed!"
    )
else:
    # Re-create LLM on provider / key change
    if "llm" not in st.session_state:
        st.session_state.llm = get_llm(provider, api_key)

# ── Status bar ───────────────────────────────────────────────────────
st.markdown(
    f"""
<div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:.5rem;">
    <div class="status-pill">
        <span class="status-dot"></span>
        <span>📄 {uploaded_file.name}</span>
    </div>
    <div class="status-pill">
        <span>🧩 {st.session_state.chunk_count} chunks</span>
    </div>
    <div class="status-pill">
        <span>🤖 {provider}</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.divider()

# ── Chat history ─────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Source passages"):
                for src in msg["sources"]:
                    page = src.metadata.get("page", "?")
                    st.markdown(f"**[Page {page}]** {src.page_content[:300]}…")

# ── User input ───────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your PDF…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = ask_question(
                    llm=st.session_state.llm,
                    retriever=st.session_state.retriever,
                    question=prompt,
                    chat_history=st.session_state.messages,
                )
                answer = result["answer"]
                sources = result.get("source_documents", [])
            except Exception as e:
                answer = f"⚠️ Error: {e}"
                sources = []

        st.markdown(answer)

        if sources:
            with st.expander("📚 Source passages"):
                for src in sources:
                    page = src.metadata.get("page", "?")
                    st.markdown(f"**[Page {page}]** {src.page_content[:300]}…")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
