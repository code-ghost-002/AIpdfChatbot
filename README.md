# 📄 AI PDF Chatbot

live demo - https://aipdfchatbot-kzizlgkqhkcnmnj9bfgsfd.streamlit.app/

> **Built an AI-powered PDF Question Answering system using LangChain, ChromaDB, and Streamlit — implementing Retrieval Augmented Generation (RAG).**

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-1C3C3C?logo=langchain&logoColor=white)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📤 **PDF Upload** | Drag-and-drop any PDF (up to 200 MB) |
| 🧩 **Smart Chunking** | Recursive text splitting for optimal retrieval accuracy |
| 🗄️ **Vector Database** | ChromaDB stores embeddings with persistent storage |
| 🤖 **RAG Responses** | LLM answers are grounded in your document content |
| 💬 **Conversational Memory** | Follow-up questions keep context from earlier turns |
| 📚 **Source Citations** | See exact passages the AI used to answer |
| 🔀 **Dual LLM Support** | Switch between **Groq** (free) and **OpenAI** |

---

## 🏗️ Architecture

```
User uploads PDF
       │
       ▼
┌──────────────┐
│  PyPDF Loader │  ── Extract text from each page
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ Recursive Text Split │  ── 1000-char chunks, 200-char overlap
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Sentence Embeddings │  ── all-MiniLM-L6-v2 (local, free)
└──────┬───────────────┘
       │
       ▼
┌──────────────┐
│   ChromaDB   │  ── Persistent vector store
└──────┬───────┘
       │
       ▼
┌────────────────────────────────┐
│ Conversational Retrieval Chain │
│   Retriever  ──►  LLM (Groq   │
│              or OpenAI)        │
└────────────────────────────────┘
       │
       ▼
   AI Answer + Source Passages
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/AIpdfchatbot.git
cd AIpdfchatbot

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy the example and fill in your key
cp .env.example .env
```

Edit `.env` and add your **Groq** or **OpenAI** API key:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxx
```

> 💡 **Groq is free** — get a key at [console.groq.com](https://console.groq.com)

### 3. Run

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** 🎉

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit |
| **Orchestration** | LangChain |
| **Vector Store** | ChromaDB |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **LLM** | Groq (`llama-3.3-70b`) / OpenAI (`gpt-4o-mini`) |
| **PDF Parsing** | PyPDF |

---

## 📁 Project Structure

```
AIpdfchatbot/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore
├── README.md
└── chroma_db/          # Auto-created — persistent vector store
```

---

## 📜 License

MIT — free to use, modify, and distribute.
