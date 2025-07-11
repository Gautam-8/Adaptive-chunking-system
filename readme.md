# 🧠 Adaptive Document Chunking & Retrieval System

This project is an AI-powered document processing and semantic retrieval system designed for enterprise knowledge bases. It automatically detects the type of document, applies intelligent chunking strategies, embeds the chunks, and stores them for fast and accurate semantic search.

---

## 🚀 Features

- 📂 Upload and process PDFs, DOCX, TXT, Markdown, or HTML files
- 🧠 Classifies document types (e.g., tutorial, support ticket, API reference)
- ✂️ Uses adaptive chunking strategies based on content type
- 🧬 Embeds text using `all-MiniLM-L6-v2` (Sentence Transformers)
- 🗃️ Stores vectors in a local ChromaDB
- 🔍 Search using natural language queries
- 🧩 View how each document was chunked
- 🧪 Evaluate retrieval accuracy using embedding similarity (Precision, Recall)

---

## 🧰 Tech Stack

- [LangChain](https://docs.langchain.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [ChromaDB](https://docs.trychroma.com/)
- [Streamlit](https://streamlit.io/)

---

## ⚙️ Installation

1. Clone the repo
2. Create a virtual environment (optional but recommended)
3. Install dependencies

```bash
pip install -r requirements.txt
```
