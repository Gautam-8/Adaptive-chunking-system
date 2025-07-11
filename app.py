# app.py

import streamlit as st
from rag_pipeline import RAGPipeline
import tempfile
import os

# Initialize RAG pipeline
rag = RAGPipeline(persist_directory="vector_db")

# Set page title
st.set_page_config(page_title="📄 Adaptive Document Chunker", layout="wide")
st.title("📄 Adaptive Document Chunking & Retrieval")

# File upload
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt", "md", "html"])

if uploaded_file:
    with st.spinner("Processing document..."):
        # Save to temp file
        temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Ingest and chunk
        result = rag.ingest(temp_path)
        st.success(f"✅ Processed '{result['file']}' as type: **{result['doc_type']}** with {result['total_chunks']} chunks.")

        # Show all chunks
        with st.expander("🧩 View All Chunks", expanded=True):
            docs = rag.load_file(temp_path)
            for doc in docs:
                doc_type = rag.classify(doc.page_content)
                chunker = rag.get_chunker(doc_type)
                chunks = chunker.split_text(doc.page_content)

                for i, chunk in enumerate(chunks):
                    st.markdown(f"**Chunk {i+1}**")
                    st.code(chunk, language="markdown")
                    st.markdown("---")
                    vector_embedding = rag.embedder.embed_query(chunk)
                    st.write(vector_embedding)

# Query interface
st.markdown("### 🔍 Ask a Question")
user_query = st.text_input("Enter your question about the uploaded document:")

if user_query:
    with st.spinner("Searching..."):
        results = rag.query(user_query)
        for i, res in enumerate(results):
            st.markdown(f"**Result {i+1} (Score: {res['score']:.2f})**")
            st.markdown(f"Type: `{res['metadata'].get('type', 'unknown')}`")
            st.code(res['content'], language="markdown")
            st.markdown("---")
