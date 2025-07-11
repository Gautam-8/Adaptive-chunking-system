# rag_pipeline.py

import os
import re
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader
)


class RAGPipeline:
    def __init__(self, persist_directory: str = "vector_db"):
        # Initialize classifier rules
        self.rules = {
            "support_ticket": ["error", "issue", "expected", "actual", "resolved", "ticket"],
            "technical_doc": ["architecture", "deployment", "configuration", "design"],
            "api_reference": ["endpoint", "request", "response", "parameters", "curl", "status code"],
            "policy_doc": ["policy", "compliance", "regulation", "must", "shall", "adherence"],
            "tutorial": ["step", "how to", "guide", "introduction", "example", "walkthrough"]
        }

        # Set up embedder and vectorstore
        self.embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedder
        )

        self.chunker_config = {
            "tutorial": {"separators": ["\n\n", "\n", " "], "chunk_size": 500, "chunk_overlap": 50},
            "technical_doc": {"separators": ["\n\n", "\n", " "], "chunk_size": 500, "chunk_overlap": 50},
            "api_reference": {"separators": ["```", "\n\n", "\n", " "], "chunk_size": 700, "chunk_overlap": 100},
            "policy_doc": {"separators": [r"\n\d+(\.\d+)*\s", "\n\n", "\n"], "chunk_size": 800, "chunk_overlap": 100},
            "support_ticket": {"separators": [r"(?i)(step \d+|expected|actual|resolution|fix)", "\n"], "chunk_size": 400, "chunk_overlap": 50},
        }

    def classify(self, content: str) -> str:
        content_lower = content.lower()
        scores = {doc_type: 0 for doc_type in self.rules}

        for doc_type, keywords in self.rules.items():
            for kw in keywords:
                if re.search(rf'\b{kw}\b', content_lower):
                    scores[doc_type] += 1

        return max(scores.items(), key=lambda item: item[1])[0]


    def get_chunker(self, doc_type: str):
        cfg = self.chunker_config.get(doc_type, self.chunker_config["tutorial"])
        return RecursiveCharacterTextSplitter(
            separators=cfg["separators"],
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"]
        )

    def load_file(self, file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext == ".html":
            loader = UnstructuredHTMLLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        return loader.load()

    def ingest(self, file_path: str) -> Dict:
        docs = self.load_file(file_path)

        all_chunks = []
        for doc in docs:
            text = doc.page_content
            metadata = doc.metadata or {}

            doc_type = self.classify(text)
            chunker = self.get_chunker(doc_type)
            chunks = chunker.split_text(text)

            # Add metadata to each chunk
            chunk_docs = [
                Document(page_content=chunk, metadata={"type": doc_type, **metadata})
                for chunk in chunks
            ]

            all_chunks.extend(chunk_docs)

        self.vectorstore.add_documents(all_chunks)
        self.vectorstore.persist()

        return {
            "file": os.path.basename(file_path),
            "doc_type": doc_type,
            "total_chunks": len(all_chunks)
        }

    def query(self, user_query: str, top_k: int = 5) -> List[Dict]:
        results = self.vectorstore.similarity_search_with_score(user_query, k=top_k)
        return [
            {
                "content": doc.page_content,
                "score": score,
                "metadata": doc.metadata
            }
            for doc, score in results
        ]
