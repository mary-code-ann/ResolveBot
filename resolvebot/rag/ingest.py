"""Load policy markdown, chunk it, and persist a local Chroma index."""

from __future__ import annotations

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from resolvebot.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    POLICIES_DIR,
    ensure_runtime_dirs,
)

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the local embedding model once per process."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _policy_files() -> list[Path]:
    return sorted(POLICIES_DIR.glob("*.md"))


def ingest(rebuild: bool = False) -> dict:
    """Build or rebuild the Chroma collection from data/policies/*.md."""
    ensure_runtime_dirs()
    files = _policy_files()
    if not files:
        raise FileNotFoundError(f"No markdown policies found in {POLICIES_DIR}")

    if rebuild and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    loader = DirectoryLoader(
        str(POLICIES_DIR),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    for doc in documents:
        source = Path(doc.metadata.get("source", "")).name
        doc.metadata["source"] = source

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(documents)

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_name=CHROMA_COLLECTION,
        collection_metadata={"hnsw:space": "cosine"},
    )
    return {
        "files": [path.name for path in files],
        "chunks": len(chunks),
        "persist_directory": str(CHROMA_DIR),
    }


def index_exists() -> bool:
    sqlite = CHROMA_DIR / "chroma.sqlite3"
    return sqlite.exists()
