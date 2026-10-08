"""Query the persisted Chroma policy index."""

from __future__ import annotations

from langchain_chroma import Chroma

from resolvebot.config import CHROMA_COLLECTION, CHROMA_DIR, RETRIEVAL_K
from resolvebot.rag.ingest import get_embeddings, index_exists, ingest


def get_vectorstore() -> Chroma:
    if not index_exists():
        ingest(rebuild=False)
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
        collection_name=CHROMA_COLLECTION,
    )


def retrieve_policy(query: str, k: int = RETRIEVAL_K) -> tuple[list[dict], float]:
    """Return ranked chunks and the best cosine similarity (1 - distance)."""
    store = get_vectorstore()
    pairs = store.similarity_search_with_score(query, k=k)
    docs: list[dict] = []
    best = 0.0
    for document, distance in pairs:
        # Cosine space: Chroma distance is 1 - cosine_similarity for some
        # builds, or raw L2. With normalize_embeddings + cosine metadata,
        # distance is typically in [0, 2]. Convert to similarity.
        similarity = max(0.0, 1.0 - float(distance))
        best = max(best, similarity)
        docs.append(
            {
                "content": document.page_content,
                "source": document.metadata.get("source", "unknown"),
                "score": round(similarity, 4),
            }
        )
    return docs, round(best, 4)
