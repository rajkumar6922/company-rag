"""In-memory Chroma vector store with one collection per user session."""

import hashlib

import chromadb
from chromadb.utils import embedding_functions

# --- Embedding function -----------------------------------------------------
# The only place the embedding model is chosen. Chroma's default runs
# all-MiniLM-L6-v2 locally (ONNX, no API key). To switch to OpenAI later:
#   EMBEDDING_FUNCTION = embedding_functions.OpenAIEmbeddingFunction(
#       api_key=..., model_name="text-embedding-3-small")
# Existing collections must be rebuilt after a switch, because vectors from
# different models can't be compared.
EMBEDDING_FUNCTION = embedding_functions.DefaultEmbeddingFunction()

# --- Client -----------------------------------------------------------------
# EphemeralClient keeps everything in RAM: nothing is written to disk, and all
# data is gone when the process stops.
_client = chromadb.EphemeralClient()


def _collection_name(session_id: str) -> str:
    # Chroma only allows certain characters and lengths in collection names,
    # so hash the session id into a safe, fixed-length name. Different session
    # ids give different names, which keeps sessions apart.
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32]
    return f"session-{digest}"


def create_collection(session_id: str):
    """Return the collection for this session, creating it if needed."""
    return _client.get_or_create_collection(
        name=_collection_name(session_id),
        embedding_function=EMBEDDING_FUNCTION,
        # Cosine distance: 0 = same meaning, larger = less related.
        configuration={"hnsw": {"space": "cosine"}},
    )


def delete_collection(session_id: str) -> None:
    """Drop this session's collection to free memory (no error if missing)."""
    try:
        _client.delete_collection(_collection_name(session_id))
    except Exception:
        pass


def add_document(collection, filename: str, chunks: list[str]) -> None:
    """Store every chunk of one file, with its source and position."""
    if not chunks:
        return

    # If the same file is uploaded again, drop its old chunks first so they
    # aren't duplicated or left behind.
    collection.delete(where={"source": filename})

    # Ids only need to be unique within this session's collection.
    ids = [f"{filename}::{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

    # Chroma embeds the documents with EMBEDDING_FUNCTION on add.
    collection.add(ids=ids, documents=chunks, metadatas=metadatas)


def search(collection, question: str, k: int = 5) -> list[dict]:
    """Return the k chunks closest to the question, nearest first."""
    # Don't ask for more results than the collection holds.
    n = min(k, collection.count())
    if n == 0:
        return []

    res = collection.query(query_texts=[question], n_results=n)

    # query() returns one list per question; we sent one question, so take [0].
    return [
        {
            "text": doc,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "distance": dist,
        }
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]
