"""
Week 5 – Task 3: ChromaDB Setup & Style Intelligence Retrieval
================================================================
Stores CLIP embeddings in ChromaDB and provides a retrieval interface
for semantic fashion queries.

Pipeline:
  1. Load embeddings from task2
  2. Create / reset ChromaDB collection
  3. Upsert all vectors with metadata
  4. Query by natural-language text or pre-computed vector
"""

import json
import os
import numpy as np
from typing import List, Dict, Any, Optional


# ─────────────────────────────────────────────
# CHROMADB CLIENT
# ─────────────────────────────────────────────

def get_chroma_client(persist_dir: str = "database/chromadb"):
    """Returns a persistent ChromaDB client."""
    try:
        import chromadb
        os.makedirs(persist_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_dir)
        print(f"✅ ChromaDB client ready  →  {persist_dir}")
        return client
    except ImportError:
        raise ImportError("Install ChromaDB: pip install chromadb")


# ─────────────────────────────────────────────
# LOAD EMBEDDINGS INTO CHROMADB
# ─────────────────────────────────────────────

def build_chroma_collection(
    embeddings_path: str = "embeddings/fashion_embeddings.json",
    persist_dir: str = "database/chromadb",
    collection_name: str = "fashion_styles",
    reset: bool = False
):
    """
    Upserts all fashion embeddings into a ChromaDB collection.

    Args:
        embeddings_path : Output of task2_clip_embeddings.py
        persist_dir     : Local directory for the persistent DB
        collection_name : ChromaDB collection name
        reset           : If True, deletes the existing collection first
    """
    client = get_chroma_client(persist_dir)

    # Reset if requested
    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"🗑️  Deleted existing collection: {collection_name}")
        except Exception:
            pass

    # Get or create collection
    # NOTE: ChromaDB handles L2 distance natively; cosine is set via metadata
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}      # cosine similarity
    )
    print(f"📦 Collection: '{collection_name}'  (existing docs: {collection.count()})")

    # Load embeddings
    with open(embeddings_path) as f:
        data = json.load(f)
    items = data["embeddings"]
    print(f"📂 Loading {len(items)} embeddings...")

    # Batch upsert
    ids        = [item["id"]     for item in items]
    vectors    = [item["vector"] for item in items]
    documents  = [item["text"]   for item in items]
    metadatas  = [
        {
            "name": item["name"],
            "type": item["type"],
        }
        for item in items
    ]

    collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"✅ Upserted {len(ids)} entries  →  total in DB: {collection.count()}")
    return collection


# ─────────────────────────────────────────────
# QUERY ENGINE
# ─────────────────────────────────────────────

class FashionRetriever:
    """
    Semantic retrieval over the ChromaDB fashion collection.
    Supports:
      - Text queries   → encoded with CLIP (or mock) then searched
      - Vector queries → raw 512-dim vector
      - Type filtering → restrict to 'style', 'trend', 'design_element', 'occasion'
    """

    def __init__(
        self,
        persist_dir: str = "database/chromadb",
        collection_name: str = "fashion_styles",
        embeddings_path: str = "embeddings/fashion_embeddings.json",
    ):
        import chromadb
        client = chromadb.PersistentClient(path=persist_dir)
        self.collection = client.get_collection(collection_name)

        # Keep in-memory index for query-by-name lookups
        with open(embeddings_path) as f:
            data = json.load(f)
        self._index = {item["id"]: item for item in data["embeddings"]}
        print(f"🔍 FashionRetriever ready  ({self.collection.count()} entries)")

    def _encode_text(self, text: str) -> List[float]:
        """Encode a text query using CLIP (falls back to keyword mock)."""
        try:
            import clip, torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if not hasattr(self, "_clip_model"):
                self._clip_model, _ = clip.load("ViT-B/32", device=device)
                self._device = device
            tokens = clip.tokenize([text], truncate=True).to(self._device)
            with torch.no_grad():
                feat = self._clip_model.encode_text(tokens)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.cpu().numpy()[0].tolist()
        except ImportError:
            # Mock: return a random-but-deterministic vector based on text hash
            np.random.seed(abs(hash(text)) % (2**31))
            vec = np.random.randn(512).astype(np.float32)
            return (vec / np.linalg.norm(vec)).tolist()

    def query(
        self,
        text: str,
        n_results: int = 5,
        filter_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search by text.

        Args:
            text        : Natural-language fashion query
            n_results   : Number of results to return
            filter_type : One of 'style', 'trend', 'design_element', 'occasion'

        Returns:
            List of result dicts with id, name, type, score, description
        """
        query_vec = self._encode_text(text)
        where_clause = {"type": filter_type} if filter_type else None

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=n_results,
            where=where_clause,
            include=["metadatas", "distances", "documents"],
        )

        formatted = []
        for i in range(len(results["ids"][0])):
            entry_id   = results["ids"][0][i]
            meta       = results["metadatas"][0][i]
            distance   = results["distances"][0][i]
            document   = results["documents"][0][i]
            similarity = 1 - distance           # cosine distance → similarity

            formatted.append({
                "id"         : entry_id,
                "name"       : meta["name"],
                "type"       : meta["type"],
                "similarity" : round(similarity, 4),
                "description": document[:200] + "..." if len(document) > 200 else document,
            })

        return formatted

    def query_by_vector(
        self,
        vector: List[float],
        n_results: int = 5,
        filter_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search using a pre-computed CLIP vector (e.g. from an image)."""
        where_clause = {"type": filter_type} if filter_type else None
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=n_results,
            where=where_clause,
            include=["metadatas", "distances", "documents"],
        )
        return [
            {
                "id"        : results["ids"][0][i],
                "name"      : results["metadatas"][0][i]["name"],
                "type"      : results["metadatas"][0][i]["type"],
                "similarity": round(1 - results["distances"][0][i], 4),
            }
            for i in range(len(results["ids"][0]))
        ]

    def pretty_print(self, results: List[Dict], query: str = ""):
        if query:
            print(f"\n🔎 Query: \"{query}\"")
        print(f"{'Rank':<5} {'Score':<8} {'Type':<16} {'Name'}")
        print("─" * 65)
        for rank, r in enumerate(results, 1):
            print(f"{rank:<5} {r['similarity']:<8.4f} {r['type']:<16} {r['name']}")
            if "description" in r:
                print(f"      ↳ {r['description'][:90]}...")
        print()


# ─────────────────────────────────────────────
# MAIN – build DB and run sample queries
# ─────────────────────────────────────────────

def run_demo():
    """End-to-end demo: build collection → run 4 sample queries."""
    print("=" * 60)
    print("  Week 5 · Task 3 — ChromaDB Fashion Retrieval")
    print("=" * 60)

    # Build DB
    build_chroma_collection(reset=True)

    # Query
    retriever = FashionRetriever()

    demo_queries = [
        ("elegant evening look with drama and luxury", None),
        ("summer festival free spirit outfit", "style"),
        ("2024 minimalist trend for office", "trend"),
        ("romantic sheer fabric for spring", "design_element"),
    ]

    for q, ftype in demo_queries:
        results = retriever.query(q, n_results=3, filter_type=ftype)
        retriever.pretty_print(results, query=q)


if __name__ == "__main__":
    run_demo()
