"""
Week 5 – Task 2: CLIP Embeddings for Style References
=======================================================
Uses CLIP's text encoder to embed every fashion knowledge-base entry
into a 512-dim vector. These vectors power semantic similarity search.

How it works:
  - Each entry is converted into a rich text description
  - CLIP's text encoder converts text → 512-dim float vector
  - Vectors are saved to embeddings/fashion_embeddings.json

Note: If you have fashion images, swap clip.encode_image() in instead.
"""

import json
import os
import numpy as np
from typing import List, Dict, Any


# ─────────────────────────────────────────────
# TEXT → EMBEDDING TEMPLATE
# ─────────────────────────────────────────────

def entry_to_text(entry: Dict[str, Any]) -> str:
    """
    Convert a knowledge-base entry dict into a rich descriptive sentence
    that CLIP's text encoder will embed meaningfully.
    """
    t = entry["type"]
    name = entry["name"]
    desc = entry.get("description", "")

    if t == "style":
        colors = ", ".join(entry.get("colors", []))
        fabrics = ", ".join(entry.get("fabrics", []))
        mood = ", ".join(entry.get("mood", []))
        occasions = ", ".join(entry.get("occasions", []))
        tags = ", ".join(entry.get("tags", []))
        return (
            f"Fashion style: {name}. {desc} "
            f"Colors: {colors}. Fabrics: {fabrics}. "
            f"Mood: {mood}. Occasions: {occasions}. Keywords: {tags}."
        )

    elif t == "trend":
        season = entry.get("season", "")
        key_pieces = ", ".join(entry.get("key_pieces", []))
        mood = ", ".join(entry.get("mood", []))
        tags = ", ".join(entry.get("tags", []))
        return (
            f"Fashion trend ({season}): {name}. {desc} "
            f"Key pieces: {key_pieces}. Mood: {mood}. Keywords: {tags}."
        )

    elif t == "design_element":
        techniques = ", ".join(entry.get("techniques", []))
        fabrics = ", ".join(entry.get("suitable_fabrics", []))
        tags = ", ".join(entry.get("tags", []))
        return (
            f"Design element: {name}. {desc} "
            f"Techniques: {techniques}. Fabrics: {fabrics}. Keywords: {tags}."
        )

    elif t == "occasion":
        key_pieces = ", ".join(entry.get("key_pieces", []))
        tip = entry.get("style_tip", "")
        tags = ", ".join(entry.get("tags", []))
        return (
            f"Occasion: {name}. {desc} "
            f"Key pieces: {key_pieces}. Tip: {tip}. Keywords: {tags}."
        )

    return f"{name}. {desc}"


# ─────────────────────────────────────────────
# CLIP EMBEDDING PIPELINE
# ─────────────────────────────────────────────

def generate_clip_embeddings(
    db_path: str = "data/fashion_knowledge_base.json",
    output_path: str = "embeddings/fashion_embeddings.json",
    use_mock: bool = False          # set True if CLIP not installed yet
):
    """
    Main function: loads the DB, encodes each entry with CLIP, saves results.

    Args:
        db_path    : Path to fashion_knowledge_base.json
        output_path: Where to save embeddings JSON
        use_mock   : If True, generates random 512-dim vectors (for testing)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load knowledge base
    with open(db_path) as f:
        db = json.load(f)
    entries = db["entries"]
    print(f"📂 Loaded {len(entries)} entries from knowledge base")

    # Build text descriptions
    texts = [entry_to_text(e) for e in entries]

    if use_mock:
        # ── MOCK MODE (no GPU/CLIP needed) ──────────────────────────────
        print("⚠️  Mock mode: generating random 512-dim vectors for testing")
        np.random.seed(42)
        embeddings = np.random.randn(len(texts), 512).astype(np.float32)
        # L2-normalise (CLIP does this too)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
    else:
        # ── REAL CLIP MODE ───────────────────────────────────────────────
        try:
            import clip
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🔧 Loading CLIP model on {device}...")
            model, preprocess = clip.load("ViT-B/32", device=device)

            embeddings_list = []
            batch_size = 32
            print(f"🔢 Encoding {len(texts)} entries in batches of {batch_size}...")

            for i in range(0, len(texts), batch_size):
                batch = texts[i: i + batch_size]
                tokens = clip.tokenize(batch, truncate=True).to(device)
                with torch.no_grad():
                    feats = model.encode_text(tokens)
                    feats = feats / feats.norm(dim=-1, keepdim=True)   # normalise
                embeddings_list.append(feats.cpu().numpy())
                print(f"   Encoded {min(i + batch_size, len(texts))}/{len(texts)}")

            embeddings = np.vstack(embeddings_list).astype(np.float32)

        except ImportError:
            print("⚠️  CLIP not found — falling back to mock embeddings.")
            print("    Install with: pip install git+https://github.com/openai/CLIP.git")
            np.random.seed(42)
            embeddings = np.random.randn(len(texts), 512).astype(np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

    # ── SAVE ─────────────────────────────────────────────────────────────
    output = {
        "metadata": {
            "model": "ViT-B/32" if not use_mock else "mock",
            "embedding_dim": 512,
            "total": len(entries),
        },
        "embeddings": [
            {
                "id": entries[i]["id"],
                "type": entries[i]["type"],
                "name": entries[i]["name"],
                "text": texts[i],
                "vector": embeddings[i].tolist()
            }
            for i in range(len(entries))
        ]
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Embeddings saved → {output_path}")
    print(f"   Shape  : ({len(entries)}, 512)")
    print(f"   Model  : {output['metadata']['model']}")
    return output


# ─────────────────────────────────────────────
# COSINE SIMILARITY UTILITY (for quick testing)
# ─────────────────────────────────────────────

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    a, b = np.array(vec_a), np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def quick_similarity_test(embeddings_path: str = "embeddings/fashion_embeddings.json"):
    """Sanity-check: prints top-3 most similar pairs in the embedding space."""
    with open(embeddings_path) as f:
        data = json.load(f)

    items = data["embeddings"]
    print("\n🔍 Top similar pairs (cosine similarity):")
    scores = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            sim = cosine_similarity(items[i]["vector"], items[j]["vector"])
            scores.append((sim, items[i]["name"], items[j]["name"]))
    scores.sort(reverse=True)
    for sim, a, b in scores[:5]:
        print(f"   {sim:.3f}  |  {a}  ↔  {b}")


if __name__ == "__main__":
    result = generate_clip_embeddings(use_mock=True)   # change to False with real CLIP
    quick_similarity_test()
