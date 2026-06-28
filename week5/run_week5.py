"""
Week 5 – Master Run Script
===========================
Runs the complete Week 5 pipeline in sequence:
  Step 1 → Build Fashion Knowledge Database
  Step 2 → Generate CLIP Embeddings
  Step 3 → Load into ChromaDB & Test Retrieval
  Step 4 → Generate Personalized Recommendations

Run from the week5_rag/ directory:
  python run_week5.py
"""

import os
import sys

def step(n, title):
    print(f"\n{'═'*60}")
    print(f"  STEP {n}: {title}")
    print(f"{'═'*60}")

# ── STEP 1 ───────────────────────────────────────────────────
step(1, "Build Fashion Knowledge Database")
from task1_fashion_database import build_fashion_database
db = build_fashion_database("data/fashion_knowledge_base.json")

# ── STEP 2 ───────────────────────────────────────────────────
step(2, "Generate CLIP Embeddings")
from task2_clip_embeddings import generate_clip_embeddings, quick_similarity_test
generate_clip_embeddings(
    db_path="data/fashion_knowledge_base.json",
    output_path="embeddings/fashion_embeddings.json",
    use_mock=True          # ← set False once CLIP is installed
)
quick_similarity_test("embeddings/fashion_embeddings.json")

# ── STEP 3 ───────────────────────────────────────────────────
step(3, "Load into ChromaDB & Run Sample Queries")
from task3_chromadb_retrieval import build_chroma_collection, FashionRetriever
build_chroma_collection(reset=True)
retriever = FashionRetriever()

sample_queries = [
    "romantic flowy look for a summer wedding",
    "bold maximalist trend 2024",
    "minimalist office outfit neutral tones",
]
for q in sample_queries:
    results = retriever.query(q, n_results=3)
    retriever.pretty_print(results, query=q)

# ── STEP 4 ───────────────────────────────────────────────────
step(4, "Personalized Recommendations (RAG)")
from task4_recommendations import FashionRecommendationEngine, DEMO_PROFILES
engine = FashionRecommendationEngine()
for profile in DEMO_PROFILES:
    print(f"\n{'─'*60}")
    rec = engine.recommend(profile)
    print(rec)

print("\n" + "═"*60)
print("  ✅  Week 5 pipeline complete!")
print("  📁  Outputs:")
print("       data/fashion_knowledge_base.json")
print("       embeddings/fashion_embeddings.json")
print("       database/chromadb/")
print("═"*60)
