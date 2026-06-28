# Week 5 – Intelligent Style Recommendation (RAG)
### Aarivya Labs Internship | AI-Powered Fashion Design Assistant

---

## 📌 Overview

This week implements a full **Retrieval-Augmented Generation (RAG)** pipeline for personalized fashion recommendations using CLIP embeddings and ChromaDB.

---

## 📁 Project Structure

```
week5_rag/
├── task1_fashion_database.py     # Task 1 – Fashion knowledge base
├── task2_clip_embeddings.py      # Task 2 – CLIP embeddings
├── task3_chromadb_retrieval.py   # Task 3 – ChromaDB store & retrieval
├── task4_recommendations.py      # Task 4 – RAG recommendation engine
├── run_week5.py                  # Master pipeline runner
├── requirements.txt
├── data/
│   └── fashion_knowledge_base.json
├── embeddings/
│   └── fashion_embeddings.json
└── database/
    └── chromadb/
```

---

## 🛠️ Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install CLIP (optional, for real embeddings)
pip install git+https://github.com/openai/CLIP.git
```

---

## ▶️ Run

```bash
# Run full pipeline at once
python run_week5.py

# Or run each task individually
python task1_fashion_database.py
python task2_clip_embeddings.py
python task3_chromadb_retrieval.py
python task4_recommendations.py
```

---

## ✅ Deliverables

| Task | File | Description |
|------|------|-------------|
| 1 | `task1_fashion_database.py` | 14-entry fashion knowledge base (styles, trends, elements, occasions) |
| 2 | `task2_clip_embeddings.py` | CLIP text encoder → 512-dim vectors, with mock fallback |
| 3 | `task3_chromadb_retrieval.py` | ChromaDB upsert + semantic query engine with type filtering |
| 4 | `task4_recommendations.py` | Full RAG pipeline with personalized outfit recommendations |

---

## 🔍 RAG Pipeline

```
User Profile (name, body type, occasion, mood, budget)
         │
         ▼
  Query Construction  ──→  "romantic wedding guest hourglass summer look"
         │
         ▼
  CLIP Text Encoder   ──→  512-dim embedding vector
         │
         ▼
  ChromaDB Retrieval  ──→  Top-K similar styles, trends, elements
         │
         ▼
  Recommendation Gen  ──→  Personalized outfit + styling tips
```

---

## 🧠 Key Concepts Demonstrated

- **CLIP embeddings** for semantic fashion understanding
- **ChromaDB** as a vector database with cosine similarity
- **Multi-query RAG** (separate retrievals per content type)
- **UserProfile → Query** translation for personalization
- **LLM integration hook** (supports Anthropic Claude via API key)

---

## 📊 Knowledge Base Stats

| Category | Entries |
|----------|---------|
| Styles | 6 (Minimalist, Bohemian, Streetwear, Haute Couture, Y2K, Dark Academia) |
| Trends | 4 (Quiet Luxury, Dopamine Dressing, Gorpcore, Sheer Elegance) |
| Design Elements | 3 (Draped Silhouette, Puff Sleeve, Asymmetric Hem) |
| Occasions | 2 (Office, Beach/Resort) |
| **Total** | **15** |
