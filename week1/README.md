# Week 1 Submission - AI-Powered Fashion Design Assistant

## Deliverables
- Fashion workflow research
- Dataset curation plan
- Metadata schema
- Dataset curation pipeline

## Note
Week 1 focuses on research, schema design, and pipeline preparation.


## Week 1 Tasks Completed

# ✅ Task 1: Fashion Domain Research
**File:** Week1_FashionWorkflow.md

Studied fashion design workflows from sketch to garment:
- 8-stage design process (Concept → Sketching → Technical Drawing → Pattern Making → Fabric Selection → Sample Making → Fitting → Production)
- AI applications at each stage (Stable Diffusion, ControlNet, ChromaDB)
- Key insights for AI Fashion Design Assistant

# ✅ Task 2: Dataset Curation
**File:** Week1_DatasetPlan.md

Documented curation plan for:
- **DeepFashion:** 800,000+ images, 50 categories, 1,000 attributes, landmarks
- **FashionGen:** 293,008 HD images (1360×1360), professional stylist captions
- Curation strategy: 10,000–20,000 images, 70/15/15 train/val/test split

# ✅ Task 3: Metadata Schema
**File:** Week1_MetadataSchema.csv

Designed 14-attribute schema:
- image_id, category, subcategory, color, pattern, sleeves, neckline, length, fit, fabric_type, occasion, season, image_path, description
- 12 sample fashion items (dresses, tops, pants, jackets)

# ✅ Task 4: Data Pipeline
**File:** week1_curation_pipeline.py

Built Python pipeline with:
- DeepFashion JSON annotation processing
- FashionGen HDF5 image extraction
- Metadata CSV generation
- Train/val/test splitting (70/15/15)
- ChromaDB vector indexing (for Week 2)
# Week 1: Fashion Dataset Curation Pipeline

## How to Run

1. Install dependencies:
```bash
pip install pandas pillow numpy h5py chromadb sentence-transformers
```

2. Download datasets:
- DeepFashion: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
- FashionGen: git clone https://github.com/menardai/FashionGenAttnGAN

3. Run pipeline:
```bash
python week1_curation_pipeline.py
```

## Output Files
- `curated/metadata/deepfashion_metadata.csv`
- `curated/metadata/fashiongen_metadata.csv`
- `curated/metadata/combined_metadata.csv`
- `curated/metadata/train.csv`, `validation.csv`, `test.csv`
- `curated/week1_research_summary.md`
---

## Dataset Status

**Note:** Datasets will be downloaded in Week 2.
- DeepFashion: Request form at http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
- FashionGen: GitHub repo at https://github.com/menardai/FashionGenAttnGAN

---

## Tech Stack Alignment

This Week 1 deliverable aligns with the internship tech stack:
- **Stable Diffusion:** Dataset prepared for text-to-image training
- **CLIP:** Metadata schema supports CLIP-based retrieval
- **ControlNet:** DeepFashion landmarks enable pose control
- **LoRA:** Curated dataset for brand style fine-tuning (Week 4)
- **ChromaDB:** Pipeline includes vector indexing for RAG (Week 5)
- **Gradio:** Prepared for Creative Studio interface (Week 6)
