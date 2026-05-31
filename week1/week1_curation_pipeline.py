# Week 1 Dataset Curation Pipeline (Starter Template)
"""
Week 1: Fashion Dataset Curation Pipeline
This script curates DeepFashion and FashionGen datasets for AI fashion assistant.

"""

import os
import json
import pandas as pd
import h5py
from pathlib import Path
from PIL import Image
import numpy as np

# ==================== CONFIGURATION ====================
BASE_DIR = Path("fashion-dataset")
DEEPFASHION_DIR = BASE_DIR / "deepfashion"
FASHIONGEN_DIR = BASE_DIR / "fashiongen"
CURATED_DIR = BASE_DIR / "curated"

# Create directories
for d in [DEEPFASHION_DIR, FASHIONGEN_DIR, CURATED_DIR]:
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "metadata").mkdir(parents=True, exist_ok=True)

# Categories to focus on (Week 2-4 implementation)
CATEGORIES_OF_INTEREST = ["dress", "t-shirt", "top", "pants", "jacket", "coat"]

# ==================== 1. PROCESS DEEPFASHION ====================
def process_deepfashion():
    """Load DeepFashion JSON annotations and create metadata CSV"""
    print("Processing DeepFashion dataset...")
    
    annotations_path = DEEPFASHION_DIR / "DeepFashion_annotations.json"
    
    if not annotations_path.exists():
        print("⚠️ DeepFashion annotations not found.")
        print("Download from: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html")
        return None
    
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    metadata = []
    for img_id, data in annotations.items():
        meta = {
            "image_id": img_id,
            "category": data.get("category", ""),
            "color": ", ".join(data.get("colors", [])),
            "pattern": data.get("pattern", ""),
            "sleeves": data.get("sleeves", ""),
            "neckline": data.get("neckline", ""),
            "length": data.get("length", ""),
            "fit": data.get("fit", ""),
            "image_path": str(DEEPFASHION_DIR / "images" / img_id)
        }
        metadata.append(meta)
    
    df = pd.DataFrame(metadata)
    df_filtered = df[df["category"].isin(CATEGORIES_OF_INTEREST)]
    
    df_filtered.to_csv(CURATED_DIR / "metadata" / "deepfashion_metadata.csv", index=False)
    print(f"✓ Curated {len(df_filtered)} DeepFashion images")
    
    return df_filtered

# ==================== 2. PROCESS FASHIONGEN ====================
def process_fashiongen():
    """Load FashionGen HDF5 and extract images + captions"""
    print("\nProcessing FashionGen dataset...")
    
    h5_path = FASHIONGEN_DIR / "fashiongen_256_256_train.h5"
    
    if not h5_path.exists():
        print("⚠️ FashionGen HDF5 not found.")
        print("Download from: https://github.com/menardai/FashionGenAttnGAN")
        return None
    
    images_list = []
    
    with h5py.File(h5_path, 'r') as f:
        images = f["images"][:1000]
        descriptions = f["input_concat_description"][:1000]
        
        for i, (img, desc) in enumerate(zip(images, descriptions)):
            img_pil = Image.fromarray(img)
            img_path = CURATED_DIR / "images" / f"fashiongen_{i:04d}.png"
            img_pil.save(img_path)
            
            images_list.append({
                "image_id": f"fashiongen_{i:04d}",
                "image_path": str(img_path),
                "description": desc.decode('utf-8') if isinstance(desc, bytes) else desc,
                "category": "fashion",
                "color": "", "pattern": "", "sleeves": "", 
                "neckline": "", "length": "", "fit": ""
            })
    
    df = pd.DataFrame(images_list)
    df.to_csv(CURATED_DIR / "metadata" / "fashiongen_metadata.csv", index=False)
    print(f"✓ Curated {len(df)} FashionGen images")
    
    return df

# ==================== 3. CREATE COMBINED DATASET ====================
def create_combined_dataset():
    """Merge DeepFashion + FashionGen metadata"""
    print("\nCreating combined dataset...")
    
    deepfashion_df = pd.read_csv(CURATED_DIR / "metadata" / "deepfashion_metadata.csv")
    fashiongen_df = pd.read_csv(CURATED_DIR / "metadata" / "fashiongen_metadata.csv")
    
    combined = pd.concat([deepfashion_df, fashiongen_df], ignore_index=True)
    combined.to_csv(CURATED_DIR / "metadata" / "combined_metadata.csv", index=False)
    
    print(f"✓ Combined dataset: {len(combined)} total images")
    return combined

# ==================== 4. CREATE TRAIN/VAL/TEST SPLITS ====================
def create_splits(df):
    """Create 70/15/15 train/val/test split"""
    print("\nCreating train/val/test splits...")
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]
    
    train_df.to_csv(CURATED_DIR / "metadata" / "train.csv", index=False)
    val_df.to_csv(CURATED_DIR / "metadata" / "validation.csv", index=False)
    test_df.to_csv(CURATED_DIR / "metadata" / "test.csv", index=False)
    
    print(f"✓ Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ==================== 5. BUILD CHROMADB INDEX ====================
def build_chromadb_index():
    """Create vector index for style-based retrieval"""
    print("\nBuilding ChromaDB vector index...")
    
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        
        chroma_client = chromadb.PersistentClient(path="fashion-dataset/chromadb")
        collection = chroma_client.create_collection("fashion_images")
        model = SentenceTransformer("clip-ViT-B-32")
        
        metadata_df = pd.read_csv(CURATED_DIR / "metadata" / "combined_metadata.csv")
        
        for _, row in metadata_df.iterrows():
            embedding = model.encode([row["category"], row["color"], row["pattern"]])[0]
            collection.add(
                embeddings=[embedding.tolist()],
                documents=[row["category"]],
                metadatas=[{"color": row["color"], "image_path": row["image_path"]}],
                ids=[row["image_id"]]
            )
        
        print(f"✓ ChromaDB index created with {len(metadata_df)} images")
    except ImportError:
        print("⚠️ Install chromadb and sentence-transformers to use this feature")

# ==================== RUN PIPELINE ====================
if __name__ == "__main__":
    print("=" * 60)
    print("WEEK 1: FASHION DATASET CURATION PIPELINE")
    print("Intern: Anjal Suryawanshi")
    print("=" * 60)
    
    deepfashion_df = process_deepfashion()
    fashiongen_df = process_fashiongen()
    
    if deepfashion_df is not None and fashiongen_df is not None:
        combined_df = create_combined_dataset()
        create_splits(combined_df)
        build_chromadb_index()
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE! ✓")
    print("=" * 60)


