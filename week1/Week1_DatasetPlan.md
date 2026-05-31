# Week 1: Dataset Curation Plan
**Intern:** Anjal Suryawanshi
**Project:** AI-Powered Fashion Design Assistant with Generative Models

## Selected Datasets

### DeepFashion
- 800,000+ fashion images
- Clothing categories and attributes
- Landmark annotations
- Useful for retrieval and attribute prediction

### FashionGen
- 293,000+ high-resolution images
- Professional fashion descriptions
- Useful for text-to-image generation

## Metadata Fields
- image_id
- category
- color
- pattern
- sleeves
- neckline
- fabric_type
- season
- description



### Dataset 1: DeepFashion (Primary Dataset)
**Purpose:** Attribute prediction, clothing retrieval, landmark detection

| Property | Details |
|----------|---------|
| **Size** | 800,000+ diverse fashion images |
| **Categories** | 50 clothing types (dress, t-shirt, pants, jacket, shoes) |
| **Attributes** | 1,000 descriptive attributes (color, pattern, sleeves, neckline, length, fit) |
| **Annotations** | Bounding boxes + 14 clothing landmarks (left/right shoulder, neckline center, waist, hem) |
| **Cross-domain pairs** | 300,000+ image pairs (shop photo → consumer photo) |
| **Format** | JPEG images + JSON annotations |
| **Access** | Request form: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html |

**Why DeepFashion?**
- Rich attribute annotations (1,000 attributes) perfect for style-based retrieval
- Landmark annotations enable pose-controlled generation (ControlNet)
- Consumer photos help AI understand real-world fashion (not just shop photos)
- Cross-domain pairs enable consumer-to-shop retrieval use case

---

### Dataset 2: FashionGen (Text-to-Image Dataset)
**Purpose:** Text-to-image generation with Stable Diffusion

| Property | Details |
|----------|---------|
| **Size** | 293,008 high-definition images (1360×1360 pixels) |
| **Training/Validation** | 260,490 train + 32,528 validation images |
| **Classes** | 48 main classes, 121 subclasses |
| **Text Descriptions** | Professional stylist captions for each item |
| **Format** | HDF5 files (also available as 256×256 PNG) |
| **Access** | GitHub: https://github.com/menardai/FashionGenAttnGAN |

Why FashionGen?
- High-resolution images (1360×1360) ideal for Stable Diffusion training
- Professional stylist captions provide rich text-image pairs
- Optimized for text-to-image generation challenge
- Multiple angles per item enable pose variety

fashion-dataset/
├── deepfashion/images/
├── deepfashion/annotations/
├── fashiongen/images_256/
├── fashiongen/descriptions.txt
└── curated/train_val_test/

## Metadata Schema Design (14 Attributes)

| Attribute | Type | Example Values |
|-----------|------|----------------|
| image_id | string | df_00001, fg_00045 |
| category | categorical | dress, t-shirt, pants, jacket |
| subcategory | categorical | maxi dress, crew neck, skinny jeans |
| color | multi-label | navy blue, white, black |
| pattern | categorical | solid, floral, stripes, polka dot |
| sleeves | categorical | long sleeve, short sleeve, sleeveless |
| neckline | categorical | v-neck, crew, strapless, boat neck |
| length | categorical | ankle-length, knee-length, regular |
| fit | categorical | slim fit, regular, relaxed, fitted |
| fabric_type | categorical | cotton, silk, denim, polyester |
| occasion | categorical | casual, formal, athletic |
| season | categorical | summer, winter, all-season |
| image_path | string | /path/to/image.jpg |
| description | text | "Elegant navy blue maxi dress with v-neckline" |

