<div align="center">

# 🧠 BrainScan AI
### Deep Learning-Based Brain Tumor Detection, Segmentation & Diagnostic Support System

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3-000000?style=for-the-badge&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-blue?style=for-the-badge)

**PCCOE MDM Capstone Project 2025–26**
*Utkarsha Bisen · Mitali Brahmankar · Sneha Iyer*
*Guide: Mrs. Swati Chandurkar*

[Live Demo](#setup) · [Architecture](#system-architecture) · [Results](#results) · [Tech Stack](#tech-stack)

---

</div>

## 📌 What is This Project?

BrainScan AI is an end-to-end AI-powered web application that analyses brain MRI scans and provides:

- **Tumor type classification** — Glioma, Meningioma, Pituitary Tumor, or No Tumor
- **Pixel-level tumor segmentation** — exact location and boundary of the tumor
- **Grad-CAM explainability** — visual heatmap showing *why* the AI made its decision
- **Scan history database** — all results stored and retrievable

> **Research gap addressed:** No existing paper combines classification + segmentation + visual explainability in a single deployable web application with scan history.

---

## 🎯 Results

| Model | Metric | Result |
|---|---|---|
| EfficientNetB0 Classifier | Test Accuracy | **94.19%** |
| EfficientNetB0 Classifier | Macro F1-Score | **93.29%** |
| EfficientNetB0 Classifier | No-Tumor Recall | **100%** |
| U-Net Segmentation | Dice Score | **0.8779** |
| U-Net Segmentation | Validation Accuracy | **99.77%** |

**Trained on:** 7,200 MRI images (classification) + 3,929 image-mask pairs (segmentation)

---

## 🏗️ System Architecture

```
User uploads MRI
       │
       ▼
┌──────────────────┐
│   PREPROCESSING  │  Skull strip · Resize (224×224 / 256×256) · Normalize [0,1]
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────┐ ┌──────────┐
│EfficientNetB0│ │  U-Net   │
│Classifier   │ │Segmentor │
│4-class      │ │256×256   │
│softmax      │ │binary mask│
└─────┬───┘ └────┬─────┘
      │           │
      ▼           ▼
┌──────────┐  ┌──────────┐
│ Grad-CAM │  │  Mask    │
│ Heatmap  │  │ Overlay  │
└─────┬────┘  └────┬─────┘
      └──────┬─────┘
             ▼
      ┌─────────────┐
      │ SQLite DB   │  label · confidence · paths · timestamp
      └──────┬──────┘
             ▼
      ┌─────────────┐
      │Flask Dashboard│ Classification + Segmentation + Heatmap + History
      └─────────────┘
```

---

## 🔬 Algorithms & Methods — What We Used and Why

### 1. 🧩 EfficientNetB0 — Tumor Classification

**What it is:** A CNN pretrained on 1.2 million ImageNet images, fine-tuned for brain tumor classification.

**Why we chose it over alternatives:**

| Model | Why Not Used |
|---|---|
| ResNet50 | Scales only depth — less efficient |
| VGG16 | Too heavy (138M params), slow inference |
| Custom CNN | Not enough data (7200 images) to train from scratch |
| Saha (3-model ensemble) | Too heavy to deploy |
| **EfficientNetB0** | ✅ Balanced depth+width+resolution, 5.3M params, deployable |

**Compound scaling formula:**
```
depth:      d = α^φ  (α=1.2)
width:      w = β^φ  (β=1.1)
resolution: r = γ^φ  (γ=1.15)
subject to: α · β² · γ² ≈ 2
```

**Two-phase training strategy:**
- Phase 1 — Freeze EfficientNetB0, train only classification head (LR=1e-3, 5-15 epochs)
- Phase 2 — Unfreeze top 30 layers, fine-tune end-to-end (LR=1e-4, up to 30 epochs)
- BatchNorm layers kept frozen in Phase 2 to prevent instability

**Why transfer learning:** Only 7,200 images available. Training a deep CNN from scratch requires millions of images. Reusing ImageNet weights means the model already knows edges, textures, and shapes — we just teach it brain tumors.

---

### 2. 🔍 U-Net — Tumor Segmentation

**What it is:** An encoder-decoder CNN with skip connections, designed specifically for medical image segmentation.

**Why U-Net over alternatives:**

| Method | Why Not Used |
|---|---|
| YOLO (Vijayanand, Qarni) | Produces bounding boxes, not pixel masks |
| FCM (Abdelhamied) | Classical, less precise than learned segmentation |
| 3D U-Net (Rajesh) | Requires 3D MRI volumes, heavy compute |
| **U-Net** | ✅ Pixel-level masks, skip connections preserve spatial detail, works well with small datasets |

**Architecture:**
```
Input (256×256×3)
    ↓ Encoder: 64→128→256→512→1024 filters (MaxPool ×4)
    ↓ Bottleneck: 1024 filters + Dropout
    ↑ Decoder: 512→256→128→64 (ConvTranspose + Skip Connections ×4)
    ↓ Output: sigmoid → binary mask (256×256×1)
```

**Skip connections formula:**
```
h'ₗ = Conv( Concat( ConvTranspose(h'ₗ₊₁), sₗ ) )
```
Skip connections reattach encoder feature maps to the decoder — preserving spatial detail lost during downsampling.

**Why Dice + BCE combined loss:**
```
L_total = L_BCE + L_Dice

L_BCE  = −[y·log(ŷ) + (1−y)·log(1−ŷ)]
L_Dice = 1 − (2·|Y∩Ŷ| + ε) / (|Y| + |Ŷ| + ε)
```
- BCE treats all pixels equally — but tumor pixels are only ~0.8% of the image
- Dice loss focuses on overlap between predicted and actual tumor region
- Combined loss handles both pixel accuracy AND class imbalance

---

### 3. 🌡️ Grad-CAM — Explainable AI

**What it is:** Gradient-weighted Class Activation Mapping — generates a visual heatmap showing which regions of the MRI drove the classification decision.

**Why Grad-CAM over alternatives:**

| Method | Why Not Used |
|---|---|
| LIME | Slow, approximation-based, unstable |
| SHAP | Computationally expensive for images |
| LLM text (Hareesh) | Not grounded in model attention |
| **Grad-CAM** | ✅ Fast, accurate, no external library needed, works with any CNN |

**How it works (5 steps):**
```
Step 1: Forward pass → get feature maps Aᵏ from last conv layer
Step 2: Backward pass → gradients of class score w.r.t. Aᵏ
Step 3: αᶜₖ = (1/Z) · ΣᵢΣⱼ (∂yᶜ/∂Aᵏᵢⱼ)  ← importance weight per channel
Step 4: L = ReLU( Σₖ αᶜₖ · Aᵏ )           ← weighted feature map sum
Step 5: Resize to 224×224, normalize to [0,1], overlay on MRI
```

**Why manual implementation:** No external libraries (no tf-keras-vis, no pytorch-grad-cam). Built from scratch using TensorFlow GradientTape — fully transparent and portable.

---

### 4. ⚙️ Preprocessing — Otsu Skull Stripping

**What it is:** Removes skull and background from MRI before feeding to the model.

**Why skull stripping:**
- Skull pixels are irrelevant to tumor classification
- Reduces noise and improves signal-to-noise ratio

**How Otsu thresholding works:**
```
1. Convert MRI to grayscale
2. Otsu algorithm finds optimal threshold T that minimizes intra-class variance:
   σ²w(T) = w₀(T)·σ²₀(T) + w₁(T)·σ²₁(T)
3. Apply threshold → binary mask
4. Morphological closing fills small holes
5. Find largest contour (the brain)
6. Zero out everything outside brain contour
```

---

### 5. 📊 Balanced Class Sampling — Training Data Generator

**Why needed:** Dataset has equal classes (1400 each) but standard random sampling can still create biased batches in practice.

**Our approach:** `BalancedGen` — forces exactly 8 images per class per batch (8×4=32). This completely eliminates class bias.

**Why this fixed 48% accuracy:** Earlier training used random sampling which accidentally created Glioma-heavy batches. The model learned to predict Glioma for everything. Balanced sampling fixed this and accuracy jumped to 94.19%.

---

### 6. 🗄️ Singleton Model Loading Pattern

**Why:** Loading a TensorFlow model takes 3-5 seconds. Reloading on every HTTP request would make the web app unusably slow.

**Solution:** Models loaded once at Flask startup, stored as module-level globals, reused for all subsequent requests.

```python
_classifier = None  # loaded once, reused forever
def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = load_model(...)
    return _classifier
```

---

## 📁 Project Structure

```
brain-tumor-diagnostic-system/
├── app/
│   ├── database/
│   │   └── db.py              # SQLite schema (ScanResult model)
│   ├── static/
│   │   ├── uploads/           # Uploaded MRI scans
│   │   └── results/           # Generated outputs
│   ├── templates/
│   │   ├── base.html          # Dark theme navbar + footer
│   │   ├── index.html         # 3D brain hero + upload
│   │   ├── result.html        # Classification + mask + heatmap
│   │   ├── history.html       # Paginated scan history
│   │   └── about.html         # Project info
│   ├── pipeline.py            # Core inference pipeline
│   ├── routes.py              # Flask routes
│   └── run.py                 # App entry point
├── src/
│   ├── classification/
│   │   ├── efficientnet_model.py   # EfficientNetB0 architecture
│   │   └── dataset.py              # Data generators
│   ├── segmentation/
│   │   ├── unet.py                 # U-Net architecture + losses
│   │   └── dataset.py              # Segmentation data loader
│   ├── explainability/
│   │   └── gradcam.py              # Manual Grad-CAM implementation
│   ├── preprocessing/
│   │   └── preprocess.py           # Skull strip + normalize + augment
│   └── utils/
│       └── visualization.py        # Plots + composite images
├── notebooks/
│   ├── colab_train_classification.ipynb
│   └── colab_train_segmentation.ipynb
├── models/                    # Trained .keras model files (not committed)
├── data/                      # Datasets (not committed)
├── config.py                  # All paths + hyperparameters
├── evaluate.py                # Full evaluation script
└── requirements.txt
```

---

## 🚀 Setup & Run

### Prerequisites
- Python 3.10–3.12
- ~15 GB disk space for datasets + models

### 1. Clone the repo
```bash
git clone https://github.com/mitali2744/Brain-tumor-diagnostic-system.git
cd Brain-tumor-diagnostic-system
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download datasets
Place in the `data/` folder:
- [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) → `data/Brain Tumor MRI Dataset/`
- [LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) → `data/LGG MRI Segmentation/`

### 5. Train models (or use Kaggle notebooks)
```bash
# Option A — Kaggle (recommended, free GPU)
# Open notebooks/colab_train_classification.ipynb on Kaggle
# Open notebooks/colab_train_segmentation.ipynb on Kaggle
# Download .keras files → place in models/

# Option B — Local
python -m notebooks.colab_train_classification
```

### 6. Run the web app
```bash
python app/run.py
```
Open [http://localhost:5000](http://localhost:5000)

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|---|---|---|
| Classification | EfficientNetB0 (TensorFlow) | Best accuracy/size tradeoff, compound scaling |
| Segmentation | U-Net (custom Keras) | Designed for medical images, skip connections |
| Explainability | Grad-CAM (manual, GradientTape) | Visual, grounded, no external dependency |
| Preprocessing | OpenCV + Albumentations | Fast, medical-grade augmentation |
| Web Framework | Flask 2.3 | Lightweight Python, easy ML model serving |
| Database | SQLite + SQLAlchemy | Zero-config, sufficient for single-user app |
| Frontend | HTML/CSS + Bootstrap 5 + Three.js | Dark theme UI + 3D brain visualization |
| Training | Kaggle (T4 GPU × 2) | Free, stable Python 3.10 environment |
| Version Control | Git + GitHub | Source code management |

---

## 📊 Datasets

| Dataset | Purpose | Size | Source |
|---|---|---|---|
| Brain Tumor MRI (Nickparvar) | Classification training | 7,200 images, 4 classes | [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| LGG MRI Segmentation (Buda) | U-Net training | 3,929 image-mask pairs | [Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) |

### Class Distribution (Classification)
```
Glioma      ████████████████  1,400 train / 400 test
Meningioma  ████████████████  1,400 train / 400 test
No Tumor    ████████████████  1,400 train / 400 test
Pituitary   ████████████████  1,400 train / 400 test
```

---

## 📚 References

| # | Paper | Method | Result |
|---|---|---|---|
| 1 | Jansi et al. (2023) | 9-layer CNN | 98.2% (binary only) |
| 2 | Dipu et al. (2021) | YOLOv5 + ResNet34 | 95.78% (binary only) |
| 3 | Micallef et al. (2021) | U-Net++ | Dice 0.87 |
| 4 | Saha et al. (2025) | 3-model ensemble | 99.5% (classification only) |
| 5 | Hareesh et al. (2025) | Xception + Gemini | 98.9% (LLM explanation) |
| 6 | Keerthan et al. (2025) | U-Net + CNN | IoU 0.37 val (overfitting) |
| 7 | Selvaraju et al. (2017) | Grad-CAM | Original XAI paper |
| 8 | Ronneberger et al. (2015) | U-Net | Original segmentation paper |
| 9 | Tan & Le (2019) | EfficientNet | Original scaling paper |

---

## ⚠️ Medical Disclaimer

This system is intended for **research and educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified radiologist for clinical decisions.

---

## 👥 Team

| Name | PRN |
|---|---|
| Utkarsha Bisen | 123B1E017 |
| Mitali Brahmankar | 123B1E018 |
| Sneha Iyer | 123B1E105 |

**Department:** Multidisciplinary Minor (Computer Programming)
**College:** PCCOE, Pune
**Guide:** Mrs. Swati Chandurkar

---

<div align="center">
Made with ❤️ for PCCOE MDM Capstone 2025–26
</div>