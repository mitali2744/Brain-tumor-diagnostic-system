# Deep Learning-Based Brain Tumor Detection, Segmentation and Diagnostic Support System

## Overview
An end-to-end diagnostic pipeline that takes an MRI scan and automatically:
1. **Segments** the tumor region (U-Net)
2. **Classifies** the tumor type (EfficientNetB0)
3. **Explains** the prediction visually (Grad-CAM)
4. **Displays** everything on a web dashboard (Flask)

## Research Gap
No existing paper combines all four of:
- Pixel-level segmentation
- 4-class tumor classification (Glioma / Meningioma / Pituitary / No Tumor)
- Visual explainability via Grad-CAM
- Deployable web interface

## Project Structure
```
brain_tumor_system/
├── data/                        # Dataset storage (not committed)
│   ├── classification/          # Kaggle 4-class MRI dataset
│   └── segmentation/            # LGG MRI segmentation dataset
├── models/                      # Saved trained models
├── notebooks/                   # Exploratory notebooks
├── src/
│   ├── preprocessing/           # Image preprocessing utilities
│   ├── segmentation/            # U-Net model + training
│   ├── classification/          # EfficientNetB0 model + training
│   ├── explainability/          # Grad-CAM implementation
│   └── utils/                   # Shared helpers
├── app/                         # Flask web application
│   ├── static/
│   ├── templates/
│   └── database/
├── train_segmentation.py        # Train U-Net
├── train_classification.py      # Train EfficientNetB0
├── evaluate.py                  # Evaluate both models
├── requirements.txt
└── config.py                    # Central config
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare datasets
- Download [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) → `data/classification/`
- Download [LGG MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) → `data/segmentation/`

### 3. Train models
```bash
python train_classification.py
python train_segmentation.py
```

### 4. Run the web app
```bash
python app/run.py
```
Visit `http://localhost:5000`

## Pipeline
```
MRI Scan → Preprocessing → U-Net Segmentation → EfficientNetB0 Classification → Grad-CAM → Dashboard
```

## Tech Stack
| Layer | Technology |
|---|---|
| Model Training | Python, TensorFlow/Keras |
| Segmentation | U-Net (custom) |
| Classification | EfficientNetB0 (ImageNet pretrained) |
| Explainability | Grad-CAM (manual implementation) |
| Backend | Flask |
| Database | SQLite |
| Frontend | HTML/CSS + Jinja2 |
