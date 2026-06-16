# 🌿 Plant Disease Classification — Classical & Deep Learning Comparative Study

<div align="center">

** Computer Vision Project**
*Zewail City of Science, Technology and Innovation*

[Overview](#-overview) • [Results](#-results) • [Installation](#-installation) • [Usage](#-usage) • [Methodology](#-methodology) • [Future Work](#-future-work)

</div>

---

## 📋 Overview

A **comprehensive comparative study** of classical computer vision and deep learning approaches for automated plant disease classification on the PlantVillage dataset. This project evaluates four model pipelines — from a handcrafted HOG+SVM baseline to a ResNet50 transfer learning model — across both clean test data and controlled synthetic distortion scenarios that simulate real-world image degradation.

This work directly addresses the challenge of early, scalable, and cost-effective plant disease diagnosis, a critical problem given that crop diseases account for **20–40% of global agricultural production losses annually**.

---

## 🎯 Objectives

- Establish a classical handcrafted feature baseline (HOG + ORB-BoVW + SVM) for 38-class plant disease classification
- Design and train a custom CNN architecture (PlantCNN) from scratch, with architectural analysis of overfitting behaviour
- Investigate the impact of data augmentation on generalisation and distortion robustness
- Evaluate transfer learning from a ResNet50 ImageNet backbone and measure performance gains
- Conduct a systematic robustness evaluation under four categories of synthetic image distortion at multiple severity levels
- Perform an adversarial robustness analysis (FGSM) as a bonus experiment

---

## 📊 Results

### Clean Test Set Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| HOG + ORB-BoVW + SVM | 78.40% | 0.784 | 0.784 | 0.780 |
| Custom PlantCNN (no aug.) | 93.44% | 0.938 | 0.934 | 0.934 |
| PlantCNN + Augmentation | 87.48% | 0.893 | 0.875 | 0.871 |
| **ResNet50 Transfer Learning** | **97.73%** | **0.977** | **0.977** | **0.977** |

### Robustness Under Synthetic Distortions

| Model | Clean | Gaussian Noise (σ=25) | Blur (7×7) | Brightness ×0.5 | Rotation 30° |
|-------|-------|-----------------------|------------|-----------------|--------------|
| HOG+SVM | 78.40% | 42.10% | 38.75% | 55.20% | 47.90% |
| Custom CNN | 93.44% | 71.35% | 68.90% | 79.60% | 74.10% |
| CNN+Aug | 87.48% | 80.25% | 78.10% | 84.30% | 82.75% |
| Transfer (R50) | 97.73% | 88.40% | 86.55% | 92.10% | 90.35% |

> **Key finding:** Transfer learning achieves a 19.33% absolute accuracy gain over the classical baseline on clean data and retains the highest accuracy across all distortion categories.

---

## 📁 Repository Structure

```
plant-disease-classification/
│
├── 📓 CV_Project_PlantVillage_Main__20_Epochs__final.ipynb   # Main notebook (all phases)
│
├── src/
│   ├── classical/
│   │   └── hog_orb_svm_pipeline.py       # HOG + ORB-BoVW + SVM baseline
│   ├── cnn/
│   │   ├── plant_cnn.py                   # Custom PlantCNN architecture
│   │   └── train_cnn.py                   # Training loop (baseline + augmented)
│   ├── transfer_learning/
│   │   └── resnet50_finetune.py           # ResNet50 fine-tuning pipeline
│   └── utils/
│       ├── dataset.py                     # Dataset loading, splitting, tf.data pipeline
│       ├── augmentation.py                # Augmentation transforms
│       ├── distortions.py                 # Synthetic distortion utilities
│       └── metrics.py                     # Evaluation helpers
│
├── configs/
│   └── config.yaml                        # Hyperparameters and paths
│
├── results/
│   ├── figures/                           # Training curves, confusion matrices, distortion plots
│   └── tables/                            # Accuracy tables (CSV)
│
├── docs/
│   └── CV_PlantVillage_Paper.pdf          # Full project report
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🛠 Technologies Used

| Category | Tools |
|----------|-------|
| Deep Learning | TensorFlow 2.x, Keras, PyTorch |
| Transfer Learning | ResNet50 (ImageNet pretrained) |
| Classical CV | OpenCV, scikit-image (HOG, ORB) |
| ML | scikit-learn (SVM, GridSearchCV, MiniBatchKMeans) |
| Data | NumPy, Pandas, PIL |
| Visualization | Matplotlib, Seaborn |
| Dataset | PlantVillage (Kaggle) |
| Environment | Google Colab (T4 GPU) |

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- GPU recommended (NVIDIA T4 or equivalent)

### Clone & Install

```bash
git clone https://github.com/yourusername/plant-disease-classification.git
cd plant-disease-classification
pip install -r requirements.txt
```

### `requirements.txt`
```
tensorflow>=2.12
torch>=2.0
scikit-learn>=1.3
scikit-image>=0.21
opencv-python>=4.8
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
tqdm>=4.65
Pillow>=9.5
kaggle>=1.5
```

### Dataset

1. Create a Kaggle account and obtain your API token
2. Run the dataset download cell in the notebook:
```bash
kaggle datasets download -d mohitsingh1804/plantvillage -p ./data --unzip
```

---

## 🚀 Usage

### Run the Full Pipeline (Notebook)
Open `CV_Project_PlantVillage_Main__20_Epochs__final.ipynb` in Google Colab and run all cells sequentially.

The notebook is organized into phases:

| Phase | Description |
|-------|-------------|
| Phase 1 | Setup, dataset download, EDA, train/val/test split |
| Phase 2 | Classical baseline: HOG + ORB-BoVW + SVM |
| Phase 3 | Custom PlantCNN (no augmentation) |
| Phase 4 | PlantCNN + Data Augmentation |
| Phase 5 | ResNet50 Transfer Learning (frozen → fine-tuned) |
| Phase 6 | Robustness evaluation under synthetic distortions |
| Bonus | FGSM adversarial robustness analysis |

### Run Individual Modules

```python
# Classical pipeline
from src.classical.hog_orb_svm_pipeline import ClassicalPipeline
pipeline = ClassicalPipeline(data_dir="./data/plantvillage")
pipeline.train()
pipeline.evaluate()

# Transfer learning
from src.transfer_learning.resnet50_finetune import TransferModel
model = TransferModel(num_classes=38)
model.train(train_ds, val_ds, epochs=20)
model.evaluate(test_ds)
```

---

## 🧠 Methodology

### Dataset
- **PlantVillage**: 54,305 RGB leaf images, 38 disease/healthy classes, 14 plant species
- All images resized to **224 × 224**
- Stratified split: **70% train / 15% val / 15% test** (fixed seed=42, saved to CSV for reproducibility)

### Model 1 — Classical Baseline (HOG + ORB-BoVW + SVM)
- BGR → HSV colour space conversion
- HOG descriptors: 9 orientation bins, 16×16 cells, 2×2 block normalisation (~26,244-dim vector)
- ORB keypoints encoded into 200-dimensional Bag-of-Visual-Words histogram (MiniBatchKMeans)
- SVM (RBF kernel) trained on 8,000-image stratified subset; hyperparameters via 3-fold grid search

### Model 2 — Custom PlantCNN (from scratch)
- 4 convolutional blocks: Conv2D → BatchNorm → ReLU → MaxPool
- Filter progression: 32 → 64 → 128 → 256; Global Average Pooling head; Dropout(0.5)
- **399,142 trainable parameters**
- Adam (lr=1e-3, wd=1e-4), CrossEntropyLoss, ReduceLROnPlateau, early stopping (patience=7)

### Model 3 — PlantCNN + Augmentation
- Same architecture as Model 2
- Training augmentations: random flips, rotations (±30°), brightness/contrast jitter (±20%), saturation (±20%)

### Model 4 — ResNet50 Transfer Learning
- ResNet50 pretrained on ImageNet; final FC replaced with 38-class linear head
- **Phase 1 (head-only):** backbone frozen, head trained → 90.91% val accuracy
- **Phase 2 (fine-tuning):** last 2 residual blocks + head unfrozen → 98.28% val accuracy
- All augmentations from Model 3 retained

### Robustness Evaluation
Four distortion types applied exclusively at inference time:
- **Gaussian noise**: σ ∈ {10, 25, 50}
- **Gaussian blur**: kernel ∈ {3×3, 7×7, 15×15}
- **Brightness scaling**: factors ∈ {0.5, 0.75, 1.25, 1.5}
- **Rotation**: angles ∈ {15°, 30°, 45°}

---

## 🔮 Future Improvements

- [ ] Evaluate on real-world field-collected datasets (e.g., AI Challenger Plant Disease)
- [ ] Explore lightweight architectures (MobileNetV3, EfficientNet-B0) for edge/mobile deployment
- [ ] Implement adversarial training (PGD, TRADES) for distortion-robust classifiers
- [ ] Multi-scale feature fusion for fine-grained disease stage discrimination
- [ ] Explainability layer: Grad-CAM visualizations to localize disease regions
- [ ] Extend to field conditions: background clutter, partial occlusion, uncontrolled lighting
- [ ] Build a mobile app demo using TensorFlow Lite quantization

---

## 👩‍💻 Authors

| Name | ID | Role |
|------|----|------|
| Esraa Ata | 202001430 | Deep Learning, Transfer Learning, Robustness Evaluation |
| Abdulrhman Ewiah | 201800747 | Classical CV Pipeline, Dataset Preparation |
| Hala Mohamed | 202100987 | CNN Architecture, Augmentation Experiments |

**Supervised by:** Dr. Mohamed Tolba

**Course:** Computer Vision — Zewail City of Science, Technology and Innovation

<div align="center">

⭐ If this project helped you, please consider giving it a star!

</div>
