# Fuzzy-CapsNet — Brain Tumor Classification

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Task-Medical%20Imaging-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Test%20Accuracy-91.50%25-blue?style=for-the-badge"/>
</p>

A **Hybrid Fuzzy Capsule Network** for classifying brain MRI scans into 4 categories. The core innovation replaces classic CapsNet's hard dynamic routing with **Gaussian Fuzzy Membership routing**, resulting in smoother gradient flow, better interpretability, and improved handling of intra-class variation.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Installation](#installation)
- [Project Structure](#project-structure)

---

## Overview

Standard Capsule Networks use iterative dynamic routing based on dot-product agreement, which can produce hard, non-smooth coupling coefficients. This project replaces that mechanism with **Fuzzy Gaussian Membership routing**:

$$\mu(\hat{u}, v) = \exp\!\left(-\frac{\|\hat{u} - v\|^2}{2\sigma^2}\right)$$

Where $\sigma$ is a **learnable parameter per output capsule**, allowing the network to adapt its routing sensitivity during training.

**Key benefits:**
- Membership scores are naturally bounded in `[0, 1]` → directly interpretable
- Smooth gradients throughout the routing iterations
- More graceful handling of overlapping feature distributions

---

## Architecture

```
Input (3×64×64)
       │
       ▼
ConvFeatureExtractor          ← 3-block CNN (32→64→128 channels) + BatchNorm + Dropout
       │
       ▼
PrimaryCapsLayer              ← CNN features → 8-D capsule vectors (squash activation)
       │
       ▼
FuzzyCapsuleLayer             ← Gaussian Membership Routing (3 iterations, σ learnable)
       │
       ▼
L2-Norm  →  Softmax           ← Class scores from capsule vector lengths
```

| Component | Details |
|-----------|---------|
| Input size | 64 × 64 × 3 |
| Primary capsules | 32 capsules × 8-D |
| Output capsules | 4 capsules × 16-D (one per class) |
| Routing iterations | 3 |
| Total parameters | **1,437,444** |
| Loss function | CapsNet Margin Loss (Hinton et al., 2017) |

---

## Dataset

Uses the **Brain Tumor MRI Dataset** (available on [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)), organized as:

```
dataset/
├── Training/
│   ├── glioma/        (1400 images)
│   ├── meningioma/    (1400 images)
│   ├── notumor/       (1400 images)
│   └── pituitary/     (1400 images)
└── Testing/
    ├── glioma/        (400 images)
    ├── meningioma/    (400 images)
    ├── notumor/       (400 images)
    └── pituitary/     (400 images)
```

| Split | Samples |
|-------|---------|
| Train | 4,760 |
| Validation (15% of train) | 840 |
| Test | 1,600 |

**Training augmentations:** Random horizontal flip, rotation (±15°), color jitter.

---

## Results

### Test Set Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | **91.25%** |
| Precision (weighted) | 91.70% |
| Recall (weighted) | 91.25% |
| F1-Score (weighted) | 91.03% |
| ROC-AUC (macro OvR) | **96.47%** |

### Per-Class Breakdown (Test Set)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | 0.9743 | 0.7575  | 0.8523 |
| Meningioma | 0.8670 | 0.9125 | 0.8892 |
| No Tumor | 0.8864 | 0.9950 | 0.9376 |
| Pituitary | 0.9403 | 0.9850 | 0.9621 |

> **Note:** Glioma shows slightly lower recall — a known challenge due to its visual similarity with Meningioma in MRI scans.

### Training Set Performance

| Metric | Score |
|--------|-------|
| Accuracy | 99.12% |
| F1-Score (macro) | 99.13% |
| ROC-AUC (macro) | 99.97% |

Training ran for **50 epochs** on CPU. Best model checkpoint selected via early stopping on validation loss (`val_loss = 0.0342`).

---

## Usage

### 1. Prepare the Dataset

Download the dataset from Kaggle and place it under `./dataset/` following the structure shown above.

### 2. Run the Notebook

Open `fuzzy_capsnet.ipynb` in Jupyter and run all cells sequentially:

```bash
jupyter notebook fuzzy_capsnet.ipynb
```

The notebook is organized into 7 sections:

| Section | Description |
|---------|-------------|
| 1 | Imports & Configuration |
| 2 | Model Architecture |
| 3 | Data Loading |
| 4 | Training (50 epochs, early stopping) |
| 5 | Quick Evaluation — metrics & plots |
| 6 | Full Evaluation — 8 detailed plots |
| 7 | Single-Image Prediction & Explainability |

### 3. Single Image Prediction

```python
# Load a saved model and run inference on one image
model = FuzzyCapsNet(img_size=64)
model.load_state_dict(torch.load('fuzzy_capsnet.pt', map_location='cpu'))

sample_imgs, _ = next(iter(test_eval_loader))
pred_class, probs, membership = predict_and_explain(
    model, sample_imgs[0:1], save_path='prediction_explanation.png'
)
```

Output includes:
- **Softmax probabilities** per class
- **Fuzzy Membership Scores** (0–1) — key for explainability

### Configuration

All hyperparameters are defined at the top of Section 1:

```python
IMG_SIZE            = 64
BATCH_SIZE          = 16
NUM_EPOCHS          = 50
LEARNING_RATE       = 1e-3
WEIGHT_DECAY        = 1e-4
DROPOUT_P           = 0.3
EARLY_STOP_PATIENCE = 5
VAL_SPLIT           = 0.15
ROUTING_ITERS       = 3
```

---

## Project Structure

```
fuzzy-capsnet/
│
├── app.py
├── fuzzy_capsnet.ipynb       # Main notebook (all sections)
├── fuzzy_capsnet.pt          # Saved model weights (after training)
│
├── dataset/
│   ├── Training/
│   └── Testing/
│
└── eval_output/              # Generated evaluation plots
    ├── 1_class_distribution.png
    ├── 2_sample_images.png
    ├── 3_preprocessing.png
    ├── 4_confusion_matrix.png
    ├── 5_per_class_accuracy.png
    ├── 6_confidence_distribution.png
    ├── 7_fuzzy_membership.png
    └── 8_misclassified.png
    ├── per_class_metrics.png
    ├── prediction_explanation.png
    ├── roc_auc.png
    └── training_curves.png
```

---

## How Fuzzy Routing Works

Unlike standard dynamic routing (which uses softmax over raw dot-product logits), fuzzy routing measures **Gaussian similarity** between predicted capsule vectors $\hat{u}_{j|i}$ and the current output capsule $v_j$:

```
for iteration in range(routing_iters):
    c  = softmax(b)                          # coupling coefficients
    s  = weighted_sum(c, u_hat)              # weighted prediction
    v  = squash(s)                           # output capsule

    # Fuzzy update (instead of dot-product agreement):
    membership = exp(-‖u_hat − v‖² / 2σ²)   # Gaussian similarity
    b += membership                           # update routing logits
```

Each output capsule has its own learnable `log_sigma`, allowing the routing bandwidth to adapt per class during training.

---

## References

- Sabour, S., Frosst, N., & Hinton, G. E. (2017). *Dynamic Routing Between Capsules*. NeurIPS.
- Brain Tumor MRI Dataset — [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

