# Fuzzy-CapsNet — Brain Tumor Classification

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Task-Medical%20Imaging-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Test%20Accuracy-91.69%25-blue?style=for-the-badge"/>
</p>

A **Hybrid Fuzzy Capsule Network** for classifying brain MRI scans into 4 categories. The core innovation replaces classic CapsNet's hard dynamic routing with **Gaussian Fuzzy Membership routing**, resulting in smoother gradient flow, better interpretability, and improved handling of intra-class variation.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Inference API](#inference-api)
- [Probability Calibration](#probability-calibration)
- [Fuzzy Membership Scores](#fuzzy-membership-scores)
- [Dataset](#dataset)
- [Results](#results)
- [Live Demo](#live-demo)
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
L2-Norm                       ← Capsule norms ∈ (0,1) — used directly for MarginLoss
       │
       ▼
Prediction API                ← Temperature-scaled softmax / normalised memberships
```

| Component | Details |
|-----------|---------|
| Input size | 64 × 64 × 3 |
| Primary capsules | 32 capsules × 8-D |
| Output capsules | 4 capsules × 16-D (one per class) |
| Routing iterations | 3 |
| Total parameters | **1,437,444** |
| Loss function | CapsNet Margin Loss (Hinton et al., 2017) |
| Inference calibration | Temperature Scaling (T = 0.1) |

---

## Inference API

`forward()` returns raw capsule norms only — a single tensor `(B, num_classes)`. Four dedicated inference helpers are provided so that training and evaluation concerns stay cleanly separated:

| Method | Returns | Purpose |
|--------|---------|---------|
| `forward(x)` | norms `(B, C)` | Training — passed directly to MarginLoss |
| `predict_logits(x)` | norms `(B, C)` | Raw capsule activations for inspection |
| `predict_proba(x)` | probs `(B, C)` | Calibrated class probabilities (temperature-scaled) |
| `predict_membership(x)` | memberships `(B, C)` | Normalised fuzzy memberships (sums to 1) |
| `predict_with_details(x)` | dict | All of the above in one call |

`predict_with_details()` returns:
```python
{
    "norms":         Tensor (B, C),   # raw capsule norms
    "probabilities": Tensor (B, C),   # temperature-scaled softmax
    "memberships":   Tensor (B, C),   # normalised fuzzy memberships
    "predictions":   Tensor (B,),     # argmax class indices
    "confidence":    Tensor (B,),     # max probability per sample
}
```

The inference temperature can be adjusted at any time without retraining:
```python
model.set_temperature(0.2)   # softer
model.set_temperature(0.05)  # sharper
```

---

## Probability Calibration

Capsule norms produced by the squash nonlinearity are compressed into `(0, 1)`. After training, the correct class typically reaches a norm of ~0.85–0.93 while incorrect classes settle at ~0.60–0.75 — a relative difference of only 0.15–0.30. Standard softmax applied directly on these values spreads probability mass nearly uniformly:

```
norms  = [0.91, 0.73, 0.69, 0.65]
softmax           → [0.40, 0.24, 0.22, 0.21]   ← weak confidence
softmax (T=0.1)   → [0.87, 0.05, 0.04, 0.04]   ← sharp, calibrated
```

The model uses **temperature-scaled softmax** at inference:

$$p_i = \frac{\exp(\text{norm}_i \;/\; T)}{\sum_j \exp(\text{norm}_j \;/\; T)}$$

With default `T = 0.1`. This is applied **only at inference** — training via MarginLoss is completely unaffected, so accuracy is preserved. Temperature scaling also explains why MarginLoss alone does not drive sharp probabilities: MarginLoss only penalises when the correct class norm falls below `m_plus = 0.9` or a wrong class norm exceeds `m_minus = 0.1`, with no explicit term that pushes the softmax distribution toward 0/1.

---

## Fuzzy Membership Scores

Fuzzy membership scores reflect **how strongly each class capsule was activated**, expressed as a fraction of total capsule activity:

$$\mu_i = \frac{\text{norm}_i}{\sum_j \text{norm}_j}$$

This is a proper **possibility distribution** — values sum to exactly 1 and preserve the relative activation ratios between class capsules. It is consistent with fuzzy-set-theoretic semantics: each class receives a share of the total "evidence" proportional to its capsule norm.

This is different from the softmax probabilities: memberships reflect the raw routing agreement across all classes simultaneously, while probabilities are a sharpened decision output. Both views are exposed in the prediction API and in the explainability panel.

> **Why fuzzy routing naturally distributes partial memberships:** The Gaussian kernel is never exactly zero for any finite distance. Every primary capsule always contributes *some* membership to *every* output capsule. This is correct fuzzy behaviour — partial membership is the design intent, not a limitation to be removed.

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
| **Accuracy** | **91.69%** |
| Precision (weighted) | 92.22% |
| Recall (weighted) | 91.69% |
| F1-Score (weighted) | 91.47% |
| ROC-AUC (macro OvR) | **98.42%** |

### Per-Class Breakdown (Test Set)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | 0.9773 | 0.7525 | 0.8503 |
| Meningioma | 0.8465 | 0.9375 | 0.8897 |
| No Tumor | 0.9089 | 0.9975 | 0.9511 |
| Pituitary | 0.9561 | 0.9800 | 0.9679 |

> **Note:** Glioma shows slightly lower recall — a known challenge due to its visual similarity with Meningioma in MRI scans.

### Training Set Performance

| Metric | Score |
|--------|-------|
| Accuracy | 99.41% |
| F1-Score (macro) | 99.41% |
| ROC-AUC (macro) | 99.99% |

Training ran for **42 epochs** on CPU. Best model checkpoint selected via early stopping on validation loss (`val_loss = 0.0353`).

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
| 2 | Model Architecture — FuzzyCapsNet + full inference API |
| 3 | Data Loading |
| 4 | Training (50 epochs, MarginLoss, early stopping) |
| 5 | Quick Evaluation — metrics & plots |
| 6 | Full Evaluation — 8 detailed plots |
| 7 | Single-Image Prediction & Explainability |

### 3. Single Image Prediction

```python
model = FuzzyCapsNet(img_size=64)
model.load_state_dict(torch.load('fuzzy_capsnet.pt', map_location='cpu'))

sample_imgs, _ = next(iter(test_eval_loader))

# Option A — full details dict
details = model.predict_with_details(sample_imgs[0:1])
print(details["predictions"], details["confidence"])

# Option B — explainability panel (saves PNG + prints console summary)
pred_class, probs, membership = predict_and_explain(
    model, sample_imgs[0:1], save_path='prediction_explanation.png'
)
```

Output includes:
- **Calibrated probabilities** per class (temperature-scaled softmax, T=0.1)
- **Normalised Fuzzy Membership Scores** (sum to 1) — key for explainability

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

## Live Demo

> **Try it instantly — no installation required.**

An interactive demo is hosted on **Hugging Face Spaces** using **Gradio**:

**[Click here to try the live demo](https://huggingface.co/spaces/nooragab/brain-tumor-classification)**

> App version **v1.1** — includes temperature-calibrated probabilities and normalised fuzzy memberships.

### What the app does

Upload any brain MRI scan and the model returns:

- **Predicted class** — Glioma, Meningioma, No Tumor, or Pituitary
- **Calibrated confidence scores** — temperature-scaled probability bar chart for all 4 classes.
- **Fuzzy Membership Scores** — normalised capsule activations (sum to 1), visualised as a bar chart and polar radar

### Run the app locally

```bash
pip install gradio torch torchvision Pillow
python app.py
```

Then open `http://localhost:7860` in your browser.

---

## How Fuzzy Routing Works

Unlike standard dynamic routing (which uses softmax over raw dot-product logits), fuzzy routing measures **Gaussian similarity** between predicted capsule vectors $\hat{u}_{j|i}$ and the current output capsule $v_j$:

```
for iteration in range(routing_iters):
    c  = softmax(b)                          # coupling coefficients
    s  = weighted_sum(c, u_hat)              # weighted prediction
    v  = squash(s)                           # output capsule

    # Fuzzy update (instead of dot-product agreement):
    membership = exp(-‖u_hat − v‖² / 2σ²)   # Gaussian similarity ∈ (0,1]
    b += membership                           # update routing logits
```

Each output capsule has its own learnable `log_sigma`, allowing the routing bandwidth to adapt per class during training. The `return_memberships=True` flag on `FuzzyCapsuleLayer.forward()` exposes the final routing membership matrix `(B, Ni, No)` for analysis and debugging without affecting standard inference.

---

## Project Structure

```
fuzzy-capsnet/
│
├── app.py                    # Gradio demo app (Hugging Face Spaces)
├── fuzzy_capsnet.ipynb       # Main notebook (all sections)
├── fuzzy_capsnet.pt          # Saved model weights (after training)
│
├── dataset/
│   ├── Training/
│   └── Testing/
│
└── eval_output/              # Generated evaluation plots
    ├── 01_class_distribution.png
    ├── 02_sample_images.png
    ├── 03_preprocessing.png
    ├── 04_confusion_matrix.png
    ├── 05_per_class_accuracy.png
    ├── 06_confidence_distribution.png
    ├── 07_fuzzy_membership.png
    ├── 08_misclassified.png
    ├── per_class_metrics.png
    ├── prediction_explanation.png
    ├── roc_auc.png
    └── training_curves.png
```

---

## References

- Sabour, S., Frosst, N., & Hinton, G. E. (2017). *Dynamic Routing Between Capsules*. NeurIPS.
- Brain Tumor MRI Dataset — [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
