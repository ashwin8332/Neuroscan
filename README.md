# 🧠 Brain Tumor Detection & Multi-Model Comparative Study

A complete, production-grade Machine Learning and Deep Learning system for **Brain Tumor MRI Classification and Multi-Model Comparison**.

This project trains and systematically evaluates **four distinct neural architectures** on a 4-class Brain MRI dataset, performing detailed side-by-side analysis of **error metrics (cross-entropy loss, error rate, false positive/negative rates)** and **accuracy metrics (top-1 accuracy, balanced accuracy, precision, recall, macro F1, and multi-class ROC-AUC)**.

---

## 🔬 Dataset Overview

The dataset classifies MRI scans into four pathology categories:
- **Glioma**: Malignant tumors originating in glial cells with infiltrative margins.
- **Meningioma**: Typically benign, extra-axial dural-based tumors.
- **No Tumor**: Healthy brain scans without detectable lesions.
- **Pituitary**: Neoplasms affecting the pituitary gland near the sella turcica.

### Dataset Split Summary
- **Training Set**: 5,600 MRI images (~1,400 per class, balanced)
- **Testing Set**: 1,600 MRI images (~400 per class, balanced)
- **Image Resolution**: 224 x 224 RGB

---

## 🏗️ Model Architectures

1. **Custom Deep CNN** (`BrainTumorCNN`):
   - 4 convolutional blocks (32 -> 64 -> 128 -> 256 channels).
   - Batch Normalization & LeakyReLU ($\alpha=0.1$) activations.
   - Adaptive Average Pooling $(4 \times 4)$, Dropout (0.4), and Dense classification head.
   - Trainable parameters: ~1.44M.

2. **ResNet-18 (Transfer Learning)**:
   - Deep residual architecture with skip connections.
   - Pre-trained on ImageNet-1K with a custom classification head.
   - Trainable parameters: ~11.3M.

3. **MobileNetV3-Small (Hardware-Efficient CNN)**:
   - Modern, lightweight architecture with Squeeze-and-Excitation (SE) attention and Hard-Swish activations.
   - Optimized for ultra-low latency inference on edge/CPU devices.
   - Trainable parameters: ~1.8M.

4. **Vision Transformer (ViT-B/16)**:
   - Pure Transformer architecture using multi-head self-attention over $16 \times 16$ non-overlapping image patches.
   - State-of-the-art representation capacity.
   - Trainable parameters: ~86M.

---

## 📊 Error & Accuracy Comparison Metrics

The evaluation pipeline rigorously computes:
- **Error Metrics**:
  - **Cross-Entropy Loss (Log-Loss)**: Penalizes overconfident wrong predictions.
  - **Top-1 Error Rate (%)**: $100\% - \text{Accuracy}\%$.
  - **Misclassification Counts**: Total wrong predictions out of the test set.
  - **Class-Specific Miss Rate**: $1 - \text{Recall}$ for each tumor class.
- **Accuracy & Discrimination Metrics**:
  - **Overall Test Accuracy (%)**
  - **Balanced Accuracy (%)** (averages recall across all classes)
  - **Macro Precision, Recall, and F1-Score**
  - **Per-Class F1-Scores** (Glioma, Meningioma, No Tumor, Pituitary)
  - **Multi-Class ROC-AUC** (One-vs-Rest)
  - **Confusion Matrices**
- **Operational Efficiency**:
  - **Inference Latency** (milliseconds per image)
  - **Parameter Footprint** (Total vs. Trainable)

---

## 🚀 Quick Start Guide

### 1. Environment Setup
The project uses Python 3.11 with a virtual environment managed via `uv`:

```bash
# Run any script directly via uv:
uv run python <script_name>.py
```

Or activate the virtual environment:
```powershell
.venv\Scripts\activate
```

### 2. Train Models
Train individual models or all architectures:

```bash
# Train Custom CNN
uv run train.py --model custom_cnn --epochs 10

# Train ResNet-18
uv run train.py --model resnet18 --epochs 10

# Train MobileNetV3
uv run train.py --model mobilenet_v3 --epochs 10

# Train Vision Transformer
uv run train.py --model vit --epochs 10

# Train all models sequentially
uv run train.py --model all --epochs 10
```

> **Tip for quick testing**: Use `--subset-fraction 0.1` for a fast 10% dry-run.

### 3. Evaluate and Benchmark Models
Run comprehensive evaluation on the 1,600 test images:

```bash
uv run evaluate.py
```
This prints the comparative table in console and exports:
- `results/model_comparison_metrics.csv`
- `results/model_comparison_metrics.json`

### 4. Generate Publication-Quality Visualizations
Generate side-by-side comparative plots:

```bash
uv run compare.py
```
Figures generated in `results/plots/`:
- `accuracy_and_error_comparison.png`: Dual bar chart comparing Accuracy vs Error Rate.
- `f1_and_auc_comparison.png`: Macro F1 and ROC-AUC comparisons.
- `per_class_f1_heatmap.png`: Matrix of per-class F1-scores across models.
- `confusion_matrices_grid.png`: Side-by-side confusion matrices for all models.
- `training_loss_accuracy_curves.png`: Convergence loss and accuracy curves over epochs.

### 5. Launch the NeuroScan AI Web Application (Recommended)
Launch the web interface powered by FastAPI:

```bash
uv run python server.py
```
Open your browser at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

Features:
- **Live Diagnostic Lab**: Real-time inference across Custom CNN, ResNet-18, MobileNetV3, and ViT with **Grad-CAM Attention Heatmaps** (with interactive opacity slider).
- **1-Click Test Presets**: Pre-loaded validated MRI scans from all 4 classes.
- **Model Benchmark & Comparison Hub**: Interactive metrics tables and Chart.js plots for Accuracy vs Error Rate, Macro F1 vs ROC-AUC, and Per-Class Breakdown.
- **Error & Confusion Analysis**: Normalized confusion matrix inspector and clinical failure mode analysis.
- **Pathology Atlas**: Radiographic guide and anatomical details for Glioma, Meningioma, Pituitary Adenoma, and Normal scans.

*(Alternative Gradio interface also available at `uv run python app.py` on port 7861)*

### 6. Predict Single MRI Image via CLI
```bash
uv run predict.py "Dataset/brain-tumor-mri-dataset/Testing/glioma/Te-gl_1.jpg" --model resnet18
```

### 7. Interactive Jupyter Notebook
Open `notebooks/model_comparison.ipynb` to step through the complete analysis, exploratory data analysis (EDA), code explanations, and visualization figures.

---

## 📁 Repository Structure

```text
AIML Project/
├── Dataset/
│   └── brain-tumor-mri-dataset/
│       ├── Training/           # 5,600 training images (4 classes)
│       └── Testing/            # 1,600 testing images (4 classes)
├── models/                     # Saved .pth model checkpoints
├── results/                    # Metrics CSV/JSON and evaluation output
│   └── plots/                  # Comparison figures (PNG)
├── notebooks/
│   └── model_comparison.ipynb # Step-by-step Jupyter Notebook
├── src/
│   ├── __init__.py
│   ├── config.py              # Centralized paths, classes, hyperparameters
│   ├── data.py                # Augmentation transforms & data loaders
│   ├── models.py              # CNN, ResNet-18, MobileNetV3, ViT architectures
│   ├── metrics.py             # Error, accuracy, loss, and latency calculators
│   └── inference.py           # Preprocessing, Grad-CAM, single-image inference
├── app.py                     # Gradio multi-model comparative web app
├── train.py                   # Multi-architecture training CLI
├── evaluate.py                # Test set evaluation and metrics exporter
├── compare.py                 # Comparative visualization generator
├── predict.py                 # Single-image prediction CLI
└── requirements.txt           # Verified environment dependencies
```
