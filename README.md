# GCPDS Computer Vision Python Kit

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.1.0.63-green.svg)](https://github.com/UN-GCPDS/gcpds-cv-pykit)

A comprehensive PyTorch-based toolkit for computer vision and semantic segmentation tasks, developed by the GCPDS Team at Universidad Nacional de Colombia.

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Examples](#examples)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Available Models](#available-models)
- [Loss Functions](#loss-functions)
- [Datasets](#datasets)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Experiment Tracking](#experiment-tracking)
- [Performance Evaluation](#performance-evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

---

## 🔍 Overview

**gcpds-cv-pykit** is a powerful and flexible toolkit designed for semantic segmentation tasks in computer vision. Built on PyTorch, it provides a complete pipeline from dataset preparation to model training and evaluation, with built-in support for experiment tracking via Weights & Biases.

### Key Highlights

- 🎯 **Multiple State-of-the-Art Models**: UNet, ResUNet, DeepLabV3+, FCN
- 📊 **Comprehensive Loss Functions**: DICE, Cross-Entropy, Focal, Tversky
- 🗂️ **Easy Dataset Management**: Built-in support for Kaggle datasets
- 📈 **Experiment Tracking**: Seamless integration with Weights & Biases
- 🔧 **Flexible Configuration**: Dictionary-based configuration system
- 🚀 **Production Ready**: Mixed precision training, GPU optimization
- 📉 **Rich Evaluation Metrics**: DICE, Jaccard, Sensitivity, Specificity

---

## ✨ Features

### Core Capabilities

- **Baseline Segmentation Models**
  - UNet with customizable depth and filters
  - ResUNet with residual connections
  - DeepLabV3+ with atrous spatial pyramid pooling
  - Fully Convolutional Networks (FCN)

- **Advanced Training Features**
  - Mixed precision training (AMP) for faster computation
  - Automatic learning rate scheduling
  - Training phases to leverage features learned by pre-trained models
  - Early stopping with patience
  - Model checkpointing (best and last)
  - GPU memory optimization
  - Multi-GPU support

- **Loss Functions**
  - DICE Loss for imbalanced segmentation
  - Cross-Entropy Loss
  - Focal Loss for hard example mining
  - Tversky Loss for precision-recall trade-off

- **Dataset Utilities**
  - Automatic Kaggle dataset download
  - Pre-configured datasets: OxfordIITPet, SeedGermination, BreastCancer, FeetMamitas
  - Support for crowd-sourced datasets
  - Custom dataset integration

- **Visualization Tools**
  - Random sample visualizations
  - Training progress plots
  - Prediction overlays

- **Performance Evaluation**
  - Comprehensive metrics calculation
  - Per-class and global statistics
  - Results export to Numpy files

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- pip or conda package manager

### Basic Installation

```bash
pip install gcpds-cv-pykit
```

### Installation from Source

```bash
# Clone the repository
git clone https://github.com/UN-GCPDS/gcpds-cv-pykit.git
cd gcpds-cv-pykit

# Install in development mode
pip install -e .
```

### Installation with Optional Dependencies

```bash
# For development (includes testing and linting tools)
pip install gcpds-cv-pykit[dev]

# For documentation building
pip install gcpds-cv-pykit[docs]

# For Jupyter notebook support
pip install gcpds-cv-pykit[jupyter]

# Install all optional dependencies
pip install gcpds-cv-pykit[all]
```

### Dependencies

Core dependencies include:
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.21.0
- opencv-python >= 4.6.0
- matplotlib >= 3.5.0
- wandb >= 0.15.0
- tqdm >= 4.64.0
- kagglehub (for dataset downloads)

---

## 🚀 Quick Start

### 1. Download a Dataset

```python
from gcpds_cv_pykit.segmentation.datasets import OxfordIITPet

# Download and prepare the Oxford-IIIT Pet dataset
dataset_path = OxfordIITPet()
print(f"Dataset ready at: {dataset_path}")
```

### 2. Prepare Your Data Loaders

This toolkit includes ready-to-use, high-performance datasets and dataloader helpers for semantic segmentation, supporting both standard single/multi-class masks and multi-annotator scenarios.

#### Option A: Standard Segmentation (single or multi-class)

Use Segmentation_Dataset and Segmentation_DataLoader for typical datasets organized as:
- {data_dir}/{Partition}/images/*.png|jpg|jpeg
- {data_dir}/{Partition}/masks/class_0/*.png
- {data_dir}/{Partition}/masks/class_1/*.png
- ...
Notes:
- Masks are loaded per-class; missing masks are handled as empty (zeros).
- Augmentations (flips, rotations, color jitter, light noise) are applied only during training.
- Images are normalized to [0, 1] and resized to the target image_size.

Example:
```python
from gcpds_cv_pykit.segmentation.baseline.dataloaders import Segmentation_DataLoader

data_dir = "/path/to/dataset"
image_size = (256, 256)   # (H, W)
num_classes = 3
batch_size = 8

train_loader = Segmentation_DataLoader(
    data_dir=data_dir,
    batch_size=batch_size,
    image_size=image_size,
    num_classes=num_classes,
    partition="Train",
    single_class=None,          # set to an int (e.g., 0) to load only that class
    augment=True,               # augmentations only applied when partition == "Train"
    images_folder="images",     # custom images folder name (default: "images")
    num_workers=4,
    prefetch_factor=2,
    pin_memory=True,
)

val_loader = Segmentation_DataLoader(
    data_dir=data_dir,
    batch_size=batch_size,
    image_size=image_size,
    num_classes=num_classes,
    partition="Val",
    augment=False,
    images_folder="images",
    num_workers=4,
    pin_memory=True,
)

# Each batch returns:
#   images: FloatTensor [B, 3, H, W], in [0,1]
#   masks:  FloatTensor [B, C, H, W], binary per class
```

Key behaviors:
- Natural/alphanumeric file sorting ensures consistent pairing between images and masks.
- Supported image formats: .png, .jpg, .jpeg
- Missing mask files are treated as zeros (no annotation for that class).
- Rotation preserves mask binarization.

#### Option B: Multi-Annotator + Ground Truth (Annotator Harmony)

Use AnnotHarmonyDataset and AnnotHarmonyDataloader when you have multiple annotators per sample and optional ground truth. Directory structure:
- {data_dir}/{Partition}/patches/*.png
- {data_dir}/{Partition}/masks/{annotator_id}/class_{k}/*.png  (for each annotator and class)
- {data_dir}/{Partition}/masks/ground_truth/class_{k}/*.png    (optional GT)

Notes:
- Concatenates annotator masks along channel dimension as [num_annotators * num_classes, H, W].
- Returns a one-hot vector indicating which annotators provided a valid mask per sample.
- Handles missing annotator masks by filling with an ignored value (default 0.6), which is not treated as foreground/background.
- Supports training-time augmentations applied consistently to image, annotator masks, and ground truth.

Example:
```python
from gcpds_cv_pykit.segmentation.baseline.dataloaders import AnnotHarmonyDataloader

data_dir = "/path/to/harmony_dataset"
image_size = (256, 256)
num_classes = 3
num_annotators = 5
batch_size = 4

# Train with annotators + ground truth
train_loader = AnnotHarmonyDataloader(
    data_dir=data_dir,
    batch_size=batch_size,
    image_size=image_size,
    num_classes=num_classes,
    num_annotators=num_annotators,
    partition="Train",
    annotators=True,
    ground_truth=True,
    single_class=None,      # or an int to restrict to a single class
    augment=True,
    num_workers=4,
    prefetch_factor=2,
    pin_memory=True,
)

# Validation without augmentations
val_loader = AnnotHarmonyDataloader(
    data_dir=data_dir,
    batch_size=batch_size,
    image_size=image_size,
    num_classes=num_classes,
    num_annotators=num_annotators,
    partition="Val",
    annotators=True,
    ground_truth=True,
    augment=False,
    num_workers=4,
    pin_memory=True,
)

# Each batch returns (depending on annotators/ground_truth flags):
# - If annotators and ground_truth:
#     images:      [B, 3, H, W]
#     masks:       [B, num_annotators * num_classes, H, W]
#     anns_onehot: [B, num_annotators]  (1 if annotator present in sample, else 0)
#     gt:          [B, C, H, W]
# - If only annotators:
#     images, masks, anns_onehot
# - If only ground_truth:
#     images, gt
```

Tips:
- Set single_class to focus training on a specific class while preserving API consistency.
- Use num_workers > 0 with pin_memory=True for faster GPU input pipelines.
- For custom image folder names, use images_folder in Segmentation_DataLoader; AnnotHarmony expects images under patches by default.
- Augmentations include horizontal/vertical flips, small rotations, brightness/contrast/saturation jitter, and light Gaussian noise on images only.

### 3. Configure and Train a Model

You can train either a standard segmentation model (UNet/ResUNet/DeepLabV3+/FCN) or the Annotator Harmony model for multi-annotator learning. Both trainers support: phased fine-tuning, AMP, W&B logging, best/last checkpointing, and rich metric plots.

#### Option A: Baseline Segmentation (UNet, ResUNet, DeepLabV3+, FCN)

Trainer: `SegmentationModel_Trainer`

- Models: UNet, ResUNet, DeepLabV3+, FCN
- Losses: DICE, CrossEntropy, Focal, Tversky
- Phased training: progressively unfreeze encoder (phases 1→4)
- Metrics per-epoch: global/per-class DICE, IoU, Sensitivity, Specificity
- Visualizations every 5 epochs
- Saves: best_model.pt (by best Val DICE), last_model.pt
- Plots saved under results/experiment_X/

Example:

```python
from gcpds_cv_pykit.segmentation.baseline.trainers import SegmentationModel_Trainer

# Configuration dictionary
config = {
    # Model
    'Model': 'UNet',                      # 'UNet' | 'ResUNet' | 'DeepLabV3+' | 'FCN'
    'Backbone': 'resnet34',
    'Number of classes': 3,
    'Input size': [3, 256, 256],          # channels, H, W
    'Image size': (256, 256),             # used for plotting/metadata
    'Pretrained': True,
    'Activation function': None,          # None | 'sigmoid' (applied as model final_activation)

    # Loss
    'Loss function': 'DICE',              # 'DICE' | 'CrossEntropy' | 'Focal' | 'Tversky'
    'Smooth': 1.0,
    'Reduction': 'mean',
    'Alpha': 0.75,                        # Focal/Tversky
    'Beta': 0.3,                          # Tversky
    'Gamma': 2.0,                         # Focal

    # Training
    'Epochs': 50,
    'Device': 'cuda',                     # 'cuda' | 'cpu' | 'cuda:0', ...
    'AMixPre': True,                      # Automatic Mixed Precision (AMP)
    'Train phases': True,                 # phased fine-tuning (see below)
    'Single class train': None,           # int to train/evaluate a single class
    'Single class valid': None,

    # Monitoring
    'Wandb monitoring': None,             # or ['api_key', 'project', 'run_name']

    # Checkpoints/dirs (saved automatically)
    # models_dir = './models' (internal default)
}

# Initialize and start training
trainer = SegmentationModel_Trainer(
    train_loader=train_loader,            # yields (images, masks)
    valid_loader=valid_loader,            # yields (images, masks)
    config=config
)
trainer.start()                           # trains, logs, saves plots and checkpoints
```

Training phases (when 'Train phases' = True):
- Phase 1 (epochs 0–9): Freeze encoder, train decoder + segmentation_head (lr=1e-4)
- Phase 2 (10–19): + Unfreeze encoder BatchNorm layers (lr=1e-5)
- Phase 3 (20–29): + Unfreeze encoder layer4 (lr=1e-5)
- Phase 4 (30+): + Unfreeze encoder layer3 (lr=1e-5), scheduler ExponentialLR(gamma=0.94)

Notes:
- For Focal/CrossEntropy, predictions are sigmoid-ed internally for metric computation and visualizations.
- Visualizations sample random classes; for single-class mode, GT display adapts automatically.

#### Option B: Annotator Harmony (Multi-Annotator Learning)

Trainer: `AnnotHarmonyTrainer`

- Model: `AnnotHarmonyModel` (joint segmentation + annotator reliability)
- Input: images + stacked annotator masks + annotator presence one-hot (+ optional GT)
- Loss: TGCE_SS (robust to noisy annotations; supports ignored value)
- Flexible validation: with/without annotator masks and/or ground truth
- Metrics over GT when provided: global/per-class DICE, IoU, Sensitivity, Specificity
- Reliability map visualizations per annotator
- Saves: best_model.pt (by best Val DICE when GT available), last_model.pt
- Plots saved under results/experiment_X/

Example (with annotators + GT):

```python
from gcpds_cv_pykit.segmentation.baseline.trainers import AnnotHarmonyTrainer

config = {
    # Model
    'Model': 'AnnotHarmony',              # informational; model is constructed internally
    'Input size': [3, 256, 256],
    'Number of classes': 3,
    'Num of annotators': 5,
    'Activation seg': 'sparse_softmax',   # segmentation head activation inside model
    'Activation rel': 'softmax',          # reliability head activation

    # Loss (TGCE_SS)
    'Loss function': 'TGCE_SS',
    'Ignore value': 0.6,                  # value used to fill missing annotator masks
    'Q parameter': 0.7243854912956864,    # TGCE hyperparam

    # Training
    'Epochs': 50,
    'Device': 'cuda',
    'AMixPre': True,                      # AMP
    'Train phases': True,                 # phased fine-tuning similar to baseline
    'Single class train': None,           # int to focus metrics on one class
    'Single class valid': None,

    # Data flags
    'Ground truth train': True,           # train loader returns GT masks
    'Ground truth valid': True,           # valid loader returns GT masks
    'Annotators valid': True,             # valid loader returns annotator masks

    # Monitoring
    'Wandb monitoring': None,             # or ['api_key', 'project', 'run_name']
}

# Initialize and start training
trainer = AnnotHarmonyTrainer(
    train_loader=train_loader,            # yields (images, ann_masks, ann_onehot, gt) or (images, ann_masks, ann_onehot)
    valid_loader=valid_loader,            # supports (images, ann_masks, ann_onehot, gt), (images, ann_masks, ann_onehot), or (images, gt)
    config=config
)
trainer.start()
```

Validation/data combinations supported:
- annotators + ground truth: returns loss and metrics
- annotators only: returns loss
- ground truth only: returns metrics

Training phases (when 'Train phases' = True):
- Phase 1 (0–9): Freeze encoder; train decoder, seg_head, ann_rel (lr=1e-4)
- Phase 2 (10–19): + Unfreeze encoder BatchNorm (lr=1e-5)
- Phase 3 (20–29): + Unfreeze encoder layer4 (lr=1e-5)
- Phase 4 (30+): + Unfreeze encoder layer3 (lr=1e-5), scheduler ExponentialLR(gamma=0.94)

Outputs and logging (both trainers):
- Console: device info (CUDA, memory), per-epoch losses and metrics (global and per-class)
- Weights & Biases (optional): losses and metrics per epoch, prediction visualizations
- Files:
  - ./models/best_model.pt (best Val DICE when available)
  - ./models/last_model.pt
  - ./results/experiment_X/{Loss.png, DICE.png, Jaccard.png, Sensitivity.png, Specificity.png}

Tips:
- Set 'AMixPre': True for faster training with AMP on CUDA.
- Use 'Single class train/valid' to focus metrics on one class without changing labels.
- Enable 'Train phases' for stable fine-tuning of pretrained backbones.
- For W&B, set 'Wandb monitoring' to ['api_key', 'project', 'run_name'].

### 4. Evaluate Model Performance

Once training is complete, use the evaluation utilities to compute test-set metrics and optionally save detailed results. We provide two evaluators:

- Baseline evaluator for standard segmentation models: `PerformanceModels` (class)
- Annotator Harmony evaluator for multi-annotator models: `PerformanceAnnotHarmony` (function)

Both evaluators report global and per-class metrics:
- DICE (F1), Jaccard (IoU), Sensitivity (Recall), Specificity
- Mean ± std across the test set
- Optional .npy dumps of global and per-class metric arrays

#### 4.1 Baseline Segmentation Evaluator

Use `PerformanceModels` to evaluate UNet/ResUNet/DeepLabV3+/FCN models on a test DataLoader that yields (images, gt_masks).

Example:

```python
from gcpds_cv_pykit.segmentation.baseline import PerformanceModels
from torch.utils.data import DataLoader

# Build test loader
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

# Config notes:
# - Must include keys used during training, e.g. 'Device', 'Number of classes', 'Loss function'
# - Optional: 'AMixPre' (AMP), 'Single class test' (int), 'Ignored value' (float), 'Save results' (bool), 'drive_dir'
config.update({
    'Device': 'cuda:0',
    'Save results': True,         # to save .npy files
    'drive_dir': './',            # base folder for results
    # 'Single class test': 0,     # optionally restrict evaluation to a single class index
    # 'Ignored value': 0.6,       # pixels equal to this value are ignored in metrics
})

# Evaluate
evaluator = PerformanceModels(
    model=trainer.model,          # already-trained baseline model
    test_dataset=test_loader,     # DataLoader yielding (images, gt_masks)
    config=config
)

# Quick access to global means (convenience attributes)
print(f"Mean DICE Score: {evaluator.mean_dice:.4f}")
print(f"Mean Jaccard: {evaluator.mean_jjacard:.4f}")
```

What it does:
- Moves the model to config['Device'], switches to eval mode, and runs inference with or without AMP (config['AMixPre']).
- Applies sigmoid to predictions when evaluating CrossEntropy/Focal losses so metrics use probabilities.
- Thresholds predictions at 0.5 to compute discrete masks and then computes metrics per-batch and per-class.
- Ignores pixels that match config['Ignored value'] (default 0.6) when computing metrics.
- Prints global and per-class summaries; optionally saves .npy arrays under:
  - results/{Dataset}_{Model}_{Loss function}_DICE_global.npy, etc.
  - results/{Dataset}_{Model}_{Loss function}_DICE_class{c}.npy, etc.

Notes and tips:
- Single-class evaluation: set `'Single class test': class_index` to evaluate only that class.
- Ensure config['Number of classes'] matches the model output.
- If using Focal/CrossEntropy losses for training, the evaluator handles activation for fair metric computation.

#### 4.2 Annotator Harmony Evaluator (Multi-Annotator)

Use `PerformanceAnnotHarmony` when evaluating the Annotator Harmony model trained with TGCE_SS. The test DataLoader must yield batches like:
- (images, annotator_masks, annotator_presence_one_hot, gt_masks)

If ground-truth masks are available, the evaluator computes metrics against GT; it always computes the TGCE_SS loss against annotator masks.

Example:

```python
from gcpds_cv_pykit.segmentation.baseline import PerformanceAnnotHarmony
from torch.utils.data import DataLoader

# Build test loader for AnnotHarmonyDataset
# Each batch must be (images, ann_masks, ann_onehot, gt_masks)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

config.update({
    'Device': 'cuda:0',
    'Num of annotators': 5,
    'Number of classes': 3,
    'AMixPre': True,                 # AMP for faster inference
    'Ignored value': 0.6,            # ignore marker in GT
    'Q paramater': 0.7243854912956864,  # TGCE_SS q
    'Smooth': 1e-7,
    'Main_model': 'AnnotHarmony',
    'Dataset': 'MyDataset',
    'drive_dir': './',
})

# Evaluate; set save_results=True to dump .npy files
PerformanceAnnotHarmony(
    model=trainer.model,              # trained AnnotHarmonyModel
    test_dataset=test_loader,
    config=config,
    save_results=True
)
```

What it does:
- Runs the model with inputs (images, annotator_presence_one_hot).
- Computes TGCE_SS loss against annotator masks.
- Uses ground-truth masks (if present in the batch) to compute DICE, Jaccard, Sensitivity, Specificity.
- Supports optional single-class evaluation via `'Single class test': int`.
- Saves arrays to results/{Main_model}_{Dataset}_*.npy when save_results=True.

Expected saved files (when saving enabled):
- Global: Loss, Dice_global, Jaccard_global, Sensitivity_global, Specificity_global
- Per class: Dice_class{c}, Jaccard_class{c}, Sensitivity_class{c}, Specificity_class{c}

#### 4.3 Configuration Keys Reference

Common keys:
- Device: 'cuda', 'cuda:0', or 'cpu'
- Number of classes: int
- AMixPre: bool, use autocast AMP for speed on CUDA
- Single class test: int or None, restrict metrics to one class channel
- Ignored value: float, pixels equal to this in GT are excluded from metrics
- Smooth: float, epsilon for metric stability
- Save results: bool, save .npy arrays of metrics
- drive_dir: str, base directory for results/

Annotator Harmony-specific:
- Num of annotators: int
- Q paramater: float, TGCE_SS hyperparameter
- Main_model, Dataset: strings used in saved filenames

#### 4.4 Metric Definitions

Given per-pixel TP, FP, FN, TN and smoothing parameter s:
- DICE: (2·TP + s) / (2·TP + FP + FN + s)
- Jaccard (IoU): (TP + s) / (TP + FP + FN + s)
- Sensitivity (Recall): (TP + s) / (TP + FN + s)
- Specificity: (TN + s) / (TN + FP + s)

These are computed per-class, then aggregated.

#### 4.5 Troubleshooting

- All-zero predictions or masks: smoothing avoids NaNs; still verify class balance.
- Mismatched shapes: ensure DataLoader yields [B, C, H, W] masks with correct number of channels.
- Class indexing: for single-class tests, ensure the chosen class aligns with your dataset label mapping.
- AMP on CPU: set AMixPre=False if running on CPU (AMP is CUDA-optimized).
- CrossEntropy/Focal outputs: evaluator applies sigmoid before thresholding for fair comparison.

This evaluation section complements the training pipeline, giving consistent, reproducible metrics and optional artifacts for post-analysis.

---

## 🏗️ Architecture

```
gcpds_cv_pykit/
├── segmentation/
│   ├── baseline/
│   │   ├── models/          # Segmentation architectures
│   │   │   ├── UNet.py
│   │   │   ├── ResUNet.py
│   │   │   ├── DeepLabV3Plus.py
│   │   │   └── FCN.py
│   │   ├── losses/          # Loss functions
│   │   │   ├── DICE.py
│   │   │   ├── CrossEntropy.py
│   │   │   ├── Focal.py
│   │   │   └── Tversky.py
│   │   ├── trainers/        # Training pipeline
│   │   │   └── trainer.py
│   │   ├── dataloaders/     # Data loading utilities
│   │   │   └── dataloader.py
│   │   └── performance_model.py  # Evaluation metrics
│   ├── datasets/            # Dataset utilities
│   │   └── datasets.py
│   ├── visuals/             # Visualization tools
│   │   └── random_sample_visualizations.py
│   ├── crowd/               # Crowd-sourced annotation support
│   │   ├── models/
│   │   │   └── AnnotHarmony.py
│   │   ├── losses/          
│   │   │   └── TGCE_SS.py
│   │   ├── trainers/        
│   │   │   └── AnnotHarmonyTrainer.py
│   │   ├── dataloaders/    
│   │   │   └── annot_harmony_dataloader.py
│   │   └── performance/    
│   │       └── performance_annotharmony.py
└── _version.py
```

---

## 🧠 Available Models

### UNet
Classic U-Net architecture with encoder-decoder structure and skip connections.

```python
config = {
    'Model': 'UNet',
    'Number of classes': 3,
    'Image size': (256, 256),
}
```

**Features:**
- Symmetric encoder-decoder architecture
- Skip connections for feature preservation
- Customizable depth and filter sizes
- Pre-trained backbone support
- Batch normalization and dropout support

### ResUNet
U-Net with residual blocks for improved gradient flow.

```python
config = {
    'Model': 'ResUNet',
    'Number of classes': 3,
    'Backbone': 'resnet34',  # or mobilenetv3
}
```

**Features:**
- Residual connections in encoder blocks
- Pre-trained backbone support
- Better convergence for deep networks
- Reduced vanishing gradient problems

### DeepLabV3+
State-of-the-art model with atrous spatial pyramid pooling.

```python
config = {
    'Model': 'DeepLabV3Plus',
    'Number of classes': 3,
    'Backbone': 'resnet34',
}
```

**Features:**
- Atrous Spatial Pyramid Pooling (ASPP)
- Multi-scale feature extraction
- Encoder-decoder with atrous convolution
- Pre-trained backbone support
- Excellent for complex scenes

### FCN (Fully Convolutional Network)
Efficient fully convolutional architecture.

```python
config = {
    'Model': 'FCN',
    'Number of classes': 3,
    'Backbone': 'resnet34',
}
```

**Features:**
- End-to-end convolutional architecture
- No fully connected layers
- Fast inference
- Pre-trained backbone support
- Good baseline model

---

## 📉 Loss Functions

### DICE Loss
Optimizes the DICE coefficient directly, ideal for imbalanced segmentation.

```python
config = {
    'Loss function': 'DICE',
    'Smooth': 1.0,
    'Reduction': 'mean',
}
```

### Cross-Entropy Loss
Standard pixel-wise classification loss.

```python
config = {
    'Loss function': 'CrossEntropy',
    'Reduction': 'mean',
}
```

### Focal Loss
Addresses class imbalance by down-weighting easy examples.

```python
config = {
    'Loss function': 'Focal',
    'Alpha': 0.25,
    'Gamma': 2.0,
    'Reduction': 'mean',
}
```

### Tversky Loss
Allows control over false positives and false negatives trade-off.

```python
config = {
    'Loss function': 'Tversky',
    'Alpha': 0.5,  # Weight for false positives
    'Beta': 0.5,   # Weight for false negatives
    'Smooth': 1.0,
}
```

---

## 🗂️ Datasets

### Built-in Dataset Support

The toolkit provides easy access to several pre-configured datasets from Kaggle:

```python
from gcpds_cv_pykit.segmentation.datasets import (
    OxfordIITPet,
    SeedGermination,
    BreastCancer,
    FeetMamitas,
    OxfordIITPet_Crowd,
    BreastCancer_Crowd
)

# Download datasets
oxford_path = OxfordIITPet()
seeds_path = SeedGermination()
cancer_path = BreastCancer()
feet_path = FeetMamitas()

# Crowd-sourced annotation datasets
oxford_crowd_path = OxfordIITPet_Crowd()
cancer_crowd_path = BreastCancer_Crowd()
```

### Custom Dataset Integration

You can easily integrate your own datasets:

```python
from torch.utils.data import Dataset

class CustomSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Your initialization code
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.load_image(idx)
        mask = self.load_mask(idx)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        return image, mask
```

---

## 💡 Usage Examples

### Complete Training Pipeline

```python
import torch
from torch.utils.data import DataLoader
from gcpds_cv_pykit.segmentation.baseline.trainers import SegmentationModel_Trainer
from gcpds_cv_pykit.segmentation.datasets import OxfordIITPet

# 1. Download dataset
dataset_path = OxfordIITPet()

# 2. Create data loaders (assuming you have a dataset class)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=4)

# 3. Configure training
config = {
    # Model configuration
    'Model': 'ResUNet',
    'Number of classes': 3,
    'Image size': (256, 256),
    'Backbone': 'resnet34',
    'Activation function': 'relu',
    
    # Training configuration
    'Loss function': 'DICE',
    'Learning rate': 0.001,
    'Optimizer': 'Adam',
    'Epochs': 100,
    'Batch size': 16,
    
    # Device and performance
    'Device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'Mixed precision': True,
    'Num workers': 4,
    
    # Regularization
    'Weight decay': 1e-5,
    'Dropout': 0.2,
    
    # Early stopping
    'Early stopping': True,
    'Patience': 15,
    
    # Model saving
    'Save model': True,
    'Model name': 'resunet_oxford_pets',
    'Save path': './checkpoints/',
    
    # Monitoring
    'Verbose': True,
}

# 4. Initialize and train
trainer = SegmentationModel_Trainer(
    train_loader=train_loader,
    valid_loader=valid_loader,
    config=config
)

# 5. Start training
trainer.train()

# 6. Access trained model
model = trainer.model
```

### Training with Weights & Biases

```python
# Add WandB configuration
config['WandB monitoring'] = [
    'your_wandb_api_key',
    'project_name',
    'experiment_name'
]

trainer = SegmentationModel_Trainer(
    train_loader=train_loader,
    valid_loader=valid_loader,
    config=config
)

trainer.train()
```

### Visualization

```python
from gcpds_cv_pykit.segmentation.visuals import random_sample_visualizations

# Visualize random samples from dataset
random_sample_visualizations(
    dataset=train_dataset,
    num_samples=5,
    save_path='./visualizations/'
)
```

---

## ⚙️ Configuration

### Complete Configuration Reference

```python
config = {
    # ============ Model Architecture ============
    'Model': 'UNet',  # Options: 'UNet', 'ResUNet', 'DeepLabV3Plus', 'FCN'
    'Number of classes': 3,
    'Image size': (256, 256),
    'Backbone': 'resnet34',  # For ResUNet, DeepLabV3+, FCN
    'Activation function': 'relu',  # Options: 'sigmoid', 'softmax'
    'Pretrained': True,  # Use pretrained backbone
    
    # ============ Loss Function ============
    'Loss function': 'DICE',  # Options: 'DICE', 'CrossEntropy', 'Focal', 'Tversky'
    'Smooth': 1.0,  # For DICE and Tversky
    'Alpha': 0.25,  # For Focal and Tversky
    'Beta': 0.5,   # For Tversky
    'Gamma': 2.0,  # For Focal
    'Reduction': 'mean',  # Options: 'mean', 'sum', 'none'
    
    # ============ Training ============
    'Epochs': 100,
    'Batch size': 16,
    'Num workers': 4,
    'Pin memory': True,
    
    # ============ Device & Performance ============
    'Device': 'cuda',  # Options: 'cuda', 'cpu', 'cuda:0', 'cuda:1'
    'Mixed precision': True,  # Use AMP for faster training
    'Gradient clipping': 1.0,  # Max gradient norm
    
    # ============ Model Saving ============
    'Save model': True,
    'Model name': 'my_segmentation_model',
    'Save path': './checkpoints/',
    
    # ============ Monitoring ============
    'Print frequency': 10,  # Print every N batches
    'WandB monitoring': None,  # Or ['api_key', 'project', 'run_name']
    
    # ============ Evaluation ============
    'Save results': True,
    'Results path': './results/',
    'Results format': 'npz',
}
```

---

## 📊 Experiment Tracking

### Weights & Biases Integration

The toolkit seamlessly integrates with Weights & Biases for experiment tracking:

```python
# Configure WandB
config['WandB monitoring'] = [
    'your_api_key',      # Your WandB API key
    'project_name',      # Project name
    'experiment_name'    # Run name
]

# Training will automatically log:
# - Training and validation loss
# - Learning rate changes
# - Model architecture
# - Hyperparameters
# - System metrics (GPU usage, etc.)
# - Sample predictions (if configured)
```

### Logged Metrics

- **Per Epoch**: Loss, DICE score, IoU, learning rate
- **Per Batch**: Training loss, batch processing time
- **System**: GPU memory usage, CPU usage
- **Model**: Parameter count, model architecture

---

## 📈 Performance Evaluation

### Available Metrics

The `PerformanceModels` class calculates:

- **DICE Coefficient**: Overlap between prediction and ground truth
- **Jaccard Index (IoU)**: Intersection over Union
- **Sensitivity (Recall)**: True positive rate
- **Specificity**: True negative rate

### Metrics Calculation

```python
evaluator = PerformanceModels(model, test_loader, config)

# Global metrics (averaged across all classes)
print(f"Global DICE: {evaluator.global_dice:.4f}")
print(f"Global IoU: {evaluator.global_iou:.4f}")

# Per-class metrics
for i, (dice, iou) in enumerate(zip(evaluator.per_class_dice, evaluator.per_class_iou)):
    print(f"Class {i} - DICE: {dice:.4f}, IoU: {iou:.4f}")

# Confusion matrix
print(evaluator.confusion_matrix)
```

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute

- 🐛 Report bugs and issues
- 💡 Suggest new features or improvements
- 📝 Improve documentation
- 🔧 Submit pull requests
- ⭐ Star the repository

### Development Setup

```bash
# Clone the repository
git clone https://github.com/UN-GCPDS/gcpds-cv-pykit.git
cd gcpds-cv-pykit

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Check code style
black gcpds_cv_pykit/
flake8 gcpds_cv_pykit/
isort gcpds_cv_pykit/
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 SPRG - GCPDS Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 📚 Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{gcpds_cv_pykit,
  title = {GCPDS Computer Vision Python Kit},
  author = {GCPDS Team},
  year = {2025},
  url = {https://github.com/UN-GCPDS/gcpds-cv-pykit},
  version = {0.1.0}
}
```

---

## 📞 Contact

**GCPDS Team** - Universidad Nacional de Colombia

- 📧 Email: gcpds_man@unal.edu.co
- 🌐 GitHub: [UN-GCPDS](https://github.com/UN-GCPDS)
- 📖 Documentation: [https://gcpds-cv-pykit.readthedocs.io/](https://gcpds-cv-pykit.readthedocs.io/)
- 🐛 Issues: [GitHub Issues](https://github.com/UN-GCPDS/gcpds-cv-pykit/issues)

---

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- The computer vision research community
- Contributors and users of this toolkit
- Universidad Nacional de Colombia

---

<div align="center">

**Made with ❤️ by the GCPDS Team**

[⬆ Back to Top](#gcpds-computer-vision-python-kit)

</div>
