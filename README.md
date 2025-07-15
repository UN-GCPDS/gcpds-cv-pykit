# GCPDS Computer Vision Python Kit

A comprehensive toolkit for computer vision and segmentation tasks, developed by the GCPDS Team. This package provides state-of-the-art tools for training, evaluating, and deploying segmentation models with support for various architectures, loss functions, and performance metrics.

## 🚀 Features

- **Segmentation Models**: Support for UNet, ResUNet, DeepLabV3Plus, and FCN architectures
- **Multiple Loss Functions**: DICE, Cross Entropy, Focal Loss, and Tversky Loss
- **Performance Evaluation**: Comprehensive metrics including Dice, Jaccard, Sensitivity, and Specificity
- **Training Pipeline**: Complete training workflow with validation and monitoring
- **Data Loading**: Efficient data loading utilities for segmentation tasks
- **Experiment Tracking**: Integration with Weights & Biases (wandb)
- **Mixed Precision Training**: Automatic Mixed Precision (AMP) support for faster training
- **Dataset Management**: Built-in Kaggle dataset download and preparation utilities
- **Visualization Tools**: Random sample visualization utilities for dataset exploration
- **Memory Management**: Efficient memory handling and cleanup utilities

## 📋 Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA-compatible GPU (recommended)

## 🔧 Installation

### From PyPI
```bash
pip install gcpds-cv-pykit
```

### From Source
```bash
git clone https://github.com/UN-GCPDS/gcpds-cv-pykit.git
cd gcpds-cv-pykit
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/UN-GCPDS/gcpds-cv-pykit.git
cd gcpds-cv-pykit
pip install -e ".[dev,docs]"
```

## 📦 Dependencies

### Core Dependencies
- `torch>=2.0.0` - Deep learning framework
- `torchvision>=0.15.0` - Computer vision utilities
- `numpy>=1.21.0` - Numerical computing
- `opencv-python>=4.6.0` - Image processing
- `matplotlib>=3.5.0` - Plotting and visualization
- `wandb>=0.15.0` - Experiment tracking
- `tqdm>=4.64.0` - Progress bars
- `Pillow>=9.0.0` - Image handling
- `scipy>=1.9.0` - Scientific computing
- `pandas>=1.4.0` - Data manipulation
- `kagglehub` - Kaggle dataset downloads

### Optional Dependencies
- **Development**: `pytest>=7.0.0`, `pytest-cov>=4.0.0`, `black>=22.0.0`, `flake8>=5.0.0`, `isort>=5.10.0`
- **Documentation**: `sphinx>=5.0.0`, `sphinx-rtd-theme>=1.0.0`

## 🏗️ Project Structure

```
gcpds_cv_pykit/
├── baseline/
│   ├── trainers/           # Training utilities
│   │   ├── trainer.py      # Main training class
│   │   └── __init__.py
│   ├── models/             # Model architectures
│   │   ├── UNet.py         # U-Net implementation
│   │   ├── ResUNet.py      # Residual U-Net
│   │   ├── DeepLabV3Plus.py # DeepLab v3+ implementation
│   │   ├── FCN.py          # Fully Convolutional Network
│   │   └── __init__.py
│   ├── losses/             # Loss functions
│   │   ├── DICE.py         # DICE loss implementation
│   │   ├── CrossEntropy.py # Cross entropy loss
│   │   ├── Focal.py        # Focal loss implementation
│   │   ├── Tversky.py      # Tversky loss implementation
│   │   └── __init__.py
│   ├── dataloaders/        # Data loading utilities
│   │   ├── dataloader.py   # Custom data loading implementations
│   │   └── __init__.py
│   └── performance_model.py # Model evaluation and performance metrics
├── crowd/                  # Crowd-specific implementations (under development)
├── datasets/               # Dataset utilities and Kaggle integration
│   ├── datasets.py         # Kaggle dataset download and preparation
│   └── __init__.py
├── visuals/               # Visualization tools
│   ├── random_sample_visualizations.py  # Dataset visualization utilities
│   └── __init__.py
├── _version.py            # Version information
└── __init__.py
```

## 🚀 Quick Start

### Dataset Download and Preparation

```python
from gcpds_cv_pykit.datasets import download_and_prepare_dataset

# Download a Kaggle dataset
dataset_path = download_and_prepare_dataset('username/dataset-name/versions/1')
print(f"Dataset prepared at: {dataset_path}")
```

### Dataset Visualization

```python
from gcpds_cv_pykit.visuals import random_sample_visualization
from torch.utils.data import DataLoader

# Visualize random samples from your dataset
random_sample_visualization(
    dataset=your_dataloader,
    num_classes=2,
    single_class=None,  # Show all classes
    max_classes_to_show=7,
    type="baseline"
)
```

### Model Training

```python
from gcpds_cv_pykit.baseline.trainers import Trainer
from gcpds_cv_pykit.baseline.models import UNet

# Initialize model
model = UNet(in_channels=3, out_channels=2, pretrained=True)

# Configure training
config = {
    'Device': 'cuda',
    'Loss Function': 'DICE',
    'Number of classes': 2,
    'Learning Rate': 0.001,
    'Epochs': 100,
    'Batch Size': 8,
    # ... other configuration parameters
}

# Initialize trainer
trainer = Trainer(
    model=model,
    train_dataset=train_dataloader,
    val_dataset=val_dataloader,
    config=config
)

# Start training
trainer.train()
```

### Model Performance Evaluation

```python
from gcpds_cv_pykit.baseline import PerformanceModels

# Evaluate trained model
config = {
    'Device': 'cuda',
    'Loss Function': 'DICE',
    'Number of classes': 2,
    # ... other configuration parameters
}

evaluator = PerformanceModels(
    model=trained_model,
    test_dataset=test_dataloader,
    config=config
)
```

## 📊 Supported Models

The toolkit provides several state-of-the-art segmentation model architectures, all with ResNet34 backbone support:

### **UNet**
- **Architecture**: Classic U-Net with ResNet34 encoder
- **Features**:
  - Skip connections for precise localization
  - Pretrained ResNet34 backbone support
  - Configurable encoder/decoder channels
  - Optional final activation functions (sigmoid, softmax, tanh)
  - Bilinear interpolation for upsampling
- **Use Case**: General-purpose segmentation tasks

### **ResUNet**
- **Architecture**: Residual U-Net with ResNet34 backbone
- **Features**:
  - Enhanced skip connections with residual blocks
  - ResNet34 pretrained encoder
  - Improved gradient flow through residual connections
  - Batch normalization and ReLU activations
- **Use Case**: Complex segmentation tasks requiring deeper feature learning

### **DeepLabV3Plus**
- **Architecture**: DeepLab v3+ with Atrous Spatial Pyramid Pooling (ASPP)
- **Features**:
  - ASPP module for multi-scale feature extraction
  - Separable convolutions for efficiency
  - ResNet34 backbone with dilated convolutions
  - Low-level feature fusion
  - Configurable atrous rates
- **Use Case**: High-resolution segmentation with multi-scale context

### **FCN (Fully Convolutional Network)**
- **Architecture**: FCN with ResNet34 backbone
- **Features**:
  - Multi-scale skip connections (FCN-8s, FCN-16s, FCN-32s style)
  - Transposed convolutions for upsampling
  - ResNet34 pretrained encoder
  - Feature fusion at multiple scales
- **Use Case**: Semantic segmentation with multi-scale feature integration

### **Common Model Features**
- **Backbone**: ResNet34 with ImageNet pretrained weights
- **Input Channels**: Configurable (default: 3 for RGB)
- **Output Channels**: Configurable number of classes
- **Activation Functions**: Support for sigmoid, softmax, tanh, or none
- **Mixed Precision**: Compatible with AMP training
- **Memory Efficient**: Optimized for GPU memory usage

### **Model Usage Example**
```python
from gcpds_cv_pykit.baseline.models import UNet, ResUNet, DeepLabV3Plus, FCN

# UNet with default settings
model = UNet(
    in_channels=3,
    out_channels=2,  # Binary segmentation
    pretrained=True,
    final_activation='sigmoid'
)

# DeepLabV3Plus for multi-class segmentation
model = DeepLabV3Plus(
    in_channels=3,
    out_channels=5,  # 5-class segmentation
    pretrained=True,
    final_activation='softmax'
)

# ResUNet for complex segmentation
model = ResUNet(
    in_channels=3,
    out_channels=1,
    pretrained=True
)

# FCN for semantic segmentation
model = FCN(
    in_channels=3,
    out_channels=10,
    pretrained=True,
    final_activation='softmax'
)
```

## 🎯 Loss Functions

The following loss functions are available through the baseline.losses module:

- **DICE Loss**: Optimized for segmentation tasks with class imbalance
- **Cross Entropy**: Standard classification loss for multi-class segmentation
- **Focal Loss**: Addresses class imbalance by focusing on hard examples
- **Tversky Loss**: Generalization of Dice loss with configurable precision/recall balance

### **Loss Function Usage**
```python
from gcpds_cv_pykit.baseline.losses import DICELoss, CrossEntropyLoss, FocalLoss, TverskyLoss

# DICE Loss for binary segmentation
dice_loss = DICELoss()

# Cross Entropy for multi-class segmentation
ce_loss = CrossEntropyLoss()

# Focal Loss for handling class imbalance
focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

# Tversky Loss with custom alpha/beta
tversky_loss = TverskyLoss(alpha=0.3, beta=0.7)
```

## 📈 Metrics

The toolkit provides comprehensive evaluation metrics through the PerformanceModels class:

- **Dice Coefficient**: Overlap-based similarity measure
- **Jaccard Index (IoU)**: Intersection over Union
- **Sensitivity (Recall)**: True positive rate
- **Specificity**: True negative rate

All metrics are calculated both globally and per-class with detailed statistical analysis.

## 🔧 Configuration

The toolkit uses dictionary-based configuration. Key parameters include:

```python
config = {
    # Model Configuration
    'Model': 'UNet',
    'Backbone': 'resnet34',
    'Pretrained': True,
    'Number of classes': 2,
    'Input size': [3, 256, 256],
    
    # Training Configuration
    'Loss Function': 'DICE',
    'Optimizer': 'Adam',
    'Learning Rate': 0.001,
    'Epochs': 100,
    'Batch Size': 8,
    
    # Advanced Options
    'AMP': True,  # Automatic Mixed Precision
    'Device': 'cuda',
    'Wandb monitoring': ['api_key', 'project_name', 'run_name']
}
```

## 📊 Experiment Tracking

Integration with Weights & Biases for experiment tracking:

```python
config['Wandb monitoring'] = [
    'your_wandb_api_key',
    'your_project_name',
    'experiment_name'
]
```

## 🎨 Visualization

Built-in visualization tools for:
- Random dataset sample visualization
- Multi-class segmentation mask display
- Training/validation curves (through wandb integration)
- Model predictions vs ground truth

Example usage:
```python
from gcpds_cv_pykit.visuals import random_sample_visualization

# Visualize a single class
random_sample_visualization(
    dataset=dataloader,
    num_classes=5,
    single_class=1,  # Show only class 1
    type="baseline"
)

# Visualize multiple classes
random_sample_visualization(
    dataset=dataloader,
    num_classes=5,
    max_classes_to_show=3,
    type="baseline"
)
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gcpds_cv_pykit

# Run specific test file
pytest tests/test_models.py
```

## 📚 Documentation

Build documentation locally:

```bash
cd docs
make html
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone the repository
git clone https://github.com/UN-GCPDS/gcpds-cv-pykit.git
cd gcpds-cv-pykit

# Install in development mode
pip install -e ".[dev]"

# Run code formatting
black gcpds_cv_pykit/
isort gcpds_cv_pykit/

# Run linting
flake8 gcpds_cv_pykit/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **GCPDS Team** - [gcpds_man@unal.edu.co](mailto:gcpds_man@unal.edu.co)

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- The computer vision community for inspiration and best practices
- Kaggle for providing accessible datasets through kagglehub
- Contributors and users of this toolkit

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/UN-GCPDS/gcpds-cv-pykit/issues)
- **Documentation**: [Read the Docs](https://gcpds-cv-pykit.readthedocs.io/)
- **Email**: gcpds_man@unal.edu.co

## 🔄 Changelog

### Version 0.1.0.39 (Current)
- Complete segmentation model implementations (UNet, ResUNet, DeepLabV3Plus, FCN)
- Full loss function suite (DICE, CrossEntropy, Focal, Tversky)
- Training pipeline with Trainer class
- Data loading utilities for segmentation tasks
- Dataset download and preparation utilities via Kaggle integration
- Random sample visualization tools for dataset exploration
- Performance evaluation framework with comprehensive metrics
- Mixed precision training support
- Weights & Biases integration for experiment tracking

### Upcoming Features
- Additional model architectures (Attention U-Net, U-Net++)
- Advanced data augmentation utilities
- Model ensemble capabilities
- Extended visualization and analysis tools
- Pre-trained model zoo

---

**Note**: This project is actively maintained and regularly updated. The API is stable for the current feature set. Please check the documentation and changelog for the latest updates and new features.