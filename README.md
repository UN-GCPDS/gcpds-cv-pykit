# GCPDS Computer Vision Python Kit

A comprehensive toolkit for computer vision and segmentation tasks, developed by the GCPDS Team. This package provides state-of-the-art tools for training, evaluating, and deploying segmentation models with support for various architectures, loss functions, and performance metrics.

## 🚀 Features

- **Segmentation Models**: Support for UNet and other popular architectures
- **Multiple Loss Functions**: DICE, Cross Entropy, Focal Loss, and Tversky Loss
- **Performance Evaluation**: Comprehensive metrics including Dice, Jaccard, Sensitivity, and Specificity
- **Training Pipeline**: Complete training workflow with validation and monitoring
- **Experiment Tracking**: Integration with Weights & Biases (wandb)
- **Mixed Precision Training**: Automatic Mixed Precision (AMP) support for faster training
- **Flexible Configuration**: YAML/JSON-based configuration system
- **Visualization Tools**: Built-in visualization utilities for model predictions
- **Memory Management**: Efficient memory handling and cleanup utilities

## 📋 Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA-compatible GPU (recommended)

## 🔧 Installation

### From PyPI (when available)
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

### Optional Dependencies
- **Development**: `pytest`, `black`, `flake8`, `isort`
- **Documentation**: `sphinx`, `sphinx-rtd-theme`

## 🏗️ Project Structure

```
gcpds_cv_pykit/
├── baseline/
│   ├── trainers/           # Training utilities
│   ├── models/             # Model architectures
│   ├── losses/             # Loss functions
│   ├── dataloaders/        # Data loading utilities
│   └── performance_model.py # Model evaluation
├── crowd/                  # Crowd-specific implementations
├── datasets/               # Dataset utilities
└── visuals/               # Visualization tools
```

## 🚀 Quick Start

### Basic Training Example

```python
from gcpds_cv_pykit.baseline.trainers import SegmentationModel_Trainer
from torch.utils.data import DataLoader

# Define your configuration
config = {
    'Model': 'UNet',
    'Backbone': 'resnet34',
    'Number of classes': 2,
    'Loss Function': 'DICE',
    'Optimizer': 'Adam',
    'Learning Rate': 0.001,
    'Epochs': 100,
    'Batch Size': 8,
    'Input size': [3, 256, 256],
    'AMP': True,
    'Device': 'cuda'
}

# Initialize trainer
trainer = SegmentationModel_Trainer(
    train_loader=train_dataloader,
    valid_loader=valid_dataloader,
    config=config
)

# Start training
trainer.start()
```

### Model Evaluation Example

```python
from gcpds_cv_pykit.baseline import PerformanceModels

# Evaluate trained model
evaluator = PerformanceModels(
    model=trained_model,
    test_dataset=test_dataloader,
    config=config,
    save_results=True
)
```

### Custom Loss Function

```python
from gcpds_cv_pykit.baseline.losses import DICELoss, FocalLoss

# DICE Loss
dice_loss = DICELoss(smooth=1.0, reduction='mean')

# Focal Loss
focal_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
```

## 📊 Supported Models

- **UNet**: Classic U-Net architecture with various backbone options
  - Backbones: ResNet, EfficientNet, and more
  - Pretrained weights support
  - Customizable activation functions

## 🎯 Loss Functions

- **DICE Loss**: Optimized for segmentation tasks
- **Cross Entropy**: Standard classification loss
- **Focal Loss**: Addresses class imbalance
- **Tversky Loss**: Generalization of Dice loss

## 📈 Metrics

The toolkit provides comprehensive evaluation metrics:

- **Dice Coefficient**: Overlap-based similarity measure
- **Jaccard Index (IoU)**: Intersection over Union
- **Sensitivity (Recall)**: True positive rate
- **Specificity**: True negative rate

All metrics are calculated both globally and per-class.

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
- Training/validation curves
- Model predictions vs ground truth
- Metric evolution across epochs
- Sample predictions

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

- **GCPDS Team** - *Initial work* - [gcpds_man@unal.edu.co](mailto:gcpds_man@unal.edu.co)

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- The computer vision community for inspiration and best practices
- Contributors and users of this toolkit

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/UN-GCPDS/gcpds-cv-pykit/issues)
- **Documentation**: [Read the Docs](https://gcpds-cv-pykit.readthedocs.io/)
- **Email**: your-email@example.com

## 🔄 Changelog

### Version 0.1.0 (Alpha)
- Initial release
- Basic UNet implementation
- Core loss functions
- Training and evaluation pipeline
- Weights & Biases integration

---

**Note**: This project is in active development. APIs may change between versions. Please check the documentation for the latest updates.
