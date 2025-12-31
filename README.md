# Retinal Disease Classification with CNN and GLCM Fusion

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive deep learning framework for retinal disease classification using state-of-the-art CNN architectures combined with GLCM (Gray-Level Co-occurrence Matrix) texture features. This project implements multiple approaches for Optical Coherence Tomography (OCT) image analysis and diabetic retinopathy detection.

## Key Features

- **FastViT Architecture**: State-of-the-art vision transformer for 8-class OCT classification
- **CNN+GLCM Fusion**: Novel approach combining deep features with texture analysis
- **Hyperparameter Optimization**: Automated tuning using Optuna
- **Comprehensive Evaluation**: Advanced metrics and visualization tools
- **Class Imbalance Handling**: Weighted sampling and balanced training
- **Production Ready**: Complete training pipeline with checkpointing

## Medical Applications

### OCT 8-Class Classification (FastViT)
- **Classes**: DR, NORMAL, DME, AMD, CNV, DRUSEN, MH, CSR
- **Architecture**: FastViT (T8/T12/S12/SA12/SA24/SA36/MA36)
- **Features**: Advanced data augmentation, class balancing, early stopping

### Diabetic Retinopathy Classification (CNN + GLCM)
- **Architectures**: DenseNet121, ResNet50, EfficientNet-B3
- **Features**: GLCM texture features (12-dimensional), Optuna optimization
- **Input**: Folder-based dataset structure with automatic preprocessing

## Project Structure

```
retinal-disease-classification/
├── README.md                    # Project documentation
├── requirements.txt             # Dependencies
├── demo_synthetic.py           # Complete demo with synthetic data
├── example_usage.py            # Usage examples
├── config/                     # Configuration files
│   ├── oct_config.py              # OCT 8-class configuration
│   └── dr_config.py               # DR classification configuration
├── src/                        # Source code
│   ├── data/                   # Data handling modules
│   │   ├── oct_dataset.py         # OCT dataset processing
│   │   └── dr_dataset.py          # DR dataset with GLCM features
│   ├── models/                 # Model definitions
│   │   ├── fastvit_model.py       # FastViT wrapper
│   │   └── cnn_glcm_models.py     # CNN+GLCM architectures
│   ├── features/               # Feature extraction
│   │   └── glcm_extractor.py      # GLCM texture analysis
│   ├── training/               # Training modules
│   │   ├── oct_trainer.py         # OCT model trainer
│   │   └── dr_trainer.py          # DR model trainer
│   └── utils/                  # Utilities
│       ├── visualization.py       # Plotting and visualization
│       └── metrics.py             # Evaluation metrics
├── scripts/                    # Training scripts
│   ├── train_oct_fastvit.py       # OCT training pipeline
│   └── train_dr_models.py         # DR training pipeline
└── outputs/                    # Generated files
    ├── models/                    # Saved model checkpoints
    ├── plots/                     # Training visualizations
    └── logs/                      # Training logs
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Devrajj-14/Retinal-Disease-Classification-with-CNN-and-GLCM-Fusion.git
cd Retinal-Disease-Classification-with-CNN-and-GLCM-Fusion

# Install dependencies
pip install -r requirements.txt
```

### Demo with Synthetic Data

```bash
# Run complete demo to test all components
python demo_synthetic.py
```

### Training Models

#### OCT 8-Class Classification
```bash
python scripts/train_oct_fastvit.py \
    --data_dir /path/to/oct/dataset \
    --model_name fastvit_t8 \
    --epochs 30 \
    --batch_size 32
```

#### Diabetic Retinopathy Classification
```bash
# With hyperparameter optimization
python scripts/train_dr_models.py \
    --data_dir /path/to/dr/dataset \
    --model densenet121 \
    --epochs 50 \
    --n_trials 10

# Skip optimization for faster training
python scripts/train_dr_models.py \
    --data_dir /path/to/dr/dataset \
    --model resnet50 \
    --skip_optimization
```

## Technical Details

### FastViT Models
- **FastViT-T8**: ~3.3M parameters, optimized for speed
- **FastViT-SA36**: Higher accuracy, more parameters
- **Input**: 224×224 RGB images
- **Output**: 8-class probability distribution

### CNN+GLCM Architecture
- **DenseNet121**: 1024 CNN features + 12 GLCM features
- **ResNet50**: 2048 CNN features + 12 GLCM features  
- **EfficientNet-B3**: 1536 CNN features + 12 GLCM features (300×300 input)

### GLCM Features
- **Distances**: [1, 2] pixels
- **Angles**: [0°, 45°, 90°, 135°]
- **Properties**: Contrast, Homogeneity, Energy (mean & std)
- **Total**: 12-dimensional feature vector

## Performance

The models achieve state-of-the-art performance on retinal disease classification:

- **FastViT**: Efficient transformer architecture with competitive accuracy
- **CNN+GLCM**: Enhanced performance through texture feature fusion
- **Optuna Optimization**: Automated hyperparameter tuning for optimal results

## Configuration

Update configuration files to match your setup:

```python
# config/oct_config.py
BASE_DATA_DIR = "/path/to/your/oct/dataset"
TARGET_CLASSES = ['DR', 'NORMAL', 'DME', 'AMD', 'CNV', 'DRUSEN', 'MH', 'CSR']

# config/dr_config.py  
DATA_DIR = "/path/to/your/dr/dataset"
```

## Visualization

The framework includes comprehensive visualization tools:

- Training curves (loss, accuracy)
- Confusion matrices
- Per-class performance metrics
- Dataset distribution analysis
- ROC curves (when applicable)

## Advanced Usage

### Custom Model Training

```python
from src.models.cnn_glcm_models import get_model
from src.training.dr_trainer import DRTrainer

# Create custom model
model = get_model('densenet121', n_classes=5, glcm_size=12)

# Initialize trainer with custom parameters
trainer = DRTrainer('densenet121', 5, best_params)
```

### Feature Extraction

```python
from src.features.glcm_extractor import GLCMFeatureExtractor

# Extract GLCM features
extractor = GLCMFeatureExtractor()
features = extractor.extract_glcm_features(image)
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- timm (for FastViT models)
- scikit-learn
- scikit-image
- matplotlib, seaborn
- optuna (for hyperparameter optimization)
- PIL/Pillow
- pandas, numpy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- FastViT architecture from the timm library
- GLCM implementation using scikit-image
- Optuna for hyperparameter optimization
- PyTorch team for the deep learning framework

## Contact

For questions or collaborations, please open an issue or contact the maintainer.

---

**Star this repository if you find it useful!**