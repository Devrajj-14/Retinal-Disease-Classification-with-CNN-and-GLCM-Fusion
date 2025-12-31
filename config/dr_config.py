"""Configuration for Diabetic Retinopathy classification using CNN + GLCM"""

import os

class DRConfig:
    # Dataset configuration
    DATA_DIR = '/kaggle/input/diabetic-retinopathy-224x224-2019-data/colored_images'
    
    # Model configurations
    AVAILABLE_MODELS = {
        'densenet121': {
            'feature_size': 1024,
            'image_size': 224,
            'batch_size': 16
        },
        'resnet50': {
            'feature_size': 2048,
            'image_size': 224,
            'batch_size': 16
        },
        'efficientnet_b3': {
            'feature_size': 1536,
            'image_size': 300,
            'batch_size': 12
        }
    }
    
    # GLCM feature extraction
    GLCM_CONFIG = {
        'distances': [1, 2],
        'angles': [0, 45, 90, 135],
        'levels': 32,
        'feature_size': 12  # 6 features per distance
    }
    
    # Data splitting
    VAL_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Training configuration
    EPOCHS = 50
    LEARNING_RATE_RANGE = (1e-5, 1e-2)
    MOMENTUM_RANGE = (0.5, 0.99)
    WEIGHT_DECAY_RANGE = (1e-5, 1e-3)
    
    # Dropout ranges for optimization
    DROPOUT_RANGES = {
        'dropout_1': (0.1, 0.4),
        'dropout_2': (0.2, 0.5),
        'dropout_3': (0.3, 0.6)
    }
    
    # Optuna configuration
    N_TRIALS = 10
    OPTIMIZATION_TIMEOUT = 1800  # 30 minutes
    
    # Early stopping
    PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5
    
    # Paths
    OUTPUT_DIR = "outputs"
    MODEL_SAVE_PATHS = {
        'densenet121': os.path.join(OUTPUT_DIR, "models", "best_densenet_glcm_model.pth"),
        'resnet50': os.path.join(OUTPUT_DIR, "models", "best_resnet50_glcm_model.pth"),
        'efficientnet_b3': os.path.join(OUTPUT_DIR, "models", "best_efficientnet_b3_glcm_model.pth")
    }
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
    LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
    
    # Device configuration
    USE_CUDA = True
    NUM_WORKERS = 2
    PIN_MEMORY = True