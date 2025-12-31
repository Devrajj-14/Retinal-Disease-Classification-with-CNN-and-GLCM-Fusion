"""Configuration for OCT 8-class classification using FastViT"""

import os

class OCTConfig:
    # Dataset configuration
    BASE_DATA_DIR = "/kaggle/input/retinal-oct-c8/RetinalOCT_Dataset/RetinalOCT_Dataset"
    TARGET_CLASSES = ['DR', 'NORMAL', 'DME', 'AMD', 'CNV', 'DRUSEN', 'MH', 'CSR']
    
    # Model configuration
    MODEL_NAME = "fastvit_t8"  # Options: fastvit_t8, fastvit_t12, fastvit_s12, etc.
    NUM_CLASSES = 8
    PRETRAINED = True
    
    # Training configuration
    EPOCHS = 30
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.01
    
    # Data augmentation
    IMAGE_SIZE = (224, 224)
    ROTATION_DEGREES = 15
    HORIZONTAL_FLIP_PROB = 0.5
    VERTICAL_FLIP_PROB = 0.1
    COLOR_JITTER = {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    }
    
    # Training parameters
    PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5
    
    # Paths
    OUTPUT_DIR = "outputs"
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "models", "fastvit_oct_best.pth")
    INFERENCE_MODEL_PATH = os.path.join(OUTPUT_DIR, "models", "fastvit_oct_inference.pth")
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
    
    # Device configuration
    USE_CUDA = True
    NUM_WORKERS = 4
    PIN_MEMORY = True