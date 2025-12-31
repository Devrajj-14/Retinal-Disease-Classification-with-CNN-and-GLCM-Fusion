#!/usr/bin/env python3
"""
Example usage of the OCT Classification project
"""

import os
import sys

# Add src to path
sys.path.append('src')

def example_oct_classification():
    """Example of OCT 8-class classification"""
    
    print("=== OCT 8-Class Classification Example ===")
    
    # Update these paths to match your dataset
    data_dir = "/path/to/your/oct/dataset"  # Update this path
    
    # Check if dataset exists
    if not os.path.exists(data_dir):
        print(f"❌ Dataset not found at: {data_dir}")
        print("Please update the data_dir path in this script")
        return
    
    # Import required modules
    from config.oct_config import OCTConfig
    from src.data.oct_dataset import OCTDataProcessor
    from src.models.fastvit_model import FastViTModel
    from src.training.oct_trainer import ModelTrainer
    
    # Update config
    OCTConfig.BASE_DATA_DIR = data_dir
    
    # Initialize components
    data_processor = OCTDataProcessor(data_dir, OCTConfig.TARGET_CLASSES)
    model_builder = FastViTModel(model_name="fastvit_t8", num_classes=8)
    trainer = ModelTrainer(data_processor, model_builder)
    
    # Process data
    datasets = data_processor.scan_dataset_structure()
    dataframes = data_processor.create_dataframe_from_directories()
    
    # Train model (use small epochs for demo)
    model, history = trainer.train_model(dataframes, epochs=5, batch_size=16)
    
    print("✅ OCT classification example completed!")

def example_dr_classification():
    """Example of DR classification with CNN + GLCM"""
    
    print("\n=== Diabetic Retinopathy Classification Example ===")
    
    # Update this path to match your dataset
    data_dir = "/path/to/your/dr/dataset"  # Update this path
    
    # Check if dataset exists
    if not os.path.exists(data_dir):
        print(f"❌ Dataset not found at: {data_dir}")
        print("Please update the data_dir path in this script")
        return
    
    # Import required modules
    from src.data.dr_dataset import create_data_loaders
    from src.models.cnn_glcm_models import get_model
    from src.training.dr_trainer import DRTrainer
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        data_dir, batch_size=8, model_type='densenet121'
    )
    
    # Example hyperparameters (normally from Optuna optimization)
    best_params = {
        'lr': 0.001,
        'momentum': 0.9,
        'dropout_1': 0.2,
        'dropout_2': 0.3,
        'dropout_3': 0.4,
        'weight_decay': 1e-4
    }
    
    # Initialize trainer
    trainer = DRTrainer('densenet121', len(class_names), best_params)
    
    # Train model (use small epochs for demo)
    model, test_accuracy, best_val_acc, history = trainer.train_model(
        train_loader, val_loader, test_loader, class_names, num_epochs=3
    )
    
    print("✅ DR classification example completed!")

def run_command_line_examples():
    """Show command line usage examples"""
    
    print("\n=== Command Line Usage Examples ===")
    
    print("\n1. OCT 8-Class Classification:")
    print("python scripts/train_oct_fastvit.py \\")
    print("    --data_dir /path/to/oct/dataset \\")
    print("    --model_name fastvit_t8 \\")
    print("    --epochs 30 \\")
    print("    --batch_size 32")
    
    print("\n2. Diabetic Retinopathy Classification:")
    print("python scripts/train_dr_models.py \\")
    print("    --data_dir /path/to/dr/dataset \\")
    print("    --model densenet121 \\")
    print("    --epochs 50 \\")
    print("    --n_trials 10")
    
    print("\n3. Skip hyperparameter optimization:")
    print("python scripts/train_dr_models.py \\")
    print("    --data_dir /path/to/dr/dataset \\")
    print("    --model resnet50 \\")
    print("    --skip_optimization")

def main():
    """Main example function"""
    
    print("🚀 OCT Classification Project - Usage Examples")
    print("=" * 60)
    
    # Show command line examples
    run_command_line_examples()
    
    print("\n" + "=" * 60)
    print("📝 Notes:")
    print("1. Update dataset paths in the configuration files")
    print("2. Install required dependencies: pip install -r requirements.txt")
    print("3. Ensure CUDA is available for GPU acceleration")
    print("4. Check the README.md for detailed instructions")
    
    # Uncomment these lines to run actual training examples
    # (make sure to update the dataset paths first)
    
    # example_oct_classification()
    # example_dr_classification()

if __name__ == "__main__":
    main()