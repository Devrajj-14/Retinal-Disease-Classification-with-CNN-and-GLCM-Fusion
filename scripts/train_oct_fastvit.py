#!/usr/bin/env python3
"""
Training script for OCT 8-class classification using FastViT
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import warnings

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.oct_config import OCTConfig
from src.data.oct_dataset import OCTDataProcessor, OCTDataset
from src.models.fastvit_model import FastViTModel
from src.utils.visualization import create_output_directories, save_training_plots
from src.training.oct_trainer import ModelTrainer

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='Train FastViT for OCT Classification')
    parser.add_argument('--data_dir', type=str, default=OCTConfig.BASE_DATA_DIR,
                       help='Path to OCT dataset directory')
    parser.add_argument('--model_name', type=str, default=OCTConfig.MODEL_NAME,
                       choices=FastViTModel.get_available_models(),
                       help='FastViT model variant')
    parser.add_argument('--epochs', type=int, default=OCTConfig.EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=OCTConfig.BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=OCTConfig.LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str, default=OCTConfig.OUTPUT_DIR,
                       help='Output directory for models and plots')
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_args()
    
    print("="*60)
    print("8-CLASS OCT CLASSIFICATION USING FASTVIT")
    print("="*60)
    
    # Create output directories
    create_output_directories()
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    try:
        # Initialize data processor
        print("\n=== INITIALIZING DATA PROCESSOR ===")
        data_processor = OCTDataProcessor(args.data_dir, OCTConfig.TARGET_CLASSES)
        
        # Scan dataset structure
        datasets = data_processor.scan_dataset_structure()
        
        # Create dataframes
        dataframes = data_processor.create_dataframe_from_directories()
        
        # Analyze distribution
        combined_df = data_processor.analyze_dataset_distribution(dataframes)
        
        # Initialize model builder
        model_builder = FastViTModel(
            model_name=args.model_name, 
            num_classes=len(OCTConfig.TARGET_CLASSES)
        )
        
        # Initialize trainer
        trainer = ModelTrainer(data_processor, model_builder)
        
        # Train model
        model, history = trainer.train_model(
            dataframes, 
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            learning_rate=args.learning_rate
        )
        
        # Plot training history
        trainer.plot_training_history()
        
        # Evaluate on test set if available
        test_results = trainer.evaluate_test_set(dataframes)
        
        # Save training plots
        save_training_plots(
            history, 
            trainer.results, 
            OCTConfig.TARGET_CLASSES, 
            args.model_name,
            os.path.join(args.output_dir, "plots")
        )
        
        # Save model
        model_save_path = os.path.join(args.output_dir, "models", f"fastvit_{args.model_name}_best.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': args.model_name,
            'num_classes': len(OCTConfig.TARGET_CLASSES),
            'class_names': OCTConfig.TARGET_CLASSES,
            'config': vars(args)
        }, model_save_path)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Model: {args.model_name}")
        print(f"Model saved: {model_save_path}")
        print(f"Training plots saved in: {os.path.join(args.output_dir, 'plots')}")
        
        return trainer
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease ensure:")
        print("1. Dataset directory exists and contains train/test/val folders")
        print("2. FastViT model files are available (install timm)")
        print(f"\nCurrent configuration:")
        print(f"- Base directory: {args.data_dir}")
        print(f"- Target classes: {OCTConfig.TARGET_CLASSES}")
        print(f"- Model: {args.model_name}")
        return None
        
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    trainer = main()
    
    if trainer is not None:
        print("\n=== TRAINING SUMMARY ===")
        if hasattr(trainer, 'results') and trainer.results:
            print(f"Final validation accuracy: {trainer.results.get('accuracy', 'N/A'):.4f}")
        print("Training completed successfully!")