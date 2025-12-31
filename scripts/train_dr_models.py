#!/usr/bin/env python3
"""
Training script for Diabetic Retinopathy classification using CNN + GLCM
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import time
from tqdm import tqdm
import warnings

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.dr_config import DRConfig
from src.data.dr_dataset import create_data_loaders
from src.models.cnn_glcm_models import get_model
from src.utils.visualization import create_output_directories, save_training_plots
from src.training.dr_trainer import DRTrainer

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='Train CNN+GLCM for DR Classification')
    parser.add_argument('--data_dir', type=str, default=DRConfig.DATA_DIR,
                       help='Path to DR dataset directory')
    parser.add_argument('--model', type=str, default='densenet121',
                       choices=['densenet121', 'resnet50', 'efficientnet_b3'],
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=DRConfig.EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (auto-selected based on model if not specified)')
    parser.add_argument('--n_trials', type=int, default=DRConfig.N_TRIALS,
                       help='Number of Optuna optimization trials')
    parser.add_argument('--output_dir', type=str, default=DRConfig.OUTPUT_DIR,
                       help='Output directory for models and plots')
    parser.add_argument('--skip_optimization', action='store_true',
                       help='Skip hyperparameter optimization and use default parameters')
    
    return parser.parse_args()

def objective_function(trial, train_loader, val_loader, n_classes, model_name):
    """Optuna objective function for hyperparameter optimization"""
    
    # Suggest hyperparameters
    lr = trial.suggest_loguniform("lr", *DRConfig.LEARNING_RATE_RANGE)
    momentum = trial.suggest_uniform("momentum", *DRConfig.MOMENTUM_RANGE)
    dropout_rates = [
        trial.suggest_uniform("dropout_1", *DRConfig.DROPOUT_RANGES['dropout_1']),
        trial.suggest_uniform("dropout_2", *DRConfig.DROPOUT_RANGES['dropout_2']),
        trial.suggest_uniform("dropout_3", *DRConfig.DROPOUT_RANGES['dropout_3'])
    ]
    
    if model_name == 'efficientnet_b3':
        weight_decay = trial.suggest_loguniform("weight_decay", *DRConfig.WEIGHT_DECAY_RANGE)
    else:
        weight_decay = 1e-4
    
    print(f"🔧 Trial {trial.number}: lr={lr:.6f}, momentum={momentum:.3f}")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_name, n_classes, glcm_size=12, dropout_rates=dropout_rates)
    model = model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    # Short training for optimization
    model.train()
    for epoch in range(3):  # Few epochs for quick evaluation
        for batch_idx, (inputs, glcm_feats, labels) in enumerate(train_loader):
            if batch_idx > 8:  # Limit batches for speed
                break
            
            inputs = inputs.to(device, non_blocking=True)
            glcm_feats = glcm_feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(inputs, glcm_feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Validation evaluation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, glcm_feats, labels) in enumerate(val_loader):
            if batch_idx > 12:  # Limit for speed
                break
            
            inputs = inputs.to(device, non_blocking=True)
            glcm_feats = glcm_feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(inputs, glcm_feats)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total if total > 0 else 0
    print(f"   🎯 Trial {trial.number} accuracy: {accuracy:.4f}")
    
    # Clean up memory
    del model
    torch.cuda.empty_cache()
    
    return 1 - accuracy  # Optuna minimizes

def run_optimization(train_loader, val_loader, n_classes, model_name, n_trials):
    """Run Optuna hyperparameter optimization"""
    
    print(f"🔧 Starting Optuna optimization for {model_name}...")
    print(f"   🎯 Target: {n_trials} trials")
    
    # Create study
    study = optuna.create_study(
        direction='minimize',
        study_name=f'{model_name}_glcm_optimization',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)
    )
    
    # Run optimization
    start_time = time.time()
    study.optimize(
        lambda trial: objective_function(trial, train_loader, val_loader, n_classes, model_name),
        n_trials=n_trials,
        timeout=DRConfig.OPTIMIZATION_TIMEOUT
    )
    
    optimization_time = time.time() - start_time
    print(f"✅ Optimization completed in {optimization_time:.2f}s ({optimization_time/60:.1f} minutes)")
    
    # Best parameters
    best_params = study.best_params
    best_score = 1 - study.best_value  # Convert back to accuracy
    
    print(f"\n🏆 BEST HYPERPARAMETERS FOR {model_name.upper()}:")
    print(f"   🎯 Best Validation Accuracy: {best_score:.4f}")
    for param, value in best_params.items():
        print(f"   🔧 {param}: {value}")
    
    return best_params, best_score

def main():
    """Main training function"""
    args = parse_args()
    
    print(f"🚀 {args.model.upper()} + GLCM for Diabetic Retinopathy Classification")
    print("=" * 80)
    
    # Create output directories
    create_output_directories()
    
    # Get model configuration
    model_config = DRConfig.AVAILABLE_MODELS[args.model]
    batch_size = args.batch_size or model_config['batch_size']
    image_size = model_config['image_size']
    
    print(f"📁 Dataset directory: {args.data_dir}")
    print(f"🏗️  Model: {args.model}")
    print(f"📐 Image size: {image_size}x{image_size}")
    print(f"📦 Batch size: {batch_size}")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    try:
        # Create data loaders
        print("\n🏗️  Creating data loaders...")
        start_time = time.time()
        
        train_loader, val_loader, test_loader, class_names = create_data_loaders(
            args.data_dir, 
            batch_size=batch_size,
            val_split=DRConfig.VAL_SPLIT,
            test_split=DRConfig.TEST_SPLIT,
            image_size=image_size,
            model_type=args.model
        )
        
        loader_time = time.time() - start_time
        print(f"⏱️  Data loaders created in {loader_time:.2f}s")
        
        n_classes = len(class_names)
        print(f"📊 Dataset summary:")
        print(f"   Classes: {class_names}")
        print(f"   Number of classes: {n_classes}")
        
        # Hyperparameter optimization
        if not args.skip_optimization:
            print(f"\n🔧 Running Optuna optimization...")
            best_params, best_score = run_optimization(
                train_loader, val_loader, n_classes, args.model, args.n_trials
            )
        else:
            print("\n⏭️  Skipping optimization, using default parameters...")
            best_params = {
                'lr': 0.001,
                'momentum': 0.9,
                'dropout_1': 0.2,
                'dropout_2': 0.3,
                'dropout_3': 0.4,
                'weight_decay': 1e-4
            }
            best_score = 0.0
        
        # Train final model
        print(f"\n🏋️  Training final {args.model} model...")
        
        trainer = DRTrainer(args.model, n_classes, best_params)
        final_model, test_accuracy, best_val_acc, history = trainer.train_model(
            train_loader, val_loader, test_loader, class_names, args.epochs
        )
        
        # Save results
        model_save_path = DRConfig.MODEL_SAVE_PATHS[args.model]
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'model_name': args.model,
            'num_classes': n_classes,
            'class_names': class_names,
            'best_params': best_params,
            'test_accuracy': test_accuracy,
            'config': vars(args)
        }, model_save_path)
        
        # Final summary
        print(f"\n🎉 {args.model.upper()} + GLCM FINAL RESULTS:")
        print("=" * 60)
        print(f"🔧 Optuna Trials: {args.n_trials}")
        print(f"🎯 Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        print(f"🧪 Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"📊 Classes: {class_names}")
        print(f"🏗️  Architecture: {args.model} + GLCM features")
        print("=" * 60)
        
        print(f"\n🔧 Optimized Hyperparameters:")
        for param, value in best_params.items():
            print(f"   {param}: {value}")
        
        print(f"\n✅ Training completed! Model saved as '{model_save_path}'")
        
        return trainer
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    trainer = main()
    
    if trainer is not None:
        print("\n=== TRAINING SUMMARY ===")
        print("Training completed successfully!")