#!/usr/bin/env python3
"""
Demo script with synthetic data to show the complete workflow
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add src to path
sys.path.append('src')

from src.features.glcm_extractor import GLCMFeatureExtractor
from src.models.fastvit_model import FastViTModel
from src.models.cnn_glcm_models import get_model
from src.utils.visualization import create_output_directories, plot_dataset_distribution
from src.utils.metrics import calculate_metrics, print_metrics_summary

def create_synthetic_dataset():
    """Create a small synthetic dataset for demonstration"""
    
    print("Creating synthetic dataset...")
    
    # Create synthetic images and labels
    n_samples = 100
    images = []
    labels = []
    
    # Create 5 classes with different synthetic patterns
    class_names = ['Normal', 'Mild_DR', 'Moderate_DR', 'Severe_DR', 'Proliferative_DR']
    
    for i in range(n_samples):
        # Create random RGB image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Add some class-specific patterns
        class_id = i % len(class_names)
        if class_id == 0:  # Normal - more uniform
            img_array = img_array * 0.8 + 50
        elif class_id == 1:  # Mild - slight variations
            img_array = img_array * 0.9 + np.random.randint(0, 30, img_array.shape)
        elif class_id == 2:  # Moderate - more contrast
            img_array = img_array * 1.2
        elif class_id == 3:  # Severe - high contrast
            img_array = img_array * 1.5
        else:  # Proliferative - very high contrast
            img_array = img_array * 2.0
        
        # Clip values
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        # Convert to PIL Image
        image = Image.fromarray(img_array)
        images.append(image)
        labels.append(class_id)
    
    print(f"Created {n_samples} synthetic images with {len(class_names)} classes")
    return images, labels, class_names

def demo_glcm_features():
    """Demonstrate GLCM feature extraction"""
    
    print("\nDemonstrating GLCM Feature Extraction...")
    
    # Create GLCM extractor
    extractor = GLCMFeatureExtractor()
    
    # Create synthetic images and extract features
    images, labels, class_names = create_synthetic_dataset()
    
    # Extract features from first few images
    features_list = []
    for i in range(min(10, len(images))):
        features = extractor.extract_glcm_features(images[i])
        features_list.append(features)
        if i < 3:
            print(f"Image {i} (Class: {class_names[labels[i]]}): {features[:3]}... (showing first 3 features)")
    
    features_array = np.array(features_list)
    print(f"Extracted GLCM features: {features_array.shape}")
    
    return features_array, labels[:len(features_list)]

def demo_fastvit_model():
    """Demonstrate FastViT model"""
    
    print("\n  Demonstrating FastViT Model...")
    
    # Create model
    model_builder = FastViTModel(model_name="fastvit_t8", num_classes=5, pretrained=False)
    model = model_builder.build_model()
    
    # Test with synthetic data
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f" FastViT inference successful! Input: {dummy_input.shape}, Output: {output.shape}")
    
    # Simulate predictions
    predictions = torch.argmax(output, dim=1).numpy()
    true_labels = np.random.randint(0, 5, batch_size)
    
    return predictions, true_labels

def demo_cnn_glcm_model():
    """Demonstrate CNN+GLCM model"""
    
    print("\n  Demonstrating CNN+GLCM Model...")
    
    # Create model
    model = get_model('densenet121', n_classes=5, glcm_size=12)
    
    # Test with synthetic data
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_glcm = torch.randn(batch_size, 12)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_images, dummy_glcm)
    
    print(f" CNN+GLCM inference successful! Output: {output.shape}")
    
    # Simulate predictions
    predictions = torch.argmax(output, dim=1).numpy()
    true_labels = np.random.randint(0, 5, batch_size)
    
    return predictions, true_labels

def demo_metrics_evaluation():
    """Demonstrate metrics calculation"""
    
    print("\n Demonstrating Metrics Evaluation...")
    
    # Generate synthetic predictions and labels
    n_samples = 50
    n_classes = 5
    class_names = ['Normal', 'Mild_DR', 'Moderate_DR', 'Severe_DR', 'Proliferative_DR']
    
    # Create realistic predictions (with some accuracy)
    true_labels = np.random.randint(0, n_classes, n_samples)
    predictions = true_labels.copy()
    
    # Add some errors (80% accuracy)
    error_indices = np.random.choice(n_samples, size=int(0.2 * n_samples), replace=False)
    predictions[error_indices] = np.random.randint(0, n_classes, len(error_indices))
    
    # Calculate metrics
    metrics = calculate_metrics(true_labels, predictions, class_names)
    print_metrics_summary(metrics, "Synthetic Dataset Evaluation")
    
    return metrics

def demo_visualization():
    """Demonstrate visualization utilities"""
    
    print("\n Demonstrating Visualization...")
    
    # Create sample class distribution
    class_counts = {
        'Normal': 1000,
        'Mild_DR': 800,
        'Moderate_DR': 600,
        'Severe_DR': 400,
        'Proliferative_DR': 200
    }
    
    # Create visualization
    plot_dataset_distribution(
        class_counts, 
        'Synthetic DR Dataset Distribution', 
        'outputs/plots/demo_distribution.png'
    )
    
    print(" Dataset distribution plot saved to outputs/plots/demo_distribution.png")

def main():
    """Main demo function"""
    
    print(" OCT Classification Project - Complete Demo")
    print("=" * 60)
    
    # Create output directories
    create_output_directories()
    
    try:
        # Demo 1: GLCM Feature Extraction
        features, labels = demo_glcm_features()
        
        # Demo 2: FastViT Model
        fastvit_preds, fastvit_labels = demo_fastvit_model()
        
        # Demo 3: CNN+GLCM Model
        cnn_preds, cnn_labels = demo_cnn_glcm_model()
        
        # Demo 4: Metrics Evaluation
        metrics = demo_metrics_evaluation()
        
        # Demo 5: Visualization
        demo_visualization()
        
        print("\n" + "=" * 60)
        print(" DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(" All components working correctly:")
        print("   • GLCM feature extraction")
        print("   • FastViT model inference")
        print("   • CNN+GLCM model inference")
        print("   • Metrics calculation")
        print("   • Visualization utilities")
        print("\n Check the 'outputs/' directory for generated files")
        
        return True
        
    except Exception as e:
        print(f"\n Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n Next Steps:")
        print("1. Update dataset paths in config files")
        print("2. Run training scripts with your actual data")
        print("3. Experiment with different model architectures")
        print("4. Use Optuna for hyperparameter optimization")
    else:
        print("\n Please check the error messages above and install missing dependencies")