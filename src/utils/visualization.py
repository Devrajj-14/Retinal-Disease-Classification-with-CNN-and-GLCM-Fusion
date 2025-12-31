"""Visualization utilities for training results and analysis"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os

def plot_training_history(train_losses, val_losses, train_accs, val_accs, 
                         test_accuracy=None, model_name="Model", save_path=None):
    """Plot training history curves"""
    
    plt.figure(figsize=(15, 5))
    
    # Loss curves
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Val Loss', color='red', linewidth=2)
    plt.title(f'{model_name} Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy curves
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Acc', color='blue', linewidth=2)
    plt.plot(val_accs, label='Val Acc', color='red', linewidth=2)
    plt.title(f'{model_name} Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Validation accuracy progress
    plt.subplot(1, 3, 3)
    plt.plot([acc * 100 for acc in val_accs], 'g-', linewidth=2)
    if test_accuracy is not None:
        plt.axhline(y=test_accuracy*100, color='r', linestyle='--', 
                   label=f'Test Acc: {test_accuracy:.3f}')
    plt.title(f'{model_name} Validation Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, model_name="Model", 
                         test_accuracy=None, save_path=None):
    """Plot confusion matrix"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    title = f'{model_name} Confusion Matrix'
    if test_accuracy is not None:
        title += f' - Test Accuracy: {test_accuracy:.4f}'
    plt.title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_per_class_accuracy(y_true, y_pred, class_names, model_name="Model", save_path=None):
    """Plot per-class accuracy"""
    
    cm = confusion_matrix(y_true, y_pred)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    bars = plt.bar(class_names, class_accuracy, color=colors)
    
    plt.title(f'{model_name} Per-Class Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars, class_accuracy):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_dataset_distribution(class_counts, title="Dataset Distribution", save_path=None):
    """Plot dataset class distribution"""
    
    plt.figure(figsize=(12, 8))
    
    # Bar plot
    plt.subplot(2, 2, 1)
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
    bars = plt.bar(class_counts.keys(), class_counts.values(), color=colors)
    plt.title('Class Distribution')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    for bar, count in zip(bars, class_counts.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    plt.subplot(2, 2, 2)
    plt.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%', colors=colors)
    plt.title('Class Distribution (Percentage)')
    
    # Statistics
    plt.subplot(2, 2, 3)
    plt.axis('off')
    total = sum(class_counts.values())
    stats_text = f"Total Images: {total}\n"
    stats_text += f"Number of Classes: {len(class_counts)}\n"
    stats_text += f"Average per Class: {total/len(class_counts):.1f}\n"
    stats_text += f"Min Class Size: {min(class_counts.values())}\n"
    stats_text += f"Max Class Size: {max(class_counts.values())}"
    plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_output_directories():
    """Create necessary output directories"""
    directories = [
        'outputs',
        'outputs/models',
        'outputs/plots',
        'outputs/logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Output directories created")


def save_training_plots(history, test_results, class_names, model_name, output_dir="outputs/plots"):
    """Save all training plots"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Training history
    plot_training_history(
        history['train_loss'], history['val_loss'],
        history['train_accuracy'], history['val_accuracy'],
        test_results.get('test_accuracy'),
        model_name,
        save_path=os.path.join(output_dir, f'{model_name}_training_history.png')
    )
    
    # Confusion matrix
    if 'y_true' in test_results and 'y_pred' in test_results:
        plot_confusion_matrix(
            test_results['y_true'], test_results['y_pred'],
            class_names, model_name,
            test_results.get('test_accuracy'),
            save_path=os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
        )
        
        # Per-class accuracy
        plot_per_class_accuracy(
            test_results['y_true'], test_results['y_pred'],
            class_names, model_name,
            save_path=os.path.join(output_dir, f'{model_name}_per_class_accuracy.png')
        )