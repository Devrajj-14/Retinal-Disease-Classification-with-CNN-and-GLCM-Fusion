"""Metrics and evaluation utilities"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(y_true, y_pred, class_names=None, average='weighted'):
    """Calculate comprehensive classification metrics"""
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    # Per-class metrics
    if class_names is not None:
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['per_class'] = {
            'precision': dict(zip(class_names, per_class_precision)),
            'recall': dict(zip(class_names, per_class_recall)),
            'f1_score': dict(zip(class_names, per_class_f1))
        }
    
    return metrics

def print_metrics_summary(metrics, title="Classification Metrics"):
    """Print formatted metrics summary"""
    
    print(f"\n=== {title} ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    if 'per_class' in metrics:
        print("\nPer-Class Metrics:")
        for class_name in metrics['per_class']['precision'].keys():
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['per_class']['precision'][class_name]:.4f}")
            print(f"    Recall:    {metrics['per_class']['recall'][class_name]:.4f}")
            print(f"    F1-Score:  {metrics['per_class']['f1_score'][class_name]:.4f}")

def plot_roc_curves(y_true, y_pred_proba, class_names, save_path=None):
    """Plot ROC curves for multi-class classification"""
    
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle
    
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    n_classes = len(class_names)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink'])
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return roc_auc

def calculate_class_balance_metrics(y_true, class_names=None):
    """Calculate class balance and distribution metrics"""
    
    unique, counts = np.unique(y_true, return_counts=True)
    total_samples = len(y_true)
    
    balance_metrics = {
        'total_samples': total_samples,
        'n_classes': len(unique),
        'class_counts': dict(zip(unique, counts)),
        'class_proportions': dict(zip(unique, counts / total_samples)),
        'imbalance_ratio': max(counts) / min(counts) if min(counts) > 0 else float('inf')
    }
    
    if class_names is not None:
        balance_metrics['class_names'] = class_names
        balance_metrics['named_counts'] = {class_names[i]: counts[i] for i in unique}
        balance_metrics['named_proportions'] = {class_names[i]: counts[i]/total_samples for i in unique}
    
    return balance_metrics

def print_class_balance_summary(balance_metrics):
    """Print class balance summary"""
    
    print("\n=== Class Balance Analysis ===")
    print(f"Total samples: {balance_metrics['total_samples']}")
    print(f"Number of classes: {balance_metrics['n_classes']}")
    print(f"Imbalance ratio: {balance_metrics['imbalance_ratio']:.2f}")
    
    print("\nClass Distribution:")
    if 'named_counts' in balance_metrics:
        for class_name, count in balance_metrics['named_counts'].items():
            proportion = balance_metrics['named_proportions'][class_name]
            print(f"  {class_name}: {count} samples ({proportion:.2%})")
    else:
        for class_id, count in balance_metrics['class_counts'].items():
            proportion = balance_metrics['class_proportions'][class_id]
            print(f"  Class {class_id}: {count} samples ({proportion:.2%})")

def evaluate_model_comprehensive(y_true, y_pred, y_pred_proba=None, class_names=None, 
                                model_name="Model", save_dir=None):
    """Comprehensive model evaluation with all metrics and plots"""
    
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE EVALUATION: {model_name}")
    print(f"{'='*60}")
    
    # Basic metrics
    metrics = calculate_metrics(y_true, y_pred, class_names)
    print_metrics_summary(metrics, f"{model_name} Performance")
    
    # Class balance analysis
    balance_metrics = calculate_class_balance_metrics(y_true, class_names)
    print_class_balance_summary(balance_metrics)
    
    # Classification report
    print(f"\n=== Detailed Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # ROC curves (if probabilities available)
    if y_pred_proba is not None and class_names is not None:
        roc_path = f"{save_dir}/{model_name}_roc_curves.png" if save_dir else None
        roc_auc = plot_roc_curves(y_true, y_pred_proba, class_names, roc_path)
        metrics['roc_auc'] = roc_auc
    
    return metrics, balance_metrics