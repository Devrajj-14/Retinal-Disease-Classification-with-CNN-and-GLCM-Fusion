"""Training module for OCT classification using FastViT"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.data.oct_dataset import OCTDataset

class ModelTrainer:
    """Train and evaluate FastViT model"""
    
    def __init__(self, data_processor, model_builder):
        self.data_processor = data_processor
        self.model_builder = model_builder
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.history = None
    
    def get_transforms(self):
        """Get data transforms for training and validation"""
        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Validation/test transforms without augmentation
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def train_model(self, dataframes, epochs=30, batch_size=32, learning_rate=0.001):
        """Train the FastViT model"""
        print("\n=== TRAINING FASTVIT MODEL ===")
        
        # Get training and validation data
        train_df = dataframes.get('train', pd.DataFrame())
        val_df = dataframes.get('val', pd.DataFrame())
        
        # If no validation set, create one from training data
        if len(val_df) == 0 and len(train_df) > 0:
            print("No validation set found. Creating validation split from training data...")
            train_df, val_df = train_test_split(
                train_df, test_size=0.2, stratify=train_df['class_label'], random_state=42
            )
        
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        
        # Get transforms
        train_transform, val_transform = self.get_transforms()
        
        # Create datasets
        train_dataset = OCTDataset(train_df, transform=train_transform)
        val_dataset = OCTDataset(val_df, transform=val_transform)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=4, pin_memory=True if torch.cuda.is_available() else False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Build model
        self.model = self.model_builder.build_model()
        
        # Calculate class weights for handling imbalance
        if len(train_df) > 0:
            classes = np.unique(train_df['class_label'])
            class_weights = compute_class_weight(
                'balanced', classes=classes, y=train_df['class_label']
            )
            class_weight_tensor = torch.FloatTensor(class_weights).to(self.device)
            print(f"Class weights: {dict(zip(classes, class_weights))}")
        else:
            class_weight_tensor = None
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training loop
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        best_val_acc = 0.0
        patience_counter = 0
        patience = 10
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, (images, labels) in enumerate(train_pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{train_correct/train_total:.4f}'
                })
            
            train_acc = 100 * train_correct / train_total
            train_loss = train_loss / len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            val_loss = val_loss / len(val_loader)
            
            # Store metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            # Print epoch results
            print(f'Epoch [{epoch+1}/{epochs}]: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping and model saving
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, 'outputs/models/fastvit_oct_best.pth')
                
                print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
        
        # Store training history
        self.history = {
            'train_loss': train_losses,
            'train_accuracy': train_accuracies,
            'val_loss': val_losses,
            'val_accuracy': val_accuracies
        }
        
        # Load best model
        checkpoint = torch.load('outputs/models/fastvit_oct_best.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate model
        self.evaluate_model(val_loader, val_df)
        
        return self.model, self.history
    
    def evaluate_model(self, val_loader, val_df):
        """Evaluate model performance"""
        print("\n=== EVALUATING MODEL PERFORMANCE ===")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate accuracy
        accuracy = np.mean(np.array(all_labels) == np.array(all_predictions))
        print(f"Validation Accuracy: {accuracy:.4f}")
        
        # Classification report
        class_names = self.data_processor.target_classes
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        self.results = {
            'accuracy': accuracy,
            'y_true': all_labels,
            'y_pred': all_predictions,
            'class_names': class_names
        }
        
        return self.results
    
    def evaluate_test_set(self, dataframes):
        """Evaluate on test set if available"""
        test_df = dataframes.get('test', pd.DataFrame())
        if len(test_df) == 0:
            print("No test set available for evaluation")
            return
        
        print("\n=== EVALUATING ON TEST SET ===")
        print(f"Test samples: {len(test_df)}")
        
        # Get transforms
        _, test_transform = self.get_transforms()
        
        # Create test dataset and loader
        test_dataset = OCTDataset(test_df, transform=test_transform)
        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False,
            num_workers=4, pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Get predictions
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_labels) == np.array(all_predictions))
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        class_names = self.data_processor.target_classes
        print("\nTest Set Classification Report:")
        print(classification_report(all_labels, all_predictions, target_names=class_names))
        
        return {
            'test_accuracy': accuracy,
            'test_y_true': all_labels,
            'test_y_pred': all_predictions
        }
    
    def plot_training_history(self):
        """Plot training history"""
        if not hasattr(self, 'history') or self.history is None:
            print("No training history available")
            return
        
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_accuracy'], label='Train Accuracy', linewidth=2)
        plt.plot(self.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        plt.title('Model Accuracy - FastViT')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        plt.title('Model Loss - FastViT')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/plots/training_history_fastvit.png', dpi=300, bbox_inches='tight')
        plt.show()