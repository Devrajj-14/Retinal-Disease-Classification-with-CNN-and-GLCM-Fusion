"""Training module for Diabetic Retinopathy classification using CNN + GLCM"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import time
from tqdm import tqdm

from src.models.cnn_glcm_models import get_model
from src.utils.visualization import plot_training_history, plot_confusion_matrix

class DRTrainer:
    """Trainer for DR classification models"""
    
    def __init__(self, model_name, n_classes, best_params):
        self.model_name = model_name
        self.n_classes = n_classes
        self.best_params = best_params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.history = None
    
    def create_model(self):
        """Create model with optimized parameters"""
        dropout_rates = [
            self.best_params.get('dropout_1', 0.2),
            self.best_params.get('dropout_2', 0.3),
            self.best_params.get('dropout_3', 0.4)
        ]
        
        model = get_model(self.model_name, self.n_classes, glcm_size=12, dropout_rates=dropout_rates)
        return model.to(self.device)
    
    def train_model(self, train_loader, val_loader, test_loader, class_names, num_epochs=50):
        """Train the model with optimized hyperparameters"""
        
        print(f"🏋️  Training {self.model_name} model...")
        print(f"📊 Using parameters: {self.best_params}")
        
        # Create model
        self.model = self.create_model()
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.best_params.get('lr', 0.001),
            momentum=self.best_params.get('momentum', 0.9),
            weight_decay=self.best_params.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5
        )
        
        # Training tracking
        train_accs, val_accs, train_losses, val_losses = [], [], [], []
        best_val_acc = 0.0
        patience_counter = 0
        patience = 10
        
        print(f"🚀 Starting {self.model_name} training...")
        print("=" * 80)
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct, total = 0, 0
            
            train_pbar = tqdm(train_loader, desc=f"🏃 Epoch {epoch+1}/{num_epochs}", leave=False)
            for inputs, glcm_feats, labels in train_pbar:
                inputs = inputs.to(self.device, non_blocking=True)
                glcm_feats = glcm_feats.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = self.model(inputs, glcm_feats)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{correct/total:.4f}'
                })
            
            train_acc = correct / total
            train_loss = running_loss / len(train_loader)
            train_accs.append(train_acc)
            train_losses.append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_correct, val_total = 0, 0
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, glcm_feats, labels in val_loader:
                    inputs = inputs.to(self.device, non_blocking=True)
                    glcm_feats = glcm_feats.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    
                    outputs = self.model(inputs, glcm_feats)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = val_correct / val_total
            val_loss = val_loss / len(val_loader)
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            epoch_time = time.time() - epoch_start
            print(f"⏱️  Epoch {epoch+1:2d} ({epoch_time:5.1f}s) | "
                  f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save model state
                self.best_model_state = self.model.state_dict().copy()
                print(f"🎯 New best validation accuracy: {val_acc:.4f} - Model saved!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"🛑 Early stopping after {epoch+1} epochs")
                    break
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        
        # Store training history
        self.history = {
            'train_loss': train_losses,
            'train_accuracy': train_accs,
            'val_loss': val_losses,
            'val_accuracy': val_accs
        }
        
        # Final evaluation on test set
        test_accuracy = self.evaluate_test_set(test_loader, class_names)
        
        # Plot results
        self.plot_results(test_accuracy, class_names)
        
        return self.model, test_accuracy, best_val_acc, self.history
    
    def evaluate_test_set(self, test_loader, class_names):
        """Evaluate model on test set"""
        
        print("\n🧪 Final evaluation on test set...")
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, glcm_feats, labels in tqdm(test_loader, desc="Testing"):
                inputs = inputs.to(self.device, non_blocking=True)
                glcm_feats = glcm_feats.to(self.device, non_blocking=True)
                
                outputs = self.model(inputs, glcm_feats)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        test_accuracy = accuracy_score(all_labels, all_preds)
        print(f"🎯 Final Test Accuracy: {test_accuracy:.4f}")
        
        print("\n📊 Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        
        # Store results for plotting
        self.test_results = {
            'y_true': all_labels,
            'y_pred': all_preds,
            'test_accuracy': test_accuracy
        }
        
        return test_accuracy
    
    def plot_results(self, test_accuracy, class_names):
        """Plot training results"""
        
        if self.history is None:
            return
        
        # Plot training history
        plot_training_history(
            self.history['train_loss'],
            self.history['val_loss'],
            self.history['train_accuracy'],
            self.history['val_accuracy'],
            test_accuracy,
            f"{self.model_name.upper()}",
            f"outputs/plots/{self.model_name}_training_history.png"
        )
        
        # Plot confusion matrix
        if hasattr(self, 'test_results'):
            plot_confusion_matrix(
                self.test_results['y_true'],
                self.test_results['y_pred'],
                class_names,
                f"{self.model_name.upper()} + GLCM",
                test_accuracy,
                f"outputs/plots/{self.model_name}_confusion_matrix.png"
            )