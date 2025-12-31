"""Diabetic Retinopathy Dataset with GLCM features"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split

from src.features.glcm_extractor import GLCMFeatureExtractor

class FolderDRDataset(Dataset):
    """Dataset for folder-based DR classification with GLCM features"""
    
    def __init__(self, root_dir, transform=None, extract_glcm=True):
        self.root_dir = root_dir
        self.transform = transform
        self.extract_glcm = extract_glcm
        
        if self.extract_glcm:
            self.glcm_extractor = GLCMFeatureExtractor()
        
        # Load samples from folders
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        print(f"📁 Loading dataset from: {root_dir}")
        
        # Get class folders
        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                self.classes.append(class_name)
                class_idx = len(self.classes) - 1
                self.class_to_idx[class_name] = class_idx
                
                # Load images from this class folder
                image_count = 0
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_name)
                        self.samples.append((img_path, class_idx))
                        image_count += 1
                
                print(f"   📊 Class {class_name} (ID: {class_idx}): {image_count} images")
        
        print(f"✅ Loaded {len(self.samples)} total samples from {len(self.classes)} classes")
        
        # Calculate class distribution
        labels = [label for _, label in self.samples]
        self.class_counts = Counter(labels)
        
        # Calculate sample weights for balanced sampling
        total_samples = len(self.samples)
        self.class_weights = {cls_idx: total_samples / count 
                             for cls_idx, count in self.class_counts.items()}
        self.sample_weights = [self.class_weights[label] for _, label in self.samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"❌ Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Extract GLCM features before transform
        if self.extract_glcm:
            glcm_features = self.glcm_extractor.extract_glcm_features(image)
        else:
            glcm_features = torch.zeros(12, dtype=torch.float32)  # Default size
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(glcm_features, dtype=torch.float32), label


def create_data_loaders(data_dir, batch_size=16, val_split=0.2, test_split=0.1, 
                       image_size=224, model_type='densenet121'):
    """Create train/val/test data loaders from folder structure"""
    
    print("🎨 Setting up data transformations...")
    
    # Adjust image size based on model type
    if model_type == 'efficientnet_b3':
        image_size = 300
    
    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create full dataset
    print("📦 Creating full dataset...")
    full_dataset = FolderDRDataset(data_dir, transform=None, extract_glcm=True)
    
    # Create stratified splits
    print("✂️  Creating stratified train/val/test splits...")
    
    # Group samples by class
    class_samples = {i: [] for i in range(len(full_dataset.classes))}
    for idx, (_, label) in enumerate(full_dataset.samples):
        class_samples[label].append(idx)
    
    train_indices, val_indices, test_indices = [], [], []
    
    for class_id, samples in class_samples.items():
        np.random.shuffle(samples)
        n_samples = len(samples)
        
        # Calculate split sizes
        test_size = max(1, int(test_split * n_samples))
        val_size = max(1, int(val_split * n_samples))
        train_size = n_samples - test_size - val_size
        
        # Split indices
        test_indices.extend(samples[:test_size])
        val_indices.extend(samples[test_size:test_size + val_size])
        train_indices.extend(samples[test_size + val_size:])
    
    # Shuffle indices
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    print(f"📊 Data split:")
    print(f"   Training:   {len(train_indices)} samples")
    print(f"   Validation: {len(val_indices)} samples")
    print(f"   Test:       {len(test_indices)} samples")
    
    # Create subset datasets with transforms
    class TransformSubset(Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            original_idx = self.indices[idx]
            img_path, label = self.dataset.samples[original_idx]
            
            # Load image
            try:
                image = Image.open(img_path).convert('RGB')
            except:
                image = Image.new('RGB', (image_size, image_size), (0, 0, 0))
            
            # Extract GLCM features
            glcm_features = self.dataset.glcm_extractor.extract_glcm_features(image)
            
            # Apply transform
            if self.transform:
                image = self.transform(image)
            
            return image, torch.tensor(glcm_features, dtype=torch.float32), label
    
    # Create transformed datasets
    train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
    val_dataset = TransformSubset(full_dataset, val_indices, val_test_transform)
    test_dataset = TransformSubset(full_dataset, test_indices, val_test_transform)
    
    # Create weighted sampler for training
    train_weights = [full_dataset.sample_weights[i] for i in train_indices]
    train_sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
    
    # Create data loaders
    print("🚚 Creating data loaders...")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=2, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, full_dataset.classes