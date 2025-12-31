"""OCT Dataset processing for 8-class classification"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

class OCTDataProcessor:
    """Data processor for OCT images with directory-based class structure"""
    
    def __init__(self, base_data_dir, target_classes):
        self.base_data_dir = base_data_dir
        self.target_classes = target_classes
        self.train_dir = os.path.join(base_data_dir, 'train')
        self.test_dir = os.path.join(base_data_dir, 'test')
        self.val_dir = os.path.join(base_data_dir, 'val')
        
        # Class mapping
        self.class_mapping = {cls: idx for idx, cls in enumerate(target_classes)}
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(target_classes)
        
    def scan_dataset_structure(self):
        """Scan the dataset structure and create file mappings"""
        print("=== SCANNING DATASET STRUCTURE ===")
        datasets = {}
        
        for split in ['train', 'test', 'val']:
            split_dir = os.path.join(self.base_data_dir, split)
            if not os.path.exists(split_dir):
                print(f"Warning: {split} directory not found: {split_dir}")
                continue
                
            datasets[split] = {}
            split_total = 0
            print(f"\n{split.upper()} SET:")
            
            # Check available classes
            available_classes = [d for d in os.listdir(split_dir) 
                               if os.path.isdir(os.path.join(split_dir, d))]
            print(f"Available classes: {available_classes}")
            
            for class_name in self.target_classes:
                class_dir = os.path.join(split_dir, class_name)
                if os.path.exists(class_dir):
                    images = [f for f in os.listdir(class_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    datasets[split][class_name] = {
                        'path': class_dir,
                        'images': images,
                        'count': len(images)
                    }
                    split_total += len(images)
                    print(f"  {class_name}: {len(images)} images")
                else:
                    print(f"  {class_name}: NOT FOUND")
                    datasets[split][class_name] = {
                        'path': class_dir,
                        'images': [],
                        'count': 0
                    }
            
            print(f"  Total {split} images: {split_total}")
        
        self.datasets = datasets
        return datasets
    
    def create_dataframe_from_directories(self):
        """Create pandas DataFrames from directory structure"""
        print("\n=== CREATING DATAFRAMES FROM DIRECTORIES ===")
        dataframes = {}
        
        for split in ['train', 'test', 'val']:
            if split not in self.datasets:
                continue
                
            data_list = []
            for class_name in self.target_classes:
                if class_name in self.datasets[split]:
                    class_data = self.datasets[split][class_name]
                    class_path = class_data['path']
                    
                    for img_name in class_data['images']:
                        img_path = os.path.join(class_path, img_name)
                        class_label = self.class_mapping[class_name]
                        
                        data_list.append({
                            'image_path': img_path,
                            'image_name': img_name,
                            'class': class_name,
                            'class_label': class_label,
                            'split': split
                        })
            
            df = pd.DataFrame(data_list)
            dataframes[split] = df
            
            print(f"{split.upper()} DataFrame shape: {df.shape}")
            if len(df) > 0:
                print(f"  Class distribution: {dict(df['class'].value_counts())}")
                print(f"  Numeric label distribution: {dict(df['class_label'].value_counts())}")
        
        return dataframes
    
    def analyze_dataset_distribution(self, dataframes):
        """Analyze and visualize dataset distribution"""
        print("\n=== DATASET DISTRIBUTION ANALYSIS ===")
        
        # Combine all splits for overall analysis
        all_data = []
        for split, df in dataframes.items():
            if len(df) > 0:
                all_data.append(df)
        
        if not all_data:
            print("No data found!")
            return
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print("Overall Distribution:")
        print(f"Total images: {len(combined_df)}")
        print(f"Class distribution: {dict(combined_df['class'].value_counts())}")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Overall class distribution
        class_dist = combined_df['class'].value_counts()
        colors = ['orange', 'green', 'purple', 'pink', 'blue', 'red', 'grey', 'brown']
        bars = axes[0,0].bar(class_dist.index, class_dist.values, color=colors)
        axes[0,0].set_title('Overall Class Distribution')
        axes[0,0].set_ylabel('Count')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        for i, v in enumerate(class_dist.values):
            axes[0,0].text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Class distribution pie chart
        axes[0,1].pie(class_dist.values, labels=class_dist.index, autopct='%1.1f%%', colors=colors)
        axes[0,1].set_title('Class Distribution (Percentage)')
        
        # Split distribution
        split_dist = combined_df['split'].value_counts()
        axes[1,0].bar(split_dist.index, split_dist.values, color=['skyblue', 'lightgreen', 'salmon'])
        axes[1,0].set_title('Split Distribution')
        axes[1,0].set_ylabel('Count')
        
        for i, v in enumerate(split_dist.values):
            axes[1,0].text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Class distribution by split
        split_class = combined_df.groupby(['split', 'class']).size().unstack(fill_value=0)
        split_class.plot(kind='bar', ax=axes[1,1], color=colors)
        axes[1,1].set_title('Class Distribution by Split')
        axes[1,1].set_ylabel('Count')
        axes[1,1].legend(title='Class')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('outputs/plots/dataset_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return combined_df


class OCTDataset(Dataset):
    """PyTorch Dataset for OCT images"""
    
    def __init__(self, df, transform=None, target_size=(224, 224)):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.target_size = target_size
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            # Get image path and label
            img_path = self.df.iloc[idx]['image_path']
            label = self.df.iloc[idx]['class_label']
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            image = image.resize(self.target_size)
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return image, label
        
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            blank_image = Image.new('RGB', self.target_size, (0, 0, 0))
            if self.transform:
                blank_image = self.transform(blank_image)
            return blank_image, 0