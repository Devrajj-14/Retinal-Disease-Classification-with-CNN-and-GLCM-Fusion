"""CNN + GLCM combined models for diabetic retinopathy classification"""

import torch
import torch.nn as nn
from torchvision import models

try:
    from efficientnet_pytorch import EfficientNet
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    EFFICIENTNET_AVAILABLE = False

class CombinedDenseNetClassifier(nn.Module):
    """DenseNet121 + GLCM Combined Classifier"""
    
    def __init__(self, n_classes, glcm_size=12, dropout_rates=None):
        super(CombinedDenseNetClassifier, self).__init__()
        
        if dropout_rates is None:
            dropout_rates = [0.2, 0.3, 0.4]
        
        print(f"🏗️  Building DenseNet121 + GLCM model...")
        
        # DenseNet121 backbone
        self.base_model = models.densenet121(pretrained=True)
        self.base_model.classifier = nn.Identity()
        self.feature_size = 1024
        self.glcm_size = glcm_size
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.feature_size + self.glcm_size, momentum=0.999, eps=0.001),
            nn.Linear(self.feature_size + self.glcm_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rates[0]),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rates[2]),
            nn.Linear(256, n_classes)
        )
        
        print(f"✅ Model built: {self.feature_size} CNN + {self.glcm_size} GLCM → {n_classes} classes")
    
    def forward(self, x, glcm_feats):
        # DenseNet features
        x = self.base_model(x)
        # Combine CNN and GLCM features
        combined = torch.cat((x, glcm_feats), dim=1)
        # Final classification
        return self.classifier(combined)


class CombinedResNet50Classifier(nn.Module):
    """ResNet50 + GLCM Combined Classifier"""
    
    def __init__(self, n_classes, glcm_size=12, dropout_rates=None):
        super(CombinedResNet50Classifier, self).__init__()
        
        if dropout_rates is None:
            dropout_rates = [0.2, 0.3, 0.4]
        
        print(f"🏗️  Building ResNet50 + GLCM model...")
        
        # ResNet50 backbone
        self.base_model = models.resnet50(pretrained=True)
        self.base_model.fc = nn.Identity()  # Remove final FC layer
        self.feature_size = 2048  # ResNet50 feature size
        self.glcm_size = glcm_size
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.feature_size + self.glcm_size, momentum=0.999, eps=0.001),
            nn.Linear(self.feature_size + self.glcm_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rates[0]),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rates[2]),
            nn.Linear(256, n_classes)
        )
        
        print(f"✅ ResNet50 Model built: {self.feature_size} CNN + {self.glcm_size} GLCM → {n_classes} classes")
    
    def forward(self, x, glcm_feats):
        # ResNet50 features
        x = self.base_model(x)
        # Combine CNN and GLCM features
        combined = torch.cat((x, glcm_feats), dim=1)
        # Final classification
        return self.classifier(combined)


class CombinedEfficientNetB3Classifier(nn.Module):
    """EfficientNet-B3 + GLCM Combined Classifier"""
    
    def __init__(self, n_classes, glcm_size=12, dropout_rates=None):
        super(CombinedEfficientNetB3Classifier, self).__init__()
        
        if dropout_rates is None:
            dropout_rates = [0.2, 0.3, 0.4]
        
        print(f"🏗️  Building EfficientNet-B3 + GLCM model...")
        
        if EFFICIENTNET_AVAILABLE:
            # Use efficientnet_pytorch library
            self.base_model = EfficientNet.from_pretrained('efficientnet-b3')
            self.base_model._fc = nn.Identity()  # Remove final FC layer
            self.feature_size = 1536  # EfficientNet-B3 feature size
            self.use_pytorch_efficientnet = True
            print("✅ Using efficientnet_pytorch library")
        else:
            # Use torchvision version
            try:
                self.base_model = models.efficientnet_b3(pretrained=True)
                self.base_model.classifier = nn.Identity()
                self.feature_size = 1536
                self.use_pytorch_efficientnet = False
                print("✅ Using torchvision EfficientNet-B3")
            except:
                # Fallback to ResNet34 if EfficientNet not available
                print("⚠️  EfficientNet-B3 not available, using ResNet34 as fallback...")
                self.base_model = models.resnet34(pretrained=True)
                self.base_model.fc = nn.Identity()
                self.feature_size = 512
                self.use_pytorch_efficientnet = False
        
        self.glcm_size = glcm_size
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.feature_size + self.glcm_size, momentum=0.999, eps=0.001),
            nn.Linear(self.feature_size + self.glcm_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rates[0]),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rates[2]),
            nn.Linear(256, n_classes)
        )
        
        print(f"✅ EfficientNet-B3 Model built: {self.feature_size} CNN + {self.glcm_size} GLCM → {n_classes} classes")
    
    def forward(self, x, glcm_feats):
        # EfficientNet features
        if EFFICIENTNET_AVAILABLE and self.use_pytorch_efficientnet:
            # For efficientnet_pytorch library
            x = self.base_model.extract_features(x)
            x = self.base_model._avg_pooling(x)
            x = x.view(x.size(0), -1)
        else:
            # For torchvision version or fallback
            x = self.base_model(x)
        
        # Combine CNN and GLCM features
        combined = torch.cat((x, glcm_feats), dim=1)
        # Final classification
        return self.classifier(combined)


def get_model(model_name, n_classes, glcm_size=12, dropout_rates=None):
    """Factory function to get the specified model"""
    
    models_dict = {
        'densenet121': CombinedDenseNetClassifier,
        'resnet50': CombinedResNet50Classifier,
        'efficientnet_b3': CombinedEfficientNetB3Classifier
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Model {model_name} not supported. Available: {list(models_dict.keys())}")
    
    return models_dict[model_name](n_classes, glcm_size, dropout_rates)