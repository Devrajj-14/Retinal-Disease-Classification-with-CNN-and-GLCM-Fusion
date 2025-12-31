"""FastViT model wrapper for OCT classification"""

import torch
import torch.nn as nn
from timm.models import create_model

class FastViTModel:
    """FastViT model wrapper for multi-class classification"""
    
    def __init__(self, model_name="fastvit_t8", num_classes=8, pretrained=True):
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def build_model(self):
        """Build FastViT model for classification"""
        print(f"\n=== BUILDING FASTVIT MODEL: {self.model_name} ===")
        print(f"Device: {self.device}")
        
        # Create FastViT model
        model = create_model(
            self.model_name,
            pretrained=self.pretrained,
            num_classes=self.num_classes
        )
        
        # Move to device
        model = model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model architecture: {self.model_name}")
        
        return model
    
    @staticmethod
    def get_available_models():
        """Get list of available FastViT models"""
        return [
            "fastvit_t8", "fastvit_t12", "fastvit_s12", 
            "fastvit_sa12", "fastvit_sa24", "fastvit_sa36", 
            "fastvit_ma36"
        ]
    
    @staticmethod
    def reparameterize_model(model):
        """Reparameterize model for inference optimization"""
        model.eval()
        # Note: Actual reparameterization would depend on FastViT implementation
        # This is a placeholder for the reparameterization logic
        return model