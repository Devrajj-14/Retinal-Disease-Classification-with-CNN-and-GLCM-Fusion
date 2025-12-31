"""GLCM Feature Extractor for texture analysis"""

import numpy as np
import cv2
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
import warnings

warnings.filterwarnings('ignore')

class GLCMFeatureExtractor:
    """GLCM Feature Extractor optimized for retinal image classification"""
    
    def __init__(self, distances=[1, 2], angles=[0, 45, 90, 135], levels=32):
        self.distances = distances
        self.angles = [np.deg2rad(angle) for angle in angles]
        self.levels = levels
        print(f"🔬 GLCM Extractor: {len(distances)} distances, {len(angles)} angles, {levels} levels")
    
    def extract_glcm_features(self, image):
        """Extract compact GLCM features from PIL image"""
        try:
            # Convert PIL to numpy and preprocess
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
            
            # Use green channel for better retinal contrast
            if len(img_array.shape) == 3:
                gray = img_array[:, :, 1]  # Green channel
            else:
                gray = img_array
            
            # Resize and enhance
            gray = cv2.resize(gray, (128, 128))  # Smaller for speed
            gray = cv2.equalizeHist(gray)
            
            # Reduce gray levels
            gray = (gray // (256 // self.levels)).astype(np.uint8)
            
            # Extract GLCM features
            features = []
            for distance in self.distances:
                glcm = graycomatrix(gray, 
                                 distances=[distance], 
                                 angles=self.angles, 
                                 levels=self.levels, 
                                 symmetric=True, 
                                 normed=True)
                
                # Core texture properties
                contrast = graycoprops(glcm, 'contrast').flatten()
                homogeneity = graycoprops(glcm, 'homogeneity').flatten()
                energy = graycoprops(glcm, 'energy').flatten()
                
                # Add mean and std for each property
                features.extend([
                    np.mean(contrast), np.std(contrast),
                    np.mean(homogeneity), np.std(homogeneity),
                    np.mean(energy), np.std(energy)
                ])
            
            return np.array(features, dtype=np.float32)
        
        except Exception as e:
            print(f"❌ GLCM extraction error: {e}")
            return np.zeros(len(self.distances) * 6, dtype=np.float32)  # 6 features per distance
    
    def get_feature_names(self):
        """Get feature names for interpretation"""
        feature_names = []
        for i, distance in enumerate(self.distances):
            feature_names.extend([
                f'contrast_mean_d{distance}',
                f'contrast_std_d{distance}',
                f'homogeneity_mean_d{distance}',
                f'homogeneity_std_d{distance}',
                f'energy_mean_d{distance}',
                f'energy_std_d{distance}'
            ])
        return feature_names
    
    def get_feature_size(self):
        """Get the size of feature vector"""
        return len(self.distances) * 6