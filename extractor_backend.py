import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

class SatelliteFeatureExtractor:
    def __init__(self):
        # Using a model pre-trained on millions of shapes
        self.model = models.resnet50(pretrained=True)
        # Strip away the final classification layer so we get raw features (2048 dimensions)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.eval()

        # Normalization steps required for deep learning models
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_vector(self, rgb_array):
        # Convert our normalized 0-255 numpy array into a PIL Image format
        pil_img = Image.fromarray(rgb_array)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
            
        tensor = self.transform(pil_img).unsqueeze(0)
        with torch.no_grad():
            vector = self.model(tensor)
        
        return vector.flatten().numpy()