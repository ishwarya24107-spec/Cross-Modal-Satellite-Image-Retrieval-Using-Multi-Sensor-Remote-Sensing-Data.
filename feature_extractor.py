import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class ImageEncoder:
    def __init__(self):
        # We use a pre-trained ResNet-50 model. It already knows how to recognize shapes perfectly.
        self.model = models.resnet50(pretrained=True)
        # Remove the final classification layer because we don't want a label like "dog" or "cat", 
        # we want the raw mathematical layout vector instead.
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.eval() # Set model to evaluation mode

        # Image transformations required by deep learning models
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_vector(self, pil_image):
        # If the image is grayscale (like our cleaned SAR image), convert it to RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            
        tensor = self.transform(pil_image).unsqueeze(0)
        with torch.no_grad():
            vector = self.model(tensor)
        # Flatten the vector into a standard 1D array of numbers
        return vector.flatten().numpy()