from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import segmentation_models_pytorch as smp

image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_model():
    model = smp.Unet(
        encoder_name = 'resnet50',
        encoder_weights = 'imagenet',
        in_channels = 3,
        classes = 1
    )
    return model

def prediction(model, image, threshold=0.5):
    image_tensor = image_transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output) > 0.5
        output =(output.squeeze() > threshold).cpu().numpy()
    return output
