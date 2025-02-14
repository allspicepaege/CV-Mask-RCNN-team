from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import segmentation_models_pytorch as smp

def get_model():
    model = smp.Unet(
        encoder_name = 'resnet50',
        encoder_weights = 'imagenet',
        in_channels = 3,
        classes = 1
    )
    return model

def prediction(model, image):
    model.eval()
    image_tensor = transforms(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = torch.sigmoid(output).cpu().numpy()[0][0] > 0.5 

    return pred_mask
