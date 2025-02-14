from ultralytics import YOLO
import requests
import numpy as np
from PIL import Image, ImageFilter
from io import BytesIO

#model = YOLO("./models/model_1/best.pt")

def blur(model, images):
    results = model.predict(images, conf=0.5)

    blured_images = []
    for i, result in enumerate(results):
        blured_image = images[i].copy()
        for box in result.boxes:
            x1, y1, x2, y2 = map(round, box.xyxy.flatten().tolist())
            crop = blured_image.crop((x1, y1, x2, y2))
            blured_crop = crop.filter(ImageFilter.GaussianBlur(15))
            blured_image.paste(blured_crop, (x1, y1))
        blured_images.append(blured_image)
    
    return blured_images
