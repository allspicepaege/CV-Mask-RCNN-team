#imports
import streamlit as st
import time
import torch
import requests
import segmentation_models_pytorch as smp
import numpy as np
import cv2

#from's
from io import BytesIO
from PIL import Image
from ultralytics import YOLO

from models.model_3.model_unet import prediction
from models.model_3.model_unet import get_model

st.set_page_config(
    page_title="Сегментация леса",
    page_icon="🌲",
)

st.markdown(
    '<h1 style="text-align: center;">Модель, сегментирующая лес на аэроснимках</h1>',
    unsafe_allow_html=True
)

# Форма загрузки изображения
uploaded_files = st.file_uploader('Загрузите изображение', type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
image_url = st.text_input('Вставьте прямую ссылку на изображение')

# Обработка загруженных изображений
images = []

if uploaded_files:
    images = [Image.open(file) for file in uploaded_files]

elif image_url:
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        images.append(img)
    except Exception as e:
        st.error(f'Не удалось загрузить изображение по ссылке. Ошибка: {e}')

if not images:
    st.warning('Выберите способ загрузки изображения!')
    st.stop()

######################
### MODEL
######################

@st.cache_resource()
def load_model():    
    model = get_model()
    model.load_state_dict(torch.load('./models/model_3/best.pt'))
    model.eval()
    return model

model = load_model()

######################
### MASK FUNCTION
######################

def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):

    image = np.array(image.convert("RGB"))  # Преобразуем в RGB
    
    # Меняем размер маски под размер изображения
    mask_resized = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Создаём цветную маску
    color_mask = np.zeros_like(image)
    color_mask[mask_resized == 1] = color  

    # Накладываем с прозрачностью
    overlayed = cv2.addWeighted(image, 1, color_mask, alpha, 0)

    return Image.fromarray(overlayed)

######################
### PREDICTION
######################

start_time = time.time()

for img in images:
    st.image(img, caption='Исходное изображение', use_container_width=True)

    # Получаем маску предсказания
    mask = prediction(model, img)

    # Накладываем маску
    masked_image = overlay_mask(img, mask)

    st.image(masked_image, caption='Сегментированное изображение', use_container_width=True)

end_time = time.time()
elapsed_time = end_time - start_time

st.write('⏳ Время выполнения предсказания (сек):')
st.markdown(f"""
<h3 style='text-align: center; font-size: 30px; font-weight: bold; padding: 5px; border-radius:5px;'>
{elapsed_time:.2f}
</h3>
""", unsafe_allow_html=True)