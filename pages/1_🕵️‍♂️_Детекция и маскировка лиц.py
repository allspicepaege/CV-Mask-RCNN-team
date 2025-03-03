#imports
import streamlit as st
import time
import torch
import requests
import cv2
import numpy as np

#from's
from io import BytesIO
from PIL import Image
from ultralytics import YOLO

from models.model_1.blur import blur

st.set_page_config(
    page_title="Детекция и маскировка лиц",
    page_icon="🕵️‍♂️",
)

st.markdown(
    '<h1 style="text-align: center;">Модель, которая детектирует и маскирует лица на ваших картинках</h1>',
    unsafe_allow_html=True
)
st.write('**Пользователь загружает картинку (или несколько) в модель. Модель определяет лицо на картинке и маскирует его.**')

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
    return YOLO('./models/model_1/best.pt')

model = load_model()

def predict_image(img):
    """Функция для предсказания и наложения маски."""
    img_cv2 = np.array(img)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
    
    result_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    blurred = blur(model=model, images=[result_pil])
    return blurred

######################
### PREDICTION
######################

start_time = time.time()

for img in images:
    st.image(img, caption='Ваше изображение', use_container_width=True)

    pred_img = predict_image(img)

    st.write('Ваше предсказание:')
    st.image(pred_img, caption='Результат', use_container_width=True)

end_time = time.time()
elapsed_time = end_time - start_time

st.write('⏳ Время выполнения предсказания (сек):')
st.markdown(f"""
<h3 style='text-align: center; font-size: 30px; font-weight: bold; padding: 5px; border-radius:5px;'>
{elapsed_time:.2f}
</h3>
""", unsafe_allow_html=True)