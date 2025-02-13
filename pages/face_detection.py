#imports
import streamlit as st
import time
import torch
import requests

#from's
from io import BytesIO
from PIL import Image
from ultralytics import YOLO

st.markdown(
    '<h1 style="text-align: center;">Модель, которая детектирует и маскирует лица на ваших картинках</h1>',
    unsafe_allow_html=True
)
st.write('**Пользователь загружает картинку (или несколько) в модель. Модель определяет лицо на картинке и маскирует его.**')

#upload form
uploaded_files = st.file_uploader('Загрузите изображение', type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
image_url = st.text_input('Вставьте прямую ссылку на изображение')

if uploaded_files:
    image = []
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file)
        image.append(img)
elif image_url:
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        st.error(f'Не удалось загрузить изображение по ссылке. Ошибка {e}')
else:
    st.write('Выберите способ загрузки изображения!')

######################
### MODEL
######################

@st.cache_resourse()
def load_model(): # Загружаем наши веса в модель
    model = YOLO('models/model_1/test.pt')
    return model

model = load_model()

def predict(img):
    pred = model(img)
    return pred

######################
### PREDICTION
######################

if image:
    start_time = time.time()

    if isinstance(image, list):
        for img in image:
            st.image(img, caption='Ваше изображение')
            prediction = predict(img)
            st.write('Ваше предсказание')
            st.markdown(f"""
            <h2 style='text-align: center; font-size: 30px, font-weight: bold, padding: 10px; bolder-radius: 10px;'> 
            {prediction}
            </h2>
            """, unsafe_allow_html=True)
    else:
        st.image(img, caption='Ваше изображение')
        prediction = predict(img)
        st.write('Ваше предсказание')
        st.markdown(f"""
        <h2 style='text-align: center; font-size: 30px, font-weight: bold, padding: 10px; bolder-radius: 10px;'> 
        {prediction}
        </h2>
        """, unsafe_allow_html=True)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    st.write('Время выполнения предсказания в секундах:')
    st.markdown(f"""
    <h3 style='text-align: center; font-size: 30px; font-weight: bold; padding: 5px; bolder-radius:5px;'>
    {elapsed_time:.2f}
    </h3>
    """, unsafe_allow_html=True)

else:
    st.stop()