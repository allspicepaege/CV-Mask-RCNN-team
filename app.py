import streamlit as st

st.set_page_config(
    page_title="О проекте",
    page_icon="👋",
)

st.markdown(
    """
    <style>
        button[title^=Exit]+div [data-testid=stImage]{
            test-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)

st.markdown(
    '<h3 style="text-align: center;">Наша команда</h3>',
    unsafe_allow_html=True
)

left_co, cent_co, last_co = st.columns(3)
with cent_co:
    st.image('./images/team.jpg', width=400)
st.write('')

st.sidebar.success("Выберите страницу.")

st.markdown("""
## Микросервис с демонстрациями работы моделей детекции и сегментации.

**Авторы:** [Илья Крючков](https://github.com/xefr762), [Влад Мороз](https://github.com/VladLegenda), 
            [Нанзат Дашиев](https://github.com/nanzat), [Михаил Бутин](https://github.com/allspicepaege)

**Описание:**
- **Главная страница**: Общая информация и навигация 🌠
- **Первая страница**: Детекция судов на изображениях аэросъёмки (Модель YOLO)⛴️
- **Вторая страница**: Семантическая сегментация аэрокосмических снимков Земли (Модель Unet)🌌 
- **Третья страница**: Детекция лиц с последующей маскировкой детектированной области (Модель YOLO11)🙈
- **Третья страница**: Метрики моделей, время их обучения, описание проблем 📊 

Переключайтесь между страницами через левый сайдбар! 
""")

