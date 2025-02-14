import streamlit as st

st.set_page_config(
    page_title="Метрики",
    page_icon="📊",
)

st.title('Метрики при обучении разных моделей')

tab1, tab2, tab3 = st.tabs(['Модель 1', 'Модель 2', 'Модель 3'])
col1, col2, col3 = st.columns(3)

with tab1:
    st.header('Отчёт о работе 1 модели 🌚')
    st.markdown(
        """
        <h3 style="text-align: center;">Графики - (Не завезли) - придётся поверить нам на слова</h3>
        <h3 style="text-align: center;">mAP 50 = 0.902</h3>
        <h3 style="text-align: center;">mAP 50-95 = 0.609</h3>
        <h3 style="text-align: center;">Epochs 35, 13300 на train, 3300 на test</h3>
        """,
        unsafe_allow_html=True
    )
    st.write('Спасибо огромное гугл-колабу, мы его так сильно любим')

with tab2:
    st.header('Отчёт о работе 2 модели 🌚')
    st.markdown(
        '<h3 style="text-align: center;">Основные графики обучения модели</h3>',
        unsafe_allow_html=True
    )
    st.image('/home/marena/Elbrus_phase_2/CV-Mask-RCNN-team/images/metrics_2/results.png', 
             caption='Losses and Maps', width=800)

    st.markdown(
        """
        <h3 style="text-align: center;">30 эпох первичного обучения</h3>
        <h3 style="text-align: center;">Основные графики дообучения модели</h3>
        """,
        unsafe_allow_html=True
    )

    st.image('/home/marena/Elbrus_phase_2/CV-Mask-RCNN-team/images/metrics_2/results_ft.png', 
             caption='Losses and Maps', width=800)

    st.markdown(
        '<h3 style="text-align: center;">50 эпох дообучения</h3>',
        unsafe_allow_html=True
    )

with tab3:
    st.header('Отчёт о работе 3 модели 🍺')
    st.write('Основные метрики обучения модели')
    st.image('/home/marena/Elbrus_phase_2/CV-Mask-RCNN-team/images/metrics_3.jpg', caption='Metrics', width=800)
    st.write('Feat. by Vlad Legenda')