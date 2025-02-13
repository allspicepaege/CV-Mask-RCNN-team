# Computer Vision project

Этот проект представляет собой веб-приложение на **Streamlit**, имеющее 3 страницы с функционалом различных сетей компьютерного зрения. Как детекции, так и сегментации

За основу были взяты следующие модели:
- YOLO11
- Unet
- SAM

## Функционал
- **Загрузка изображения**: можно загружать любые фото как с компьютера (одну или несколько), так и по прямой ссылке.
- **Отображение времени предсказания**: можно увидеть, насколько быстро модели справляются с задачей.
- **Возвращение картинки с детекцией/сегментацией**: можно увидеть картинки с выделенными зонами сегментации/детекции.
- **User-friendly интерфейс**: интуитивно понятный интерфейс для невовлеченного в процесс пользователя.

## Установка и запуск
```bash
git clone https://github.com/xefr762/nn_project.git
cd nn_project/
pip install -r requirements.txt
streamlit run app.py
```

## Структура проекта

(Основная структура на данный момент - будет докручиваться по мере дальнейшего развития проекта)

- **models/** - папка с файлами и весами моделей
- **images/** - папка с используемыми в streamlit иллюстрациями
- **notebooks/** - папка с кодом создания и настройки моделей
- **pages/** - папка со страницами для Streamlit
- **app.py** - основной файл для запуска приложения Streamlit
- **README.md** - файл описания проекта
- **.gitignrore** - игнорируемые для загрузки файлы


## Краткое описание функционала каждой модели

1. Детекция [лиц](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset) с помощью YOLO11, с последующей маскировкой детектированной области пример работы - [тык](https://github.com/Elbrus-DataScience/cv_mask-rcnn/blob/master/SCR-20240807-kgpq.png) 

2. Детекция [судов_на_изображениях_аэросъёмки](https://www.kaggle.com/datasets/siddharthkumarsah/ships-in-aerial-images). Используемая модель - YOLO11

3. Семантическая сегментация [аэрокосмических_снимков](https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation). 
Реализовано в двух вариантах: 
    - Unet
    - SAM  