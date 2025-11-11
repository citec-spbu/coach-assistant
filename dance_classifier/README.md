# Классификатор танцевальных фигур

## Описание
Система классификации танцевальных фигур на основе поз, извлеченных из видео с помощью модуля DancePose.

## Структура проекта
```
dance_classifier/
├── data_preparation/
│   ├── extract_poses.py          # Извлечение поз из видео
│   ├── feature_extraction.py     # Извлечение признаков
│   └── dataset_builder.py        # Подготовка датасета
├── models/
│   ├── gru_classifier.py         # GRU модель
│   └── tcn_classifier.py         # TCN модель
├── training/
│   ├── train.py                  # Скрипт обучения
│   └── config.yaml               # Конфигурация
├── inference/
│   └── predict.py                # Инференс
├── notebooks/
│   └── analysis.ipynb            # Анализ и визуализация
└── utils/
    └── helpers.py                # Вспомогательные функции
```

## Признаки
1. **Углы суставов**: Локти, колени, бедра
2. **Расстояния**: Между ключевыми точками (руки, ноги)
3. **Скорости**: Изменение положения суставов между кадрами

## Использование
1. Извлечение поз: `python data_preparation/extract_poses.py`
2. Подготовка данных: `python data_preparation/dataset_builder.py`
3. Обучение: `python training/train.py --config training/config.yaml`
4. Инференс: `python inference/predict.py --video path/to/video.mp4`



## Описание
Система классификации танцевальных фигур на основе поз, извлеченных из видео с помощью модуля DancePose.

## Структура проекта
```
dance_classifier/
├── data_preparation/
│   ├── extract_poses.py          # Извлечение поз из видео
│   ├── feature_extraction.py     # Извлечение признаков
│   └── dataset_builder.py        # Подготовка датасета
├── models/
│   ├── gru_classifier.py         # GRU модель
│   └── tcn_classifier.py         # TCN модель
├── training/
│   ├── train.py                  # Скрипт обучения
│   └── config.yaml               # Конфигурация
├── inference/
│   └── predict.py                # Инференс
├── notebooks/
│   └── analysis.ipynb            # Анализ и визуализация
└── utils/
    └── helpers.py                # Вспомогательные функции
```

## Признаки
1. **Углы суставов**: Локти, колени, бедра
2. **Расстояния**: Между ключевыми точками (руки, ноги)
3. **Скорости**: Изменение положения суставов между кадрами

## Использование
1. Извлечение поз: `python data_preparation/extract_poses.py`
2. Подготовка данных: `python data_preparation/dataset_builder.py`
3. Обучение: `python training/train.py --config training/config.yaml`
4. Инференс: `python inference/predict.py --video path/to/video.mp4`


