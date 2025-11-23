# Классификатор танцевальных движений

Система распознавания танцевальных фигур на основе анализа поз человека в видео.

## Компоненты

1. DancePose - извлечение поз из видео с помощью YOLOv8-Pose
2. Dance Classifier - классификация танцевальных движений с помощью GRU/TCN/Hybrid моделей

## Установка

```bash
# Установка зависимостей для классификатора
pip install -r dance_classifier/requirements.txt

# Установка зависимостей для извлечения поз
pip install -r dancepose/requirements.txt
```

## Использование

### Извлечение поз из видео

```python
from dancepose.scripts.run_pose import main

main(video_path="video.mp4", output_dir="outputs/my_video")
```

### Классификация движений

```python
from dance_classifier.inference.predict import DanceClassifierPredictor

predictor = DanceClassifierPredictor("dance_classifier/best_model_20pct.pth")
result = predictor.predict_from_poses("poses.jsonl", video_path="video.mp4")
```

Результат содержит:
- predicted_class: название фигуры
- confidence: уверенность модели (0-1)
- spatial_similarity: техника исполнения (score 0-100)
- classifier_clarity: разборчивость фигуры (score 0-100)
- timing: ритм (score 0-100, если есть video_path)
- balance: баланс (score 0-100)

## Структура проекта

```
coach-assistant/
├── dancepose/                    # Извлечение поз
│   ├── src/inference/           # YOLOv8 инференс
│   ├── src/utils/               # Утилиты
│   ├── src/viz/                 # Визуализация
│   └── scripts/                  # Скрипты запуска
│
├── dance_classifier/            # Классификация движений
│   ├── data_preparation/        # Подготовка данных
│   ├── models/                  # GRU, TCN, Hybrid модели
│   ├── training/                # Обучение
│   ├── inference/               # Предсказание
│   ├── utils/                   # DTW-метрики качества
│   └── reference_trajectories/  # Эталонные траектории
│
└── outputs/                     # Результаты обработки
```

## Модели

Доступны три архитектуры:
- GRU - рекуррентная нейронная сеть
- TCN - временная свёрточная сеть
- Hybrid - комбинация TCN + GRU

Выбор модели в конфиге: `model_type: "gru"` / `"tcn"` / `"hybrid"`

## Требования

- Python 3.8+
- PyTorch 1.12+
- OpenCV
- Ultralytics YOLOv8
- NumPy, Pandas, Scikit-learn

## Обучение

```bash
cd dance_classifier
python training/train.py --config training/config.yaml
```

## Метрики качества

Модель возвращает 4 метрики качества исполнения:
- Spatial Similarity - техника (сравнение с эталоном через DTW)
- Classifier Clarity - разборчивость фигуры
- Timing - ритм (синхронизация с музыкой)
- Balance - баланс и стабильность

Подробнее в файле `ПОНЯТНОЕ_ОБЪЯСНЕНИЕ_МЕТРИК_И_ПРОЦЕССОВ.md`

## Важные файлы

- `ИНСТРУКЦИЯ_ДЛЯ_ФРОНТЕНДА.md` - подробная инструкция для фронтенд-разработчика
- `СООБЩЕНИЕ_ДЛЯ_ФРОНТЕНДА.txt` - краткое сообщение с примерами использования
- `use_classifier.py` - простой скрипт для быстрого использования классификатора

## Модель

Модель `best_model_20pct.pth` должна быть загружена отдельно из облака (слишком большая для GitHub).
Ссылка: https://cloud.mail.ru/public/DG9E/EuZWkp6gn

Эталонные траектории уже включены в репозиторий: `dance_classifier/reference_trajectories/`
