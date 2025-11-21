# Dance Movement Classifier

Классификация танцевальных движений с оценкой качества исполнения.

## Быстрый старт

### 1. Установка

```bash
git clone https://github.com/citec-spbu/coach-assistant.git
cd coach-assistant/dance_classifier
pip install -r requirements.txt
```

### 2. Скачать модель

**Облако:** https://cloud.mail.ru/public/DG9E/EuZWkp6gn

Скачать файлы:
- `models/best_model_20pct.pth` → положить в `dance_classifier/models/`
- `yolov8m-pose.pt` → положить в `coach-assistant/`

### 3. Использование

```python
from dance_classifier.inference.predict import DanceClassifierPredictor

predictor = DanceClassifierPredictor(
    model_path="dance_classifier/models/best_model_20pct.pth",
    metadata_path="dance_classifier/data/metadata.json",
    scaler_path="dance_classifier/data/scaler.pkl",
    label_encoder_path="dance_classifier/data/label_encoder.pkl"
)

result = predictor.predict_from_poses("poses.jsonl")
```

## Выход

```json
{
    "predicted_class": "FootChange",
    "confidence": 0.806,
    "spatial_similarity": {"score": 78.7},
    "timing": {"score": 76.2},
    "balance": {"score": 97.4},
    "classifier_clarity": {"score": 86.7}
}
```

## Метрики

- **spatial_similarity**: Техника (DTW с эталоном)
- **timing**: Тайминг (шаги vs музыка)
- **balance**: Баланс (стабильность)
- **classifier_clarity**: Уверенность

## Модель

- **Архитектура:** TCN + GRU
- **Accuracy:** 63%
- **Classes:** 13 фигур латины

## Структура

```
dance_classifier/
├── data_preparation/    # Подготовка датасета
├── models/             # Архитектуры моделей
├── inference/          # Предсказание
├── utils/              # DTW-метрики
└── data/               # metadata, scaler, encoder
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV
- librosa
- fastdtw

Полный список: `requirements.txt`

## Ссылки

- **Модель:** https://cloud.mail.ru/public/DG9E/EuZWkp6gn
- **GitHub:** https://github.com/citec-spbu/coach-assistant
