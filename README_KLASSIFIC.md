# Dance Movement Classifier

Распознавание танцевальных движений из видео.

**Точность:** 78.75% | **Классов:** 14 | **Модель:** GRU

---

## Установка

```bash
git clone https://github.com/YOUR_USERNAME/dance-movement-classifier.git
cd dance-movement-classifier
pip install -r dance_classifier/requirements.txt
```

**Требования:** Python 3.8+

---

## Использование

### Извлечение поз

```python
from dancepose.scripts.run_pose import main

main(video_path="video.mp4", output_dir="outputs/my_video")
```

### Классификация

```python
from dance_classifier.inference.predict import DanceClassifierPredictor

predictor = DanceClassifierPredictor(
    model_path="best_model.pth",
    metadata_path="metadata.json"
)

result = predictor.predict_from_poses("outputs/my_video/poses.jsonl")
print(f"{result['predicted_figure']}: {result['confidence']:.1%}")
```

### Полный pipeline

```python
import asyncio
from dancepose.scripts.run_pose_async import run
from dance_classifier.inference.predict import DanceClassifierPredictor

async def analyze(video_path):
    pose_result = await run(video_path)
    predictor = DanceClassifierPredictor(model_path="best_model.pth", metadata_path="metadata.json")
    result = predictor.predict_from_poses(pose_result['poses_file'])
    print(f"{result['predicted_figure']}: {result['confidence']:.1%}")

asyncio.run(analyze("dance.mp4"))
```

---

## Обучение

```bash
cd dance_classifier
python training/train.py --config training/config.yaml --data_dir dataset --output_dir models
```

**Конфигурация** (`config.yaml`):
```yaml
model_type: 'gru'
batch_size: 32
num_epochs: 50
learning_rate: 0.001
```

---

## Результаты

| Метрика | Значение |
|---------|----------|
| Accuracy | 78.75% |
| F1-score | 76.86% |
| Precision | 77.05% |
| Recall | 78.75% |

Обучено на 10% данных. Ожидается 85-92% на 100%.

---

## Классы (14)

Aida, Alemana, Fan, FootChange, HandToHandL, HandToHandR, HockyStick, NaturalTop, NewYorkL, NewYorkR, NotPerforming, OpenBasic, OpeningOut, SpotTurn

---

## Структура

```
coach-assistant/
├── dancepose/           # Извлечение поз (YOLOv8)
└── dance_classifier/    # Классификация (GRU)
```

---

## Примечания

- Модель не включена (большой размер)
- Датасет не включён
- Для обучения нужен датасет с видео танцев
