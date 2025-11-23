# DancePose

Модуль извлечения поз из видео с помощью YOLOv8-Pose.

## Использование

### Из командной строки:
```bash
python -m dancepose.scripts.run_pose --video video.mp4 --output outputs/my_video
```

### Из Python кода:
```python
from dancepose.scripts.run_pose import main

main(video_path="video.mp4", output_dir="outputs/my_video")
```

### Асинхронная версия:
```python
from dancepose.scripts.run_pose_async import process_video_async

result = await process_video_async("video.mp4", "outputs/my_video")
```

## Структура

- src/inference/pose_infer.py - загрузка модели YOLOv8 и детекция поз
- src/utils/io_utils.py - операции ввода/вывода
- src/viz/overlay.py - визуализация скелета на видео
- scripts/run_pose.py - основной скрипт
- scripts/run_pose_async.py - асинхронная версия

## Формат вывода

Результаты сохраняются в poses.jsonl - каждая строка это JSON с ключевыми точками:
- frame_idx: номер кадра
- valid: валидность детекции
- keypoints: массив из 17 точек [[x, y, confidence], ...]

Модель YOLOv8 скачивается автоматически при первом использовании.
