import os
import sys
import argparse
import time
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

# Добавляем родительскую директорию в sys.path для импорта модулей
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    from src.inference.pose_infer import PoseExtractor
    from src.utils.io_utils import ensure_dir, JsonlWriter
    from src.utils.filters import PoseStabilizer  # Используем наш новый фильтр
except ImportError as e:
    print(f"Import Error: {e}")
    print("Пожалуйста, запустите этот скрипт из директории dancepose или настройте PYTHONPATH.")
    sys.exit(1)

def process_single_video(video_path: Path, output_root: Path, model_name: str, use_stabilizer: bool):
    """
    Функция обработки одного видео.
    """
    video_name = video_path.stem
    output_dir = output_root / video_name
    ensure_dir(output_dir)
    
    print(f"[{video_name}] Начало обработки...")
    
    # Инициализация
    extractor = PoseExtractor(model_name=model_name, device='cpu')
    stabilizer = PoseStabilizer() if use_stabilizer else None
    json_writer = JsonlWriter(output_dir / "poses.jsonl")
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    processed_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Инференс (распознавание)
        valid, bbox, kps_xyc, score = extractor.infer_frame(frame)
        
        if valid and kps_xyc is not None:
            # Если включен стабилизатор, применяем фильтрацию
            if stabilizer:
                kps_xyc = stabilizer.update(kps_xyc)
                
            # Запись данных
            record = {
                "frame": processed_count,
                "timestamp": processed_count / 30.0, # Предполагаем 30fps
                "keypoints": kps_xyc.tolist(),
                "bbox": bbox.tolist(),
                "score": score
            }
            json_writer.write(record)
            
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"[{video_name}] Обработано {processed_count}/{total_frames} кадров")
            
    cap.release()
    json_writer.close()
    
    elapsed = time.time() - start_time
    print(f"[{video_name}] Завершено за {elapsed:.2f} сек. Сохранено в {output_dir}")
    return video_name, processed_count

def main():
    parser = argparse.ArgumentParser(description="Скрипт пакетной обработки видео танцев")
    parser.add_argument("--input_dir", type=str, required=True, help="Путь к папке с видео")
    parser.add_argument("--output_dir", type=str, required=True, help="Путь к папке для сохранения результатов")
    parser.add_argument("--model", type=str, default="yolov8s-pose.pt", help="Имя модели YOLO")
    parser.add_argument("--stabilize", action="store_true", help="Включить стабилизацию OneEuro")
    parser.add_argument("--parallel", type=int, default=1, help="Количество потоков для параллельной обработки")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    # Поиск видеофайлов
    extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_files = []
    for ext in extensions:
        video_files.extend(list(input_path.glob(ext)))
        
    if not video_files:
        print(f"Видеофайлы не найдены в {input_path}")
        return

    print(f"Найдено {len(video_files)} видео. Начинаем пакетную обработку...")
    print(f"Стабилизация: {'ВКЛ' if args.stabilize else 'ВЫКЛ'}")
    
    # Использование пула потоков (ThreadPool) для пакетной обработки
    results = []
    if args.parallel > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = []
            for vid in video_files:
                futures.append(executor.submit(
                    process_single_video, vid, output_path, args.model, args.stabilize
                ))
            
            for f in futures:
                results.append(f.result())
    else:
        for vid in video_files:
            results.append(process_single_video(vid, output_path, args.model, args.stabilize))
            
    print("\n=== Отчет о пакетной обработке ===")
    for name, count in results:
        print(f"Видео: {name:<20} | Кадры: {count}")
    print("===============================")

if __name__ == "__main__":
    main()
