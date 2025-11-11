"""
Скрипт извлечения поз из видео датасета с использованием DancePose
"""
import os
import sys
import json
import yaml
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse

# Добавляем путь к модулю dancepose
sys.path.append(str(Path(__file__).parent.parent.parent / "dancepose"))

from src.inference.pose_infer import PoseExtractor
from src.utils.io_utils import ensure_dir, JsonlWriter


class PoseDatasetExtractor:
    """Класс для извлечения поз из видео датасета"""
    
    def __init__(self, model_path, device="0", conf=0.25, iou=0.5):
        """
        Args:
            model_path: путь к модели YOLOv8-Pose
            device: устройство для inference ("0" для GPU, "cpu" для CPU)
            conf: порог уверенности
            iou: порог IoU
        """
        self.pose_extractor = PoseExtractor(
            model_name=model_path,
            device=device,
            imgsz=640,
            conf=conf,
            iou=iou,
            vid_stride=1
        )
    
    def extract_from_video(self, video_path, output_path=None):
        """
        Извлекает позы из одного видео
        
        Args:
            video_path: путь к видео файлу
            output_path: путь для сохранения результатов (JSON)
        
        Returns:
            dict: словарь с извлеченными позами
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Ошибка открытия видео: {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1e-3:
            fps = 25.0
        
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        poses_data = {
            'video_path': str(video_path),
            'video_name': Path(video_path).stem,
            'fps': fps,
            'width': W,
            'height': H,
            'total_frames': total_frames,
            'frames': []
        }
        
        frame_idx = 0
        valid_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / fps
            valid, bbox, kps_xyc, kp_mean = self.pose_extractor.infer_frame(frame)
            
            frame_data = {
                'frame_idx': frame_idx,
                'timestamp': round(timestamp, 4),
                'valid': bool(valid)
            }
            
            if valid:
                valid_frames += 1
                # Сохраняем bbox в формате [x, y, w, h]
                frame_data['bbox'] = [
                    float(bbox[0] - bbox[2]/2),
                    float(bbox[1] - bbox[3]/2),
                    float(bbox[2]),
                    float(bbox[3])
                ]
                # Сохраняем keypoints в формате [[x, y, score], ...]
                frame_data['keypoints'] = [
                    [float(x), float(y), float(s)] for x, y, s in kps_xyc
                ]
                frame_data['confidence'] = float(kp_mean)
            
            poses_data['frames'].append(frame_data)
            frame_idx += 1
        
        cap.release()
        
        poses_data['valid_frames'] = valid_frames
        poses_data['coverage'] = valid_frames / max(1, total_frames)
        
        # Сохраняем результаты
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(poses_data, f, ensure_ascii=False, indent=2)
        
        return poses_data
    
    def extract_from_directory(self, video_dir, output_dir, video_extensions=None):
        """
        Извлекает позы из всех видео в директории
        
        Args:
            video_dir: директория с видео файлами
            output_dir: директория для сохранения результатов
            video_extensions: список расширений видео файлов
        """
        if video_extensions is None:
            video_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.MOV']
        
        video_dir = Path(video_dir)
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        
        # Находим все видео файлы
        video_files = []
        for ext in video_extensions:
            video_files.extend(video_dir.rglob(f'*{ext}'))
        
        print(f"Найдено {len(video_files)} видео файлов")
        
        results = []
        for video_path in tqdm(video_files, desc="Обработка видео"):
            output_path = output_dir / f"{video_path.stem}_poses.json"
            
            # Пропускаем, если уже обработано
            if output_path.exists():
                print(f"Пропускаем {video_path.name} (уже обработано)")
                continue
            
            poses_data = self.extract_from_video(video_path, output_path)
            if poses_data:
                results.append({
                    'video_name': video_path.stem,
                    'video_path': str(video_path),
                    'output_path': str(output_path),
                    'valid_frames': poses_data['valid_frames'],
                    'total_frames': poses_data['total_frames'],
                    'coverage': poses_data['coverage']
                })
        
        # Сохраняем сводку
        summary_path = output_dir / 'extraction_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nОбработано {len(results)} видео")
        print(f"Результаты сохранены в: {output_dir}")
        print(f"Сводка: {summary_path}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Извлечение поз из видео датасета")
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Директория с видео файлами')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Директория для сохранения результатов')
    parser.add_argument('--model_path', type=str,
                        default='../dancepose/models/yolov8s-pose.pt',
                        help='Путь к модели YOLOv8-Pose')
    parser.add_argument('--device', type=str, default='0',
                        help='Устройство (0 для GPU, cpu для CPU)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Порог уверенности')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='Порог IoU')
    
    args = parser.parse_args()
    
    # Создаем экстрактор
    extractor = PoseDatasetExtractor(
        model_path=args.model_path,
        device=args.device,
        conf=args.conf,
        iou=args.iou
    )
    
    # Извлекаем позы из всех видео
    results = extractor.extract_from_directory(
        video_dir=args.video_dir,
        output_dir=args.output_dir
    )
    
    print("\nГотово!")


if __name__ == "__main__":
    main()


Скрипт извлечения поз из видео датасета с использованием DancePose
"""
import os
import sys
import json
import yaml
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse

# Добавляем путь к модулю dancepose
sys.path.append(str(Path(__file__).parent.parent.parent / "dancepose"))

from src.inference.pose_infer import PoseExtractor
from src.utils.io_utils import ensure_dir, JsonlWriter


class PoseDatasetExtractor:
    """Класс для извлечения поз из видео датасета"""
    
    def __init__(self, model_path, device="0", conf=0.25, iou=0.5):
        """
        Args:
            model_path: путь к модели YOLOv8-Pose
            device: устройство для inference ("0" для GPU, "cpu" для CPU)
            conf: порог уверенности
            iou: порог IoU
        """
        self.pose_extractor = PoseExtractor(
            model_name=model_path,
            device=device,
            imgsz=640,
            conf=conf,
            iou=iou,
            vid_stride=1
        )
    
    def extract_from_video(self, video_path, output_path=None):
        """
        Извлекает позы из одного видео
        
        Args:
            video_path: путь к видео файлу
            output_path: путь для сохранения результатов (JSON)
        
        Returns:
            dict: словарь с извлеченными позами
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Ошибка открытия видео: {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1e-3:
            fps = 25.0
        
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        poses_data = {
            'video_path': str(video_path),
            'video_name': Path(video_path).stem,
            'fps': fps,
            'width': W,
            'height': H,
            'total_frames': total_frames,
            'frames': []
        }
        
        frame_idx = 0
        valid_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / fps
            valid, bbox, kps_xyc, kp_mean = self.pose_extractor.infer_frame(frame)
            
            frame_data = {
                'frame_idx': frame_idx,
                'timestamp': round(timestamp, 4),
                'valid': bool(valid)
            }
            
            if valid:
                valid_frames += 1
                # Сохраняем bbox в формате [x, y, w, h]
                frame_data['bbox'] = [
                    float(bbox[0] - bbox[2]/2),
                    float(bbox[1] - bbox[3]/2),
                    float(bbox[2]),
                    float(bbox[3])
                ]
                # Сохраняем keypoints в формате [[x, y, score], ...]
                frame_data['keypoints'] = [
                    [float(x), float(y), float(s)] for x, y, s in kps_xyc
                ]
                frame_data['confidence'] = float(kp_mean)
            
            poses_data['frames'].append(frame_data)
            frame_idx += 1
        
        cap.release()
        
        poses_data['valid_frames'] = valid_frames
        poses_data['coverage'] = valid_frames / max(1, total_frames)
        
        # Сохраняем результаты
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(poses_data, f, ensure_ascii=False, indent=2)
        
        return poses_data
    
    def extract_from_directory(self, video_dir, output_dir, video_extensions=None):
        """
        Извлекает позы из всех видео в директории
        
        Args:
            video_dir: директория с видео файлами
            output_dir: директория для сохранения результатов
            video_extensions: список расширений видео файлов
        """
        if video_extensions is None:
            video_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.MOV']
        
        video_dir = Path(video_dir)
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        
        # Находим все видео файлы
        video_files = []
        for ext in video_extensions:
            video_files.extend(video_dir.rglob(f'*{ext}'))
        
        print(f"Найдено {len(video_files)} видео файлов")
        
        results = []
        for video_path in tqdm(video_files, desc="Обработка видео"):
            output_path = output_dir / f"{video_path.stem}_poses.json"
            
            # Пропускаем, если уже обработано
            if output_path.exists():
                print(f"Пропускаем {video_path.name} (уже обработано)")
                continue
            
            poses_data = self.extract_from_video(video_path, output_path)
            if poses_data:
                results.append({
                    'video_name': video_path.stem,
                    'video_path': str(video_path),
                    'output_path': str(output_path),
                    'valid_frames': poses_data['valid_frames'],
                    'total_frames': poses_data['total_frames'],
                    'coverage': poses_data['coverage']
                })
        
        # Сохраняем сводку
        summary_path = output_dir / 'extraction_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nОбработано {len(results)} видео")
        print(f"Результаты сохранены в: {output_dir}")
        print(f"Сводка: {summary_path}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Извлечение поз из видео датасета")
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Директория с видео файлами')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Директория для сохранения результатов')
    parser.add_argument('--model_path', type=str,
                        default='../dancepose/models/yolov8s-pose.pt',
                        help='Путь к модели YOLOv8-Pose')
    parser.add_argument('--device', type=str, default='0',
                        help='Устройство (0 для GPU, cpu для CPU)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Порог уверенности')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='Порог IoU')
    
    args = parser.parse_args()
    
    # Создаем экстрактор
    extractor = PoseDatasetExtractor(
        model_path=args.model_path,
        device=args.device,
        conf=args.conf,
        iou=args.iou
    )
    
    # Извлекаем позы из всех видео
    results = extractor.extract_from_directory(
        video_dir=args.video_dir,
        output_dir=args.output_dir
    )
    
    print("\nГотово!")


if __name__ == "__main__":
    main()


