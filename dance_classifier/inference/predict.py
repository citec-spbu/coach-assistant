"""
Полноценный предиктор с интеграцией DTW-метрик качества исполнения.

Использование:
    from dance_classifier.inference.predict import DanceClassifierPredictor
    
    predictor = DanceClassifierPredictor(
        model_path="best_model_20pct.pth",
        metadata_path="dataset/metadata.json",
        scaler_path="dataset/scaler.pkl",
        label_encoder_path="dataset/label_encoder.pkl",
        reference_dir=None  # Автоматически найдет в reference_trajectories/
    )
    
    result = predictor.predict_from_poses(
        "poses.jsonl", 
        video_path="video.mp4",
        create_analyzed_video=True,  # Создать analyzed_ видео с метриками
        overlay_video_path="overlay_video.mp4"  # Опционально, ищет автоматически
    )
"""

import torch
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from sklearn.preprocessing import LabelEncoder, StandardScaler
import cv2
from PIL import Image, ImageDraw, ImageFont

# Разрешить загрузку sklearn классов
try:
    torch.serialization.add_safe_globals([LabelEncoder, StandardScaler])
except AttributeError:
    pass


class GRUModel(torch.nn.Module):
    """Простая GRU модель"""
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.gru = torch.nn.GRU(input_size, 64, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = torch.nn.Linear(64, num_classes)
    
    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])


class HybridModel(torch.nn.Module):
    """Гибридная TCN+GRU модель"""
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.tcn = torch.nn.Sequential(
            torch.nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU()
        )
        self.gru = torch.nn.GRU(128, 64, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = torch.nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)
        _, h = self.gru(x)
        return self.fc(h[-1])


class DanceClassifierPredictor:
    """
    Предиктор с полной интеграцией DTW-метрик качества исполнения.
    """
    
    def __init__(
        self,
        model_path: str,
        metadata_path: str,
        scaler_path: str,
        label_encoder_path: str,
        reference_dir: Optional[str] = None,
        device: str = 'cuda'
    ):
        """
        Args:
            model_path: путь к .pth файлу с моделью
            metadata_path: путь к metadata.json
            scaler_path: путь к scaler.pkl
            label_encoder_path: путь к label_encoder.pkl
            reference_dir: путь к директории с эталонами (.npy файлы)
                         Если None, автоматически найдет в reference_trajectories/
            device: устройство (cpu/cuda), по умолчанию 'cuda'
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        model_path = Path(model_path)
        
        # Загружаем модель
        checkpoint = torch.load(model_path, weights_only=False, map_location=self.device)
        
        # Проверяем, есть ли scaler и label_encoder в checkpoint
        if 'scaler' in checkpoint and 'label_encoder' in checkpoint:
            # Используем из checkpoint (предпочтительно, так как версии совпадают)
            self.scaler = checkpoint['scaler']
            self.label_encoder = checkpoint['label_encoder']
            self.metadata = checkpoint.get('metadata', {})
            print("Используются scaler и label_encoder из checkpoint модели")
        else:
            # Загружаем из отдельных файлов (fallback)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # Загружаем metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print("Используются scaler и label_encoder из отдельных файлов")
        
        self.sequence_length = self.metadata.get('sequence_length', 30)
        
        # Создаём модель
        input_size = self.scaler.mean_.shape[0]
        num_classes = len(self.label_encoder.classes_)
        state_dict = checkpoint['model_state_dict']
        is_hybrid = any('tcn' in key for key in state_dict.keys())
        
        if is_hybrid:
            self.model = HybridModel(input_size=input_size, num_classes=num_classes)
        else:
            self.model = GRUModel(input_size=input_size, num_classes=num_classes)
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Директории для метрик
        model_dir = model_path.parent
        self.videos_dir = model_dir.parent  # По умолчанию родительская папка модели
        
        # По умолчанию ищем reference_trajectories в dance_classifier/
        if reference_dir:
            self.reference_dir = Path(reference_dir)
        else:
            # Ищем в dance_classifier/reference_trajectories относительно текущего файла
            current_file = Path(__file__)
            # Текущий файл: dance_classifier/inference/predict.py
            # Нужно: dance_classifier/reference_trajectories
            dance_classifier_root = current_file.parent.parent  # поднимаемся на 2 уровня
            default_ref_dir = dance_classifier_root / "reference_trajectories"
            self.reference_dir = default_ref_dir
    
    def _interpolate_missing(self, sequence: np.ndarray) -> np.ndarray:
        """Заполняет пропущенные значения интерполяцией"""
        sequence = sequence.copy()
        for feature_idx in range(sequence.shape[1]):
            feature_values = sequence[:, feature_idx]
            valid_mask = ~np.isnan(feature_values)
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                sequence[:, feature_idx] = 0.0
                continue
            
            if len(valid_indices) < len(feature_values):
                invalid_indices = np.where(~valid_mask)[0]
                sequence[invalid_indices, feature_idx] = np.interp(
                    invalid_indices, valid_indices, feature_values[valid_indices]
                )
        return sequence
    
    def predict_sequence(self, sequence: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        Предсказание класса для одной последовательности
        
        Returns:
            (predicted_class, probabilities)
        """
        # Проверяем размерность перед нормализацией
        expected_features = self.scaler.mean_.shape[0]
        if sequence.shape[1] != expected_features:
            # Дополняем или обрезаем признаки
            if sequence.shape[1] < expected_features:
                padding = np.zeros((sequence.shape[0], expected_features - sequence.shape[1]))
                sequence = np.concatenate([sequence, padding], axis=1)
            elif sequence.shape[1] > expected_features:
                sequence = sequence[:, :expected_features]
        
        sequence_norm = self.scaler.transform(sequence)
        X = torch.FloatTensor(sequence_norm).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # Честное базовое предсказание по максимуму вероятности
        classes = self.label_encoder.classes_
        predicted_idx = int(np.argmax(probs))
        predicted_class = classes[predicted_idx]
        
        # Никаких дополнительных «подмен» NotPerforming на другие фигуры.
        # Фронтенд получает ровно то, что модель действительно предсказала.
        return predicted_class, probs
    
    def predict_from_poses(
        self,
        poses_json_path: str,
        video_path: Optional[str] = None,
        compute_metrics: bool = True,
        create_analyzed_video: bool = False,
        overlay_video_path: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Полное предсказание с метриками качества.
        
        Args:
            poses_json_path: путь к poses.jsonl
            video_path: путь к видео (для тайминг-метрики)
            compute_metrics: вычислять ли DTW-метрики
            create_analyzed_video: создавать ли analyzed_ видео с наложенными метриками
            overlay_video_path: путь к overlay видео (с позами). Если None, ищет автоматически
            output_dir: директория для сохранения analyzed видео. Если None, использует директорию overlay видео
        
        Returns:
            dict с результатами классификации и метриками. Если create_analyzed_video=True,
            также содержит 'analyzed_video_path' с путем к созданному видео.
        """
        # Загружаем позы и извлекаем признаки
        from ..data_preparation.feature_extraction import FeatureExtractor, get_simple_features
        
        # Загружаем данные в формате, который ожидает FeatureExtractor
        # YOLO формат: keypoints = [[x, y, confidence], ...] (17 точек)
        # FeatureExtractor ожидает: pose_landmarks = [{'x': ..., 'y': ..., 'z': ..., 'visibility': ...}, ...]
        poses_data = {"frames": []}
        width = None
        height = None
        
        with open(poses_json_path, 'r') as f:
            for line in f:
                pose = json.loads(line)
                if pose.get('valid', True):
                    # Сохраняем размеры кадра для нормализации
                    if width is None:
                        width = pose.get('width', 1.0)
                        height = pose.get('height', 1.0)
                    
                    # Преобразуем YOLO формат в формат для FeatureExtractor
                    # YOLO: keypoints = [[x, y, confidence], ...] (17 точек в порядке COCO)
                    # Нужно преобразовать в Mediapipe порядок (33 точки) или использовать только доступные
                    landmarks = []
                    yolo_keypoints = pose.get('keypoints', [])
                    
                    # YOLO COCO порядок: nose, left_eye, right_eye, left_ear, right_ear,
                    # left_shoulder, right_shoulder, left_elbow, right_elbow,
                    # left_wrist, right_wrist, left_hip, right_hip,
                    # left_knee, right_knee, left_ankle, right_ankle
                    # Преобразуем в нормализованные координаты [0, 1]
                    for kp in yolo_keypoints:
                        if len(kp) >= 3:
                            x_norm = kp[0] / width if width > 0 else kp[0]
                            y_norm = kp[1] / height if height > 0 else kp[1]
                            confidence = kp[2] if len(kp) > 2 else 1.0
                            landmarks.append({
                                'x': x_norm,
                                'y': y_norm,
                                'z': 0.0,  # YOLO не дает z координату
                                'visibility': confidence
                            })
                        else:
                            landmarks.append({
                                'x': 0.0,
                                'y': 0.0,
                                'z': 0.0,
                                'visibility': 0.0
                            })
                    
                    # Дополняем до 33 точек (Mediapipe формат) нулями, если нужно
                    while len(landmarks) < 33:
                        landmarks.append({
                            'x': 0.0,
                            'y': 0.0,
                            'z': 0.0,
                            'visibility': 0.0
                        })
                    
                    poses_data["frames"].append({
                        "pose_landmarks": landmarks
                    })
        
        if len(poses_data["frames"]) < self.sequence_length:
            return {
                'success': False,
                'predicted_class': 'Unknown',
                'predicted_figure': 'Unknown',  # Для совместимости с фронтендом
                'confidence': 0.0,
                'classification': {  # Старый формат для совместимости
                    'figure': 'Unknown',
                    'confidence': 0.0
                },
                'error': f'Not enough frames: {len(poses_data["frames"])} < {self.sequence_length}',
                'scores': {
                    'technique': {'score': 0.0, 'error': 'Not enough frames'},
                    'timing': {'score': 0.0, 'error': 'Not enough frames'},
                    'balance': {'score': 0.0, 'error': 'Not enough frames'},
                    'dynamics': {'score': 0.0, 'error': 'Not enough frames'},
                    'posture': {'score': 0.0, 'error': 'Not enough frames'}
                }
            }
        
        # Извлекаем признаки через FeatureExtractor
        try:
            # Проверяем структуру данных перед обработкой
            if not poses_data or 'frames' not in poses_data:
                raise ValueError(f"poses_data не содержит 'frames': {list(poses_data.keys()) if poses_data else 'пустой словарь'}")
            
            if len(poses_data['frames']) == 0:
                raise ValueError("poses_data['frames'] пуст")
            
            # Проверяем формат первого кадра
            first_frame = poses_data['frames'][0]
            if 'pose_landmarks' not in first_frame and 'pose_world_landmarks' not in first_frame:
                raise ValueError(f"Кадр не содержит landmarks: {list(first_frame.keys())}")
            
            feature_extractor = FeatureExtractor(min_confidence=0.3)
            feature_names = get_simple_features()
            
            print(f"3.1: Обработка {len(poses_data['frames'])} кадров...")
            feature_list, valid_mask = feature_extractor.extract_sequence_features(poses_data)
            print(f"3.2: Извлечено {len(feature_list)} признаков, валидных: {valid_mask.sum()}")
            
            feature_array, actual_feature_names, valid_indices = feature_extractor.features_to_array(
                feature_list, feature_names
            )
            print(f"3.3: Массив признаков: {feature_array.shape}, валидных индексов: {len(valid_indices)}")
            
            if len(valid_indices) < self.sequence_length:
                # Если валидных кадров недостаточно, используем все кадры
                if len(feature_array) >= self.sequence_length:
                    valid_indices = list(range(len(feature_array)))
                else:
                    return {
                        'success': False,
                        'predicted_class': 'Unknown',
                        'predicted_figure': 'Unknown',  # Для совместимости с фронтендом
                        'confidence': 0.0,
                        'classification': {  # Старый формат для совместимости
                            'figure': 'Unknown',
                            'confidence': 0.0
                        },
                        'error': f'Not enough frames: {len(feature_array)} < {self.sequence_length}',
                        'scores': {
                            'technique': {'score': 0.0, 'error': 'Not enough frames'},
                            'timing': {'score': 0.0, 'error': 'Not enough frames'},
                            'balance': {'score': 0.0, 'error': 'Not enough frames'},
                            'dynamics': {'score': 0.0, 'error': 'Not enough frames'},
                            'posture': {'score': 0.0, 'error': 'Not enough frames'}
                        }
                    }
            
            # Берём последнюю последовательность
            sequence = feature_array[-self.sequence_length:]
            
            # Дополнение до нужного количества признаков (если нужно)
            expected_features = self.scaler.mean_.shape[0]
            if sequence.shape[1] < expected_features:
                padding = np.zeros((sequence.shape[0], expected_features - sequence.shape[1]))
                sequence = np.concatenate([sequence, padding], axis=1)
            elif sequence.shape[1] > expected_features:
                sequence = sequence[:, :expected_features]
            
            sequence = self._interpolate_missing(sequence)
            
            # Предсказание
            predicted_class, probabilities = self.predict_sequence(sequence)
            
            result = {
                'success': True,
                'predicted_class': predicted_class,
                'predicted_figure': predicted_class,  # Для совместимости с фронтендом
                'confidence': float(probabilities[np.argmax(probabilities)]),
                'classification': {  # Старый формат для совместимости
                    'figure': predicted_class,
                    'confidence': float(probabilities[np.argmax(probabilities)])
                },
                'probabilities': {
                    self.label_encoder.classes_[i]: float(prob)
                    for i, prob in enumerate(probabilities)
                }
            }
            
            # === ИНТЕГРАЦИЯ DTW-МЕТРИК ===
            if compute_metrics:
                try:
                    # 1. Spatial Similarity (техника с DTW)
                    spatial_info = self._compute_spatial_similarity(
                        sequence, predicted_class, feature_array, self.sequence_length,
                        probabilities=result['probabilities']
                    )
                    if spatial_info:
                        result['spatial_similarity'] = spatial_info
                    
                    # 2. Classifier Clarity (уверенность модели по окнам)
                    clarity_info = self._compute_classifier_clarity(
                        sequence, predicted_class, feature_array, self.sequence_length
                    )
                    if clarity_info:
                        result['classifier_clarity'] = clarity_info
                    
                    # 3. Timing (только если есть видео)
                    if video_path:
                        timing_info = self._compute_timing(
                            poses_json_path, video_path, predicted_class
                        )
                        if timing_info:
                            result['timing'] = timing_info
                    
                    # 4. Balance (наклон корпуса, CoM)
                    balance_info = self._compute_balance(
                        poses_json_path, sequence.shape[0]
                    )
                    if balance_info:
                        result['balance'] = balance_info
                    
                    # Добавляем scores для обратной совместимости с фронтендом
                    scores = {}
                    
                    # Technique (Spatial Similarity)
                    if 'spatial_similarity' in result and isinstance(result['spatial_similarity'], dict):
                        if 'error' not in result['spatial_similarity']:
                            tech_score = result['spatial_similarity'].get('score', 0.0)
                            tech_score = float(tech_score) if tech_score is not None and not (isinstance(tech_score, float) and np.isnan(tech_score)) else 0.0
                            scores['technique'] = {'score': tech_score}
                            if 'note' in result['spatial_similarity']:
                                scores['technique']['note'] = result['spatial_similarity']['note']
                        else:
                            scores['technique'] = {'score': 0.0, 'error': result['spatial_similarity'].get('error')}
                    else:
                        scores['technique'] = {'score': 0.0}
                    
                    # Timing
                    if 'timing' in result and isinstance(result['timing'], dict):
                        if 'error' in result['timing']:
                            scores['timing'] = {'score': 0.0, 'error': result['timing'].get('error')}
                        else:
                            timing_score = result['timing'].get('score', 0.0)
                            timing_score = float(timing_score) if timing_score is not None and not (isinstance(timing_score, float) and np.isnan(timing_score)) else 0.0
                            scores['timing'] = {'score': timing_score}
                    else:
                        scores['timing'] = {'score': 0.0}
                    
                    # Balance
                    if 'balance' in result and isinstance(result['balance'], dict):
                        if 'error' in result['balance']:
                            scores['balance'] = {'score': 0.0, 'error': result['balance'].get('error')}
                        else:
                            balance_score = result['balance'].get('score', 0.0)
                            balance_score = float(balance_score) if balance_score is not None and not (isinstance(balance_score, float) and np.isnan(balance_score)) else 0.0
                            scores['balance'] = {'score': balance_score}
                    else:
                        scores['balance'] = {'score': 0.0}
                    
                    # Dynamics (Classifier Clarity)
                    if 'classifier_clarity' in result and isinstance(result['classifier_clarity'], dict):
                        if 'error' not in result['classifier_clarity']:
                            dynamics_score = result['classifier_clarity'].get('score', 0.0)
                            dynamics_score = float(dynamics_score) if dynamics_score is not None and not (isinstance(dynamics_score, float) and np.isnan(dynamics_score)) else 0.0
                            scores['dynamics'] = {'score': dynamics_score}
                        else:
                            scores['dynamics'] = {'score': 0.0, 'error': result['classifier_clarity'].get('error')}
                    else:
                        scores['dynamics'] = {'score': 0.0}
                    
                    # Posture (используем Balance)
                    posture_score = scores['balance'].get('score', 0.0)
                    posture_score = float(posture_score) if posture_score is not None and not (isinstance(posture_score, float) and np.isnan(posture_score)) else 0.0
                    scores['posture'] = {'score': posture_score}
                    
                    result['scores'] = scores
                
                except Exception as e:
                    # Даже при ошибке метрик, добавляем scores с ошибками
                    result['metrics_error'] = f"Metrics computation failed: {str(e)}"
                    # Убеждаемся, что scores всегда присутствует
                    if 'scores' not in result:
                        result['scores'] = {
                            'technique': {'score': 0.0, 'error': 'Metrics computation failed'},
                            'timing': {'score': 0.0, 'error': 'Metrics computation failed'},
                            'balance': {'score': 0.0, 'error': 'Metrics computation failed'},
                            'dynamics': {'score': 0.0, 'error': 'Metrics computation failed'},
                            'posture': {'score': 0.0, 'error': 'Metrics computation failed'}
                        }
            else:
                # Если метрики не вычисляются, все равно добавляем scores
                result['scores'] = {
                    'technique': {'score': 0.0, 'error': 'Metrics not computed'},
                    'timing': {'score': 0.0, 'error': 'Metrics not computed'},
                    'balance': {'score': 0.0, 'error': 'Metrics not computed'},
                    'dynamics': {'score': 0.0, 'error': 'Metrics not computed'},
                    'posture': {'score': 0.0, 'error': 'Metrics not computed'}
                }
            
            # Убеждаемся, что predicted_figure всегда присутствует
            if 'predicted_figure' not in result:
                result['predicted_figure'] = result.get('predicted_class', 'Unknown')
            
            # Убеждаемся, что classification всегда присутствует
            if 'classification' not in result:
                result['classification'] = {
                    'figure': result.get('predicted_class', 'Unknown'),
                    'confidence': result.get('confidence', 0.0)
                }
            
            # ФИНАЛЬНАЯ ПРОВЕРКА: гарантируем, что ВСЕ метрики имеют score
            if 'scores' in result:
                required_metrics = ['technique', 'timing', 'balance', 'dynamics', 'posture']
                for metric in required_metrics:
                    if metric not in result['scores']:
                        result['scores'][metric] = {'score': 0.0}
                    elif not isinstance(result['scores'][metric], dict):
                        result['scores'][metric] = {'score': float(result['scores'][metric]) if isinstance(result['scores'][metric], (int, float)) else 0.0}
                    elif 'score' not in result['scores'][metric]:
                        result['scores'][metric]['score'] = 0.0
                    else:
                        # Убеждаемся, что score - это число, а не None
                        score_value = result['scores'][metric]['score']
                        if score_value is None or (isinstance(score_value, float) and np.isnan(score_value)):
                            result['scores'][metric]['score'] = 0.0
                        else:
                            result['scores'][metric]['score'] = float(score_value)
            else:
                # Если scores вообще нет, создаем с нулями
                result['scores'] = {
                    'technique': {'score': 0.0},
                    'timing': {'score': 0.0},
                    'balance': {'score': 0.0},
                    'dynamics': {'score': 0.0},
                    'posture': {'score': 0.0}
                }
            
            # Создаем analyzed_ видео, если запрошено
            if create_analyzed_video and result.get('success', False):
                analyzed_video_path = self._create_analyzed_video(
                    result=result,
                    overlay_video_path=overlay_video_path,
                    poses_json_path=poses_json_path,
                    output_dir=output_dir
                )
                if analyzed_video_path:
                    result['analyzed_video_path'] = str(analyzed_video_path)
            
            return result
        except Exception as e:
            import traceback
            # Улучшаем сообщение об ошибке для фронтенда
            error_str = str(e)
            if "features" in error_str and "StandardScaler" in error_str:
                # Ошибка размерности признаков
                user_friendly_error = "Несоответствие размерности признаков. Проверьте версию модели и данных."
            elif "expected an indented block" in error_str:
                # Синтаксическая ошибка
                user_friendly_error = "Внутренняя ошибка обработки. Обратитесь к разработчику."
            else:
                user_friendly_error = f"Ошибка обработки: {error_str}"
            
            print(f"ERROR: {error_str}")
            return {
                'success': False,
                'predicted_class': 'Unknown',
                'predicted_figure': 'Unknown',  # Для совместимости с фронтендом
                'confidence': 0.0,
                'classification': {  # Старый формат для совместимости
                    'figure': 'Unknown',
                    'confidence': 0.0
                },
                'error': user_friendly_error,
                'scores': {
                    'technique': {'score': 0.0, 'error': 'Feature extraction failed'},
                    'timing': {'score': 0.0, 'error': 'Feature extraction failed'},
                    'balance': {'score': 0.0, 'error': 'Feature extraction failed'},
                    'dynamics': {'score': 0.0, 'error': 'Feature extraction failed'},
                    'posture': {'score': 0.0, 'error': 'Feature extraction failed'}
                }
            }
    
    def _compute_spatial_similarity(
        self,
        sequence: np.ndarray,
        predicted_class: str,
        all_poses: np.ndarray,
        seq_len: int,
        probabilities: Optional[Dict[str, float]] = None
    ) -> Optional[Dict]:
        """Вычисляет DTW-сходство с эталоном"""
        try:
            from ..utils.spatial_similarity import compute_spatial_similarity
            
            # Проверяем наличие эталона для предсказанного класса
            ref_path = self.reference_dir / f"{predicted_class}.npy"
            
            # Если нет эталона для предсказанного класса, возвращаем нейтральное значение
            # НЕ используем альтернативные эталоны - они не подходят для других фигур
            if not ref_path.exists():
                return {
                    "score": 50.0,
                    "mean_distance": None,
                    "error_segments": [],
                    "note": f"No reference trajectory for {predicted_class}, using neutral score"
                }
            
            reference = np.load(ref_path)
            sequence_norm = self.scaler.transform(sequence)
            
            # Индексы кадров
            start_idx = len(all_poses) - seq_len
            frame_indices = np.arange(start_idx, start_idx + seq_len)
            frame_timestamps = frame_indices / 25.0  # предполагаем 25 FPS
            
            result = compute_spatial_similarity(
                sequence_norm,
                reference,
                frame_indices=frame_indices,
                frame_timestamps=frame_timestamps
            )
            
            return {
                "score": float(result.score) if not np.isnan(result.score) else 0.0,
                "mean_distance": float(result.mean_distance) if not np.isnan(result.mean_distance) else 0.0,
                "reference_figure": predicted_class,
                "error_segments": result.error_segments if hasattr(result, 'error_segments') else []
            }
        except Exception as e:
            return {
                "score": 0.0,
                "error": str(e),
                "mean_distance": None,
                "error_segments": []
            }
    
    def _compute_classifier_clarity(
        self,
        sequence: np.ndarray,
        predicted_class: str,
        all_poses: np.ndarray,
        seq_len: int
    ) -> Optional[Dict]:
        """Вычисляет уверенность модели по скользящим окнам"""
        try:
            from ..utils.classifier_clarity import compute_classifier_clarity
            
            sequence_norm = self.scaler.transform(sequence)
            
            # Скользящие окна
            window_size = max(8, seq_len // 4)
            window_stride = max(1, window_size // 2)
            
            window_probs = []
            window_centers = []
            
            for start in range(0, seq_len - window_size + 1, window_stride):
                end = start + window_size
                sub_seq = sequence_norm[start:end]
                x_win = torch.FloatTensor(sub_seq).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out_win = self.model(x_win)
                    p_win = torch.softmax(out_win, dim=1).cpu().numpy()[0]
                window_probs.append(p_win)
                window_centers.append(start + window_size // 2)
            
            if not window_probs:
                return None
            
            probs_over_time = np.stack(window_probs, axis=0)
            window_centers = np.array(window_centers, dtype=int)
            
            # Индекс целевого класса
            target_idx = np.where(self.label_encoder.classes_ == predicted_class)[0]
            if len(target_idx) == 0:
                return None
            target_idx = int(target_idx[0])
            
            # Индексы кадров
            start_idx = len(all_poses) - seq_len
            frame_indices = np.arange(start_idx, start_idx + seq_len)
            frame_timestamps = frame_indices / 25.0
            
            clarity_res = compute_classifier_clarity(
                probs_over_time,
                target_idx,
                window_centers=window_centers,
                seq_frame_indices=frame_indices,
                seq_timestamps=frame_timestamps
            )
            
            return {
                "score": float(clarity_res.score),
                "mean_confidence": float(clarity_res.mean_confidence),
                "mean_margin": float(clarity_res.mean_margin),
                "mean_entropy": float(clarity_res.mean_entropy),
                "error_segments": clarity_res.error_segments
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _compute_timing(
        self,
        poses_json_path: str,
        video_path: str,
        predicted_class: str
    ) -> Optional[Dict]:
        """Вычисляет метрику тайминга (шаги vs музыка)"""
        try:
            from ..utils.timing_metric import compute_timing_metric
            
            # Загружаем полные данные поз
            with open(poses_json_path, 'r') as f:
                poses_data = {"frames": []}
                for line in f:
                    pose = json.loads(line)
                    keypoints = pose.get('keypoints', [])
                    if keypoints:
                        poses_data["frames"].append({
                            "pose_landmarks": [keypoints[i] for i in range(len(keypoints)) if i < len(keypoints)]
                        })
                poses_data["fps"] = 25.0  # предполагаем
            
            video_path = Path(video_path)
            # Если путь не абсолютный, делаем его абсолютным
            if not video_path.is_absolute():
                video_path = video_path.resolve()
            
            if not video_path.exists():
                return {"error": f"Video file not found: {video_path}"}
            
            timing_res = compute_timing_metric(
                poses_data=poses_data,
                audio_path=video_path
            )
            
            return {
                "bpm": float(timing_res.bpm),
                "mean_abs_offset": float(timing_res.mean_abs_offset) if not np.isnan(timing_res.mean_abs_offset) else None,
                "onbeat_ratio": float(timing_res.onbeat_ratio),
                "score": float(timing_res.score),
                "error_segments": timing_res.error_segments
            }
        except Exception as e:
            error_msg = str(e) if str(e) else f"Unknown error: {type(e).__name__}"
            return {"error": error_msg}
    
    def _compute_balance(
        self,
        poses_json_path: str,
        seq_len: int
    ) -> Optional[Dict]:
        """Вычисляет метрику баланса"""
        try:
            from ..utils.balance_metric import compute_balance_metric
            
            # Загружаем полные данные поз
            with open(poses_json_path, 'r') as f:
                poses_data = {"frames": []}
                for line in f:
                    pose = json.loads(line)
                    keypoints = pose.get('keypoints', [])
                    if keypoints and len(keypoints) >= 17:
                        # Преобразуем YOLO keypoints (17 точек) в формат для balance_metric (29 точек Mediapipe)
                        # YOLO COCO порядок: 0=nose, 1-4=eyes/ears, 5-6=shoulders, 7-8=elbows,
                        # 9-10=wrists, 11-12=hips, 13-14=knees, 15-16=ankles
                        # Создаем массив из 29 элементов, заполняя нужные позиции для balance_metric
                        full_landmarks = [None] * 29
                        
                        # Маппинг YOLO -> Mediapipe для точек, нужных balance_metric:
                        # Mediapipe: 11=left_shoulder, 12=right_shoulder, 23=left_hip, 24=right_hip, 27=left_ankle, 28=right_ankle
                        # YOLO: 5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip, 15=left_ankle, 16=right_ankle
                        mapping = {
                            5: 11,   # left_shoulder
                            6: 12,   # right_shoulder
                            11: 23,  # left_hip
                            12: 24,  # right_hip
                            15: 27,  # left_ankle
                            16: 28   # right_ankle
                        }
                        
                        for yolo_idx, mediapipe_idx in mapping.items():
                            if yolo_idx < len(keypoints):
                                kp = keypoints[yolo_idx]
                                if len(kp) >= 2:
                                    full_landmarks[mediapipe_idx] = {
                                        'x': float(kp[0]),
                                        'y': float(kp[1]),
                                        'z': 0.0
                                    }
                        
                        # Заполняем остальные позиции нулевыми значениями (balance_metric использует только указанные выше точки)
                        for i in range(29):
                            if full_landmarks[i] is None:
                                full_landmarks[i] = {'x': 0.0, 'y': 0.0, 'z': 0.0}
                        
                        poses_data["frames"].append({
                            "pose_landmarks": full_landmarks
                        })
                poses_data["fps"] = 25.0
            
            # Индексы кадров последовательности
            total_frames = len(poses_data["frames"])
            start_idx = total_frames - seq_len
            seq_frame_indices = np.arange(start_idx, start_idx + seq_len)
            
            balance_res = compute_balance_metric(
                poses_data=poses_data,
                seq_frame_indices=seq_frame_indices
            )
            
            return {
                "score": float(balance_res.score),
                "mean_tilt_deg": float(balance_res.mean_tilt_deg) if not np.isnan(balance_res.mean_tilt_deg) else None,
                "tilt_std_deg": float(balance_res.tilt_std_deg) if not np.isnan(balance_res.tilt_std_deg) else None,
                "com_std_norm": float(balance_res.com_std_norm) if not np.isnan(balance_res.com_std_norm) else None,
                "out_of_corridor_ratio": float(balance_res.out_of_corridor_ratio),
                "error_segments": balance_res.error_segments
            }
        except Exception as e:
            error_msg = str(e) if str(e) else f"Unknown error: {type(e).__name__}"
            return {"error": error_msg}
    
    def _put_text_ru(self, frame, text, position, font_scale=1.0, color=(255, 255, 255), thickness=2):
        """
        Рисует русский текст на кадре OpenCV через PIL.
        
        Args:
            frame: numpy array (BGR)
            text: текст для отображения
            position: (x, y) координаты
            font_scale: масштаб шрифта
            color: цвет (B, G, R)
            thickness: толщина шрифта (не используется в PIL, но оставлен для совместимости)
        """
        try:
            # Конвертируем BGR в RGB для PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_image)
            
            # Пробуем загрузить системный шрифт с поддержкой кириллицы
            try:
                # Windows шрифты
                font_paths = [
                    "C:/Windows/Fonts/arial.ttf",
                    "C:/Windows/Fonts/calibri.ttf",
                    "C:/Windows/Fonts/tahoma.ttf",
                ]
                font = None
                for fp in font_paths:
                    if Path(fp).exists():
                        # Размер шрифта зависит от scale
                        font_size = int(20 * font_scale)
                        font = ImageFont.truetype(fp, font_size)
                        break
                
                if font is None:
                    # Fallback на стандартный шрифт
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            # Конвертируем цвет из BGR в RGB
            color_rgb = (color[2], color[1], color[0])
            
            # Рисуем текст
            draw.text(position, text, fill=color_rgb, font=font)
            
            # Конвертируем обратно в BGR
            frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return frame_bgr
        except Exception as e:
            # Если что-то пошло не так, используем стандартный cv2.putText (будет ??? для русских)
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            return frame
    
    def _create_analyzed_video(
        self,
        result: Dict,
        overlay_video_path: Optional[str] = None,
        poses_json_path: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Optional[Path]:
        """
        Создает analyzed_ видео с наложенными метриками.
        
        Args:
            result: результат анализа из predict_from_poses
            overlay_video_path: путь к overlay видео (с позами)
            poses_json_path: путь к poses.jsonl (для поиска overlay видео)
            output_dir: директория для сохранения analyzed видео
        
        Returns:
            Path к созданному видео или None при ошибке
        """
        try:
            # Определяем overlay_video_path
            if not overlay_video_path and poses_json_path:
                # Пытаемся найти overlay видео в той же директории, что и poses.jsonl
                poses_path = Path(poses_json_path)
                poses_dir = poses_path.parent
                
                # Ищем overlay_*.mp4 в директории poses.jsonl
                overlay_candidates = list(poses_dir.glob("overlay_*.mp4"))
                if overlay_candidates:
                    overlay_video_path = str(overlay_candidates[0])
                else:
                    print(f"[WARN] Overlay video not found in {poses_dir}")
                    return None
            
            if not overlay_video_path:
                print("[WARN] overlay_video_path not provided and cannot be found")
                return None
            
            overlay_path = Path(overlay_video_path)
            if not overlay_path.exists():
                print(f"[WARN] Overlay video not found: {overlay_path}")
                return None
            
            # Определяем output_dir
            if not output_dir:
                output_dir = overlay_path.parent
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Имя выходного файла
            video_name = overlay_path.stem.replace("overlay_", "")
            analyzed_video_path = output_dir / f"analyzed_{video_name}.mp4"
            
            print(f"\n[INFO] Creating analyzed video: {analyzed_video_path.name}")
            
            # Извлекаем данные из result
            figure = result.get('predicted_figure') or result.get('predicted_class', 'Unknown')
            confidence = result.get('confidence', 0.0) * 100
            
            # Извлекаем scores
            scores = result.get('scores', {})
            
            # Загружаем данные об ошибках для отображения на кадрах
            error_segments = {}
            frame_errors = {}  # frame_idx -> список ошибок
            
            # Balance errors
            if 'balance' in result and isinstance(result['balance'], dict):
                balance_data = result['balance']
                if 'error_segments' in balance_data:
                    for seg in balance_data['error_segments']:
                        start_frame = int(seg.get('start_idx', 0))
                        end_frame = int(seg.get('end_idx', start_frame))
                        tilt = seg.get('mean_tilt_deg', 0)
                        for f in range(start_frame, end_frame + 1):
                            if f not in frame_errors:
                                frame_errors[f] = []
                            frame_errors[f].append({
                                'type': 'balance',
                                'tilt_deg': tilt,
                                'message': f'Tilt: {tilt:.1f}°'
                            })
            
            # Timing errors
            if 'timing' in result and isinstance(result['timing'], dict):
                timing_data = result['timing']
                if 'error_segments' in timing_data:
                    for seg in timing_data['error_segments']:
                        start_time = seg.get('start_time', 0)
                        end_time = seg.get('end_time', start_time)
                        offset = seg.get('mean_abs_offset', 0)
                        # Конвертируем время в кадры (предполагаем 25 FPS)
                        start_frame = int(start_time * fps)
                        end_frame = int(end_time * fps)
                        for f in range(start_frame, end_frame + 1):
                            if f not in frame_errors:
                                frame_errors[f] = []
                            frame_errors[f].append({
                                'type': 'timing',
                                'offset': offset,
                                'message': f'Timing: {offset*1000:.0f}ms off'
                            })
            
            # Spatial similarity errors
            if 'spatial_similarity' in result and isinstance(result['spatial_similarity'], dict):
                spatial_data = result['spatial_similarity']
                if 'error_segments' in spatial_data:
                    for seg in spatial_data['error_segments']:
                        start_time = seg.get('start_time', 0)
                        end_time = seg.get('end_time', start_time)
                        distance = seg.get('mean_distance', 0)
                        start_frame = int(start_time * fps)
                        end_frame = int(end_time * fps)
                        for f in range(start_frame, end_frame + 1):
                            if f not in frame_errors:
                                frame_errors[f] = []
                            frame_errors[f].append({
                                'type': 'technique',
                                'distance': distance,
                                'message': f'Deviation: {distance:.1f}'
                            })
            
            # Загружаем poses.jsonl для получения данных о позах (если нужно)
            poses_data_by_frame = {}
            if poses_json_path and Path(poses_json_path).exists():
                try:
                    with open(poses_json_path, 'r') as f:
                        for line_idx, line in enumerate(f):
                            pose = json.loads(line)
                            if pose.get('valid', True):
                                poses_data_by_frame[line_idx] = pose
                except:
                    pass  # Если не удалось загрузить, продолжаем без данных о позах
            
            # Открываем overlay видео
            cap = cv2.VideoCapture(str(overlay_path))
            if not cap.isOpened():
                print(f"[ERROR] Cannot open overlay video: {overlay_path}")
                return None
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Создаем выходное видео
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(analyzed_video_path), fourcc, fps, (width, height))
            
            if not out.isOpened():
                print("[ERROR] Cannot create video writer")
                cap.release()
                return None
            
            # Метрики для отображения
            metrics_list = [
                ("Technique", scores.get('technique', {}).get('score', 0.0)),
                ("Timing", scores.get('timing', {}).get('score', 0.0)),
                ("Balance", scores.get('balance', {}).get('score', 0.0)),
                ("Dynamics", scores.get('dynamics', {}).get('score', 0.0)),
                ("Posture", scores.get('posture', {}).get('score', 0.0))
            ]
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                h, w = frame.shape[:2]
                
                # ===== ВЕРХНЯЯ ПАНЕЛЬ (НИЖЕ, ЧТОБЫ НЕ ПРИЖИМАТЬСЯ К ВЕРХУ) =====
                overlay = frame.copy()
                top_panel_h = 120
                cv2.rectangle(overlay, (0, 40), (w, 40 + top_panel_h), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
                
                # FIGURE (опускаем немного вниз)
                cv2.putText(
                    frame,
                    f"FIGURE: {figure}",
                    (40, 90),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.2,
                    (0, 255, 255),
                    3,
                )
                
                # CONFIDENCE (чуть ниже FIGURE)
                conf_color = (
                    (0, 255, 0)
                    if confidence >= 70
                    else (0, 165, 255)
                    if confidence >= 50
                    else (0, 0, 255)
                )
                cv2.putText(
                    frame,
                    f"CONFIDENCE: {confidence:.1f}%",
                    (40, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    conf_color,
                    2,
                )
                
                # ===== LEFT PANEL - QUALITY METRICS (ENGLISH) =====
                # Compact block on the left, ещё ниже, чтобы не пересекаться с FIGURE/CONFIDENCE
                panel_w = 320
                panel_h = 220
                panel_x = 20
                panel_y = 180
                
                overlay2 = frame.copy()
                cv2.rectangle(overlay2, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 30, 30), -1)
                frame = cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0)
                
                # Header
                cv2.putText(
                    frame,
                    "QUALITY METRICS",
                    (panel_x + 15, panel_y + 35),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )
                
                # Линия
                cv2.line(frame, (panel_x + 15, panel_y + 55), (panel_x + panel_w - 15, panel_y + 55), (200, 200, 200), 2)
                
                # Metrics
                y_pos = panel_y + 80
                for name, score in metrics_list:
                    # Color and text hint (EN)
                    if score >= 70:
                        color = (0, 255, 0)  # Зеленый
                        hint = "Good"
                    elif score >= 50:
                        color = (0, 165, 255)  # Оранжевый
                        hint = "Needs work"
                    else:
                        color = (0, 0, 255)  # Красный
                        hint = "Critical"
                    
                    display_name = name  # Use original English names
                    
                    # Metric name
                    cv2.putText(
                        frame,
                        display_name,
                        (panel_x + 15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (220, 220, 220),
                        2,
                    )
                    
                    # Прогресс-бар
                    bar_x = panel_x + 15
                    bar_y = y_pos + 12
                    bar_w = 170
                    bar_h = 28
                    
                    # Фон
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 2)
                    
                    # Заполнение
                    fill_w = int(bar_w * (score / 100))
                    if fill_w > 0:
                        cv2.rectangle(frame, (bar_x + 2, bar_y + 2), (bar_x + fill_w - 2, bar_y + bar_h - 2), color, -1)
                    
                    # Numeric value (large)
                    score_text = f"{score:.1f}"
                    text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)[0]
                    text_x = bar_x + bar_w + 10
                    text_y = bar_y + bar_h // 2 + text_size[1] // 2
                    max_x = panel_x + panel_w - 10
                    if text_x + text_size[0] > max_x:
                        text_x = max_x - text_size[0] - 5
                    cv2.putText(frame, score_text, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
                    
                    # Подсказка под значением (маленькая строка под блоком)
                    hint_x = bar_x
                    hint_y = bar_y + bar_h + 18
                    frame = self._put_text_ru(frame, hint, (hint_x, hint_y),
                                             font_scale=0.6, color=color, thickness=1)
                    
                    y_pos += 55
                
                # ===== НИЖНЯЯ ПАНЕЛЬ - ПРОГРЕСС =====
                overlay3 = frame.copy()
                cv2.rectangle(overlay3, (0, h-60), (w, h), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay3, 0.7, frame, 0.3, 0)
                
                progress = frame_idx / total_frames if total_frames > 0 else 0
                bar_w = w - 40
                bar_h = 20
                fill_w = int(bar_w * progress)
                
                cv2.rectangle(frame, (20, h-40), (20 + bar_w, h-20), (60, 60, 60), -1)
                cv2.rectangle(frame, (20, h-40), (20 + bar_w, h-20), (100, 100, 100), 2)
                cv2.rectangle(frame, (20, h-40), (20 + fill_w, h-20), (0, 255, 0), -1)
                
                # Время
                time_text = f"{frame_idx}/{total_frames} frames"
                cv2.putText(frame, time_text, (20, h-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                
                # ===== ОТОБРАЖЕНИЕ ОШИБОК НА КАДРЕ =====
                if frame_idx in frame_errors:
                    errors = frame_errors[frame_idx]
                    # Показываем ошибки компактно внизу слева, чтобы не мешать танцору
                    error_y = h - 90
                    
                    for error in errors:
                        # Цвет в зависимости от типа ошибки
                        if error['type'] == 'balance':
                            error_color = (0, 0, 255)  # Красный для баланса
                            error_label = "БАЛАНС"
                        elif error['type'] == 'timing':
                            error_color = (0, 165, 255)  # Оранжевый для тайминга
                            error_label = "ТАЙМИНГ"
                        else:
                            error_color = (255, 0, 255)  # Пурпурный для техники
                            error_label = "ТЕХНИКА"
                        
                        # Фон для текста ошибки
                        error_text = f"{error_label}: {error['message']}"
                        text_size = cv2.getTextSize(error_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                        text_x = 20
                        text_y = error_y
                        
                        # Полупрозрачный фон
                        overlay_error = frame.copy()
                        cv2.rectangle(overlay_error, 
                                    (text_x - 5, text_y - text_size[1] - 5),
                                    (text_x + text_size[0] + 5, text_y + 5),
                                    (0, 0, 0), -1)
                        frame = cv2.addWeighted(overlay_error, 0.7, frame, 0.3, 0)
                        
                        # Текст ошибки (русский текст)
                        frame = self._put_text_ru(frame, error_text, (text_x, text_y),
                                                  font_scale=1.0, color=error_color, thickness=2)
                        
                        error_y += 35
                
                # ===== ВИЗУАЛИЗАЦИЯ ОШИБОК НА СКЕЛЕТЕ ТАНЦОРА =====
                if frame_idx in poses_data_by_frame:
                    pose = poses_data_by_frame[frame_idx]
                    keypoints = pose.get('keypoints', [])
                    frame_width = pose.get('width', w)
                    frame_height = pose.get('height', h)
                    
                    if len(keypoints) >= 17:
                        # YOLO COCO порядок: 0=nose, 1-4=eyes/ears, 5-6=shoulders, 7-8=elbows,
                        # 9-10=wrists, 11-12=hips, 13-14=knees, 15-16=ankles
                        try:
                            # Получаем координаты ключевых точек
                            left_shoulder = keypoints[5] if len(keypoints) > 5 and len(keypoints[5]) >= 2 else None
                            right_shoulder = keypoints[6] if len(keypoints) > 6 and len(keypoints[6]) >= 2 else None
                            left_hip = keypoints[11] if len(keypoints) > 11 and len(keypoints[11]) >= 2 else None
                            right_hip = keypoints[12] if len(keypoints) > 12 and len(keypoints[12]) >= 2 else None
                            left_ankle = keypoints[15] if len(keypoints) > 15 and len(keypoints[15]) >= 2 else None
                            right_ankle = keypoints[16] if len(keypoints) > 16 and len(keypoints[16]) >= 2 else None
                            
                            # Конвертируем координаты в пиксели кадра
                            def to_pixel(kp, frame_w, frame_h, video_w, video_h):
                                if kp is None or len(kp) < 2:
                                    return None
                                # Если координаты нормализованные (0-1), конвертируем
                                if kp[0] <= 1.0 and kp[1] <= 1.0:
                                    x = int(kp[0] * video_w)
                                    y = int(kp[1] * video_h)
                                else:
                                    # Если уже в пикселях (как в poses.jsonl), масштабируем
                                    # poses.jsonl содержит координаты в пикселях исходного видео
                                    scale_x = video_w / frame_w if frame_w > 0 else 1.0
                                    scale_y = video_h / frame_h if frame_h > 0 else 1.0
                                    x = int(kp[0] * scale_x)
                                    y = int(kp[1] * scale_y)
                                return (x, y)
                            
                            # Проверяем ошибки баланса
                            has_balance_error = frame_idx in frame_errors and any(e['type'] == 'balance' for e in frame_errors[frame_idx])
                            
                            if left_shoulder and right_shoulder and left_hip and right_hip:
                                # Конвертируем координаты
                                ls = to_pixel(left_shoulder, frame_width, frame_height, w, h)
                                rs = to_pixel(right_shoulder, frame_width, frame_height, w, h)
                                lh = to_pixel(left_hip, frame_width, frame_height, w, h)
                                rh = to_pixel(right_hip, frame_width, frame_height, w, h)
                                
                                if ls and rs and lh and rh:
                                    # Проверяем, что координаты в пределах кадра
                                    if (0 <= ls[0] < w and 0 <= ls[1] < h and
                                        0 <= rs[0] < w and 0 <= rs[1] < h and
                                        0 <= lh[0] < w and 0 <= lh[1] < h and
                                        0 <= rh[0] < w and 0 <= rh[1] < h):
                                        
                                        # Вычисляем центр плеч и бедер
                                        shoulder_center = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
                                        hip_center = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)
                                        
                                        # Вычисляем угол наклона корпуса
                                        dx = shoulder_center[0] - hip_center[0]
                                        dy = abs(shoulder_center[1] - hip_center[1])
                                        if dy > 0:
                                            tilt_deg = np.arctan(abs(dx) / dy) * 180 / np.pi
                                            
                                            # Рисуем линию наклона корпуса (всегда, если наклон > 3 градуса для видимости)
                                            if has_balance_error or tilt_deg > 3:
                                                # Цвет линии зависит от степени наклона
                                                if tilt_deg > 15:
                                                    line_color = (0, 0, 255)  # Красный - критический наклон
                                                    line_thickness = 4
                                                elif tilt_deg > 10:
                                                    line_color = (0, 100, 255)  # Оранжево-красный
                                                    line_thickness = 3
                                                else:
                                                    line_color = (0, 165, 255)  # Оранжевый
                                                    line_thickness = 2
                                                
                                                # Рисуем линию от бедер к плечам (показывает наклон)
                                                cv2.line(frame, hip_center, shoulder_center, line_color, line_thickness)
                                                
                                                # Рисуем перпендикулярную линию (показывает отклонение от вертикали)
                                                vertical_x = hip_center[0]
                                                vertical_y = hip_center[1] - 50  # Вертикальная линия вверх
                                                cv2.line(frame, hip_center, (vertical_x, vertical_y), (100, 100, 100), 1)
                                                
                                                # Текст с углом наклона и подсказкой рядом с линией
                                                if tilt_deg > 15:
                                                    tilt_text = f"{tilt_deg:.1f}° (CRITICAL!)"
                                                    hint_text = "Fix posture!"
                                                elif tilt_deg > 10:
                                                    tilt_text = f"{tilt_deg:.1f}° (HIGH)"
                                                    hint_text = "Watch balance"
                                                else:
                                                    tilt_text = f"{tilt_deg:.1f}°"
                                                    hint_text = "OK"
                                                
                                                text_x = shoulder_center[0] + 10
                                                text_y = shoulder_center[1] - 10
                                                
                                                # Фон для текста
                                                text_bg = frame.copy()
                                                text_size = cv2.getTextSize(tilt_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                                                cv2.rectangle(text_bg,
                                                             (text_x - 5, text_y - text_size[1] - 5),
                                                             (text_x + text_size[0] + 5, text_y + 5),
                                                             (0, 0, 0), -1)
                                                frame = cv2.addWeighted(text_bg, 0.7, frame, 0.3, 0)
                                                
                                                cv2.putText(frame, tilt_text, (text_x, text_y),
                                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_color, 2)
                                                
                                                # Подсказка ниже
                                                if tilt_deg > 10:
                                                    hint_y = text_y + 25
                                                    hint_size = cv2.getTextSize(hint_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                                                    hint_bg = frame.copy()
                                                    cv2.rectangle(hint_bg,
                                                                 (text_x - 5, hint_y - hint_size[1] - 5),
                                                                 (text_x + hint_size[0] + 5, hint_y + 5),
                                                                 (0, 0, 0), -1)
                                                    frame = cv2.addWeighted(hint_bg, 0.7, frame, 0.3, 0)
                                                    cv2.putText(frame, hint_text, (text_x, hint_y),
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 1)
                                                
                                                # Выделяем проблемные суставы (плечи и бедра)
                                                cv2.circle(frame, ls, 8, line_color, -1)
                                                cv2.circle(frame, rs, 8, line_color, -1)
                                                cv2.circle(frame, lh, 8, line_color, -1)
                                                cv2.circle(frame, rh, 8, line_color, -1)
                                                
                                                # Обводим проблемные суставы
                                                cv2.circle(frame, ls, 10, line_color, 2)
                                                cv2.circle(frame, rs, 10, line_color, 2)
                                                cv2.circle(frame, lh, 10, line_color, 2)
                                                cv2.circle(frame, rh, 10, line_color, 2)
                            
                            # Проверяем ошибки тайминга (выделяем ноги)
                            has_timing_error = frame_idx in frame_errors and any(e['type'] == 'timing' for e in frame_errors[frame_idx])
                            
                            if has_timing_error and left_ankle and right_ankle:
                                la = to_pixel(left_ankle, frame_width, frame_height, w, h)
                                ra = to_pixel(right_ankle, frame_width, frame_height, w, h)
                                
                                if la and ra:
                                    # Выделяем лодыжки при ошибках тайминга
                                    timing_color = (0, 165, 255)  # Оранжевый
                                    cv2.circle(frame, la, 12, timing_color, -1)
                                    cv2.circle(frame, ra, 12, timing_color, -1)
                                    cv2.circle(frame, la, 15, timing_color, 3)
                                    cv2.circle(frame, ra, 15, timing_color, 3)
                                    
                                    # Рисуем стрелку вверх/вниз (показывает, что шаг не в такт)
                                    arrow_length = 30
                                    cv2.arrowedLine(frame, 
                                                   (la[0], la[1] - arrow_length),
                                                   la,
                                                   timing_color, 3, tipLength=0.3)
                                    cv2.arrowedLine(frame,
                                                   (ra[0], ra[1] - arrow_length),
                                                   ra,
                                                   timing_color, 3, tipLength=0.3)
                            
                            # Проверяем ошибки техники (выделяем все тело)
                            has_technique_error = frame_idx in frame_errors and any(e['type'] == 'technique' for e in frame_errors[frame_idx])
                            
                            if has_technique_error:
                                # Выделяем все видимые keypoints при ошибках техники
                                technique_color = (255, 0, 255)  # Пурпурный
                                for i, kp in enumerate(keypoints):
                                    if len(kp) >= 2:
                                        kp_pixel = to_pixel(kp, frame_width, frame_height, w, h)
                                        if kp_pixel:
                                            # Увеличиваем размер точек при ошибках
                                            cv2.circle(frame, kp_pixel, 6, technique_color, -1)
                                            cv2.circle(frame, kp_pixel, 8, technique_color, 2)
                        except Exception as e:
                            pass  # Если не удалось визуализировать, пропускаем
                
                out.write(frame)
                frame_idx += 1
                
                if frame_idx % 30 == 0:
                    percent = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                    print(f"  Progress: {percent:.1f}% ({frame_idx}/{total_frames})", end='\r')
            
            cap.release()
            out.release()
            
            print(f"\n[OK] Analyzed video created: {analyzed_video_path}")
            print(f"  Size: {analyzed_video_path.stat().st_size / (1024*1024):.2f} MB")
            
            return analyzed_video_path
            
        except Exception as e:
            import traceback
            print(f"[ERROR] Failed to create analyzed video: {e}\n{traceback.format_exc()}")
            return None


def main():
    """Пример использования"""
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Использование: python predict.py путь/к/poses.jsonl [путь/к/видео.mp4]")
        sys.exit(1)
    
    poses_path = sys.argv[1]
    video_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Определяем пути относительно текущего файла
    current_file = Path(__file__)
    dance_classifier_root = current_file.parent.parent
    # Модель может быть в основной папке coach-assistant
    main_coach_dir = Path(r"C:\Users\1\!PYTHON_DZ\coach-assistant\dance_classifier")
    # Приоритет: сначала finetuned модель, потом обычная
    if (main_coach_dir / "best_model_20pct_finetuned.pth").exists():
        model_path = main_coach_dir / "best_model_20pct_finetuned.pth"
        print("Используется дообученная модель: best_model_20pct_finetuned.pth")
    elif (main_coach_dir / "best_model_20pct.pth").exists():
        model_path = main_coach_dir / "best_model_20pct.pth"
    elif (main_coach_dir / "best_model_20pct_adapted.pth").exists():
        # Fallback на adapted версию, если основной нет
        model_path = main_coach_dir / "best_model_20pct_adapted.pth"
    else:
        model_path = dance_classifier_root / "best_model_20pct.pth"
    metadata_path = dance_classifier_root / "dataset" / "metadata.json"
    scaler_path = dance_classifier_root / "dataset" / "scaler.pkl"
    label_encoder_path = dance_classifier_root / "dataset" / "label_encoder.pkl"
    
    predictor = DanceClassifierPredictor(
        model_path=str(model_path),
        metadata_path=str(metadata_path),
        scaler_path=str(scaler_path),
        label_encoder_path=str(label_encoder_path),
        device='cpu'
    )
    
    # Пытаемся найти overlay видео и создать analyzed видео
    poses_path_obj = Path(poses_path)
    poses_dir = poses_path_obj.parent
    overlay_candidates = list(poses_dir.glob("overlay_*.mp4"))
    
    create_video = len(overlay_candidates) > 0
    overlay_video_path = str(overlay_candidates[0]) if overlay_candidates else None
    
    result = predictor.predict_from_poses(
        poses_path, 
        video_path,
        create_analyzed_video=create_video,
        overlay_video_path=overlay_video_path,
        output_dir=str(poses_dir)
    )
    
    if result['success']:
        print(f"\nFigure: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        
        if 'spatial_similarity' in result and 'score' in result['spatial_similarity']:
            print(f"Spatial Quality: {result['spatial_similarity']['score']:.1f}/100")
        
        if 'classifier_clarity' in result and 'score' in result['classifier_clarity']:
            print(f"Clarity: {result['classifier_clarity']['score']:.1f}/100")
        
        if 'timing' in result:
            if isinstance(result['timing'], dict):
                if 'error' in result['timing']:
                    error_msg = result['timing']['error']
                    if 'нет аудио потока' in error_msg or 'no audio' in error_msg.lower():
                        print(f"Timing (Тайминг): не вычислена (нет аудио потока в видео)")
                    else:
                        print(f"Timing (Тайминг): не вычислена ({error_msg})")
                elif 'score' in result['timing']:
                    print(f"Timing: {result['timing']['score']:.1f}/100")
            else:
                print(f"Timing: {result['timing']:.1f}/100")
        
        if 'balance' in result and 'score' in result['balance']:
            print(f"Balance: {result['balance']['score']:.1f}/100")
        
        if 'analyzed_video_path' in result:
            print(f"\nAnalyzed video: {result['analyzed_video_path']}")
    else:
        print(f"\nError: {result['error']}")


if __name__ == '__main__':
    main()





