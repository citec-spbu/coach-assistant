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
    
    result = predictor.predict_from_poses("poses.jsonl", video_path="video.mp4")
"""

import torch
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
        sequence_norm = self.scaler.transform(sequence)
        X = torch.FloatTensor(sequence_norm).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        predicted_idx = np.argmax(probs)
        predicted_class = self.label_encoder.classes_[predicted_idx]
        
        return predicted_class, probs
    
    def predict_from_poses(
        self,
        poses_json_path: str,
        video_path: Optional[str] = None,
        compute_metrics: bool = True
    ) -> Dict:
        """
        Полное предсказание с метриками качества.
        
        Args:
            poses_json_path: путь к poses.jsonl
            video_path: путь к видео (для тайминг-метрики)
            compute_metrics: вычислять ли DTW-метрики
        
        Returns:
            dict с результатами классификации и метриками
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
        print("3")
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
            
            print("4")
            return result
        except Exception as e:
            import traceback
            error_msg = f"Ошибка при извлечении признаков: {str(e)}\n{traceback.format_exc()}"
            print(f"ERROR: {error_msg}")
            return {
                'success': False,
                'predicted_class': 'Unknown',
                'predicted_figure': 'Unknown',  # Для совместимости с фронтендом
                'confidence': 0.0,
                'classification': {  # Старый формат для совместимости
                    'figure': 'Unknown',
                    'confidence': 0.0
                },
                'error': error_msg,
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
                "score": float(result.score),
                "mean_distance": float(result.mean_distance),
                "reference_figure": predicted_class,
                "error_segments": result.error_segments if hasattr(result, 'error_segments') else []
            }
        except Exception as e:
            return {"error": str(e)}
    
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
    if (main_coach_dir / "best_model_20pct_adapted.pth").exists():
        model_path = main_coach_dir / "best_model_20pct_adapted.pth"
    elif (main_coach_dir / "best_model_20pct.pth").exists():
        model_path = main_coach_dir / "best_model_20pct.pth"
    else:
        model_path = dance_classifier_root / "best_model_20pct_adapted.pth"
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
    
    result = predictor.predict_from_poses(poses_path, video_path)
    
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
    else:
        print(f"\nError: {result['error']}")


if __name__ == '__main__':
    main()





