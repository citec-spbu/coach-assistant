"""
Извлечение признаков из поз для классификации танцевальных движений
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


def get_simple_features():
    """Возвращает список базовых признаков для извлечения"""
    features = []
    
    # Координаты ключевых точек (x, y)
    keypoints = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    for kp in keypoints:
        features.append(f'{kp}_x')
        features.append(f'{kp}_y')
    
    # Углы между суставами
    features.extend([
        'left_elbow_angle',
        'right_elbow_angle',
        'left_knee_angle',
        'right_knee_angle',
        'left_hip_angle',
        'right_hip_angle',
        'left_shoulder_angle',
        'right_shoulder_angle'
    ])
    
    # Расстояния
    features.extend([
        'shoulder_width',
        'hip_width',
        'torso_length',
        'left_arm_length',
        'right_arm_length',
        'left_leg_length',
        'right_leg_length',
        'body_height',
        'center_x',
        'center_y'
    ])
    
    return features[:53]  # Возвращаем 53 признака


class FeatureExtractor:
    """Класс для извлечения признаков из поз"""
    
    # Индексы ключевых точек Mediapipe Pose
    KEYPOINT_INDICES = {
        'nose': 0,
        'left_eye_inner': 1,
        'left_eye': 2,
        'left_eye_outer': 3,
        'right_eye_inner': 4,
        'right_eye': 5,
        'right_eye_outer': 6,
        'left_ear': 7,
        'right_ear': 8,
        'mouth_left': 9,
        'mouth_right': 10,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_pinky': 17,
        'right_pinky': 18,
        'left_index': 19,
        'right_index': 20,
        'left_thumb': 21,
        'right_thumb': 22,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28,
        'left_heel': 29,
        'right_heel': 30,
        'left_foot_index': 31,
        'right_foot_index': 32
    }
    
    def __init__(self, min_confidence=0.3):
        """
        Args:
            min_confidence: минимальная уверенность для учета ключевой точки
        """
        self.min_confidence = min_confidence
    
    def extract_sequence_features(self, poses_data: Dict) -> Tuple[List[Dict], np.ndarray]:
        """
        Извлекает признаки из последовательности поз
        
        Args:
            poses_data: словарь с данными поз (формат из extract_poses.py)
        
        Returns:
            feature_list: список словарей с признаками для каждого кадра
            valid_mask: маска валидных кадров
        """
        frames = poses_data.get('frames', [])
        if not frames:
            return [], np.array([], dtype=bool)
        
        feature_list = []
        valid_mask = []
        
        for i, frame in enumerate(frames):
            try:
                features = self._extract_frame_features(frame)
                feature_list.append(features)
                
                # Кадр валиден, если есть хотя бы основные точки
                is_valid = self._is_frame_valid(features)
                valid_mask.append(is_valid)
            except Exception as e:
                # Если ошибка при обработке кадра, добавляем пустые признаки
                feature_list.append({})
                valid_mask.append(False)
        
        return feature_list, np.array(valid_mask, dtype=bool)
    
    def _extract_frame_features(self, frame: Dict) -> Dict:
        """Извлекает признаки из одного кадра"""
        features = {}
        
        # Получаем landmarks
        landmarks = frame.get('pose_world_landmarks') or frame.get('pose_landmarks')
        
        if landmarks is None or len(landmarks) < 33:
            return features
        
        # Извлекаем координаты ключевых точек
        keypoints = {}
        for name, idx in self.KEYPOINT_INDICES.items():
            if idx < len(landmarks):
                lm = landmarks[idx]
                if isinstance(lm, dict):
                    x = lm.get('x', np.nan)
                    y = lm.get('y', np.nan)
                    z = lm.get('z', np.nan)
                    visibility = lm.get('visibility', 1.0)
                elif isinstance(lm, (list, tuple)) and len(lm) >= 3:
                    x, y, z = lm[0], lm[1], lm[2]
                    visibility = lm[3] if len(lm) > 3 else 1.0
                else:
                    x, y, z = np.nan, np.nan, np.nan
                    visibility = 0.0
                
                if visibility < self.min_confidence:
                    x, y, z = np.nan, np.nan, np.nan
                
                keypoints[name] = {'x': x, 'y': y, 'z': z}
                features[f'{name}_x'] = x
                features[f'{name}_y'] = y
        
        # Вычисляем углы
        features.update(self._calculate_angles(keypoints))
        
        # Вычисляем расстояния
        features.update(self._calculate_distances(keypoints))
        
        return features
    
    def _calculate_angles(self, keypoints: Dict) -> Dict:
        """Вычисляет углы между суставами"""
        angles = {}
        
        # Угол левого локтя
        if all(k in keypoints for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
            angles['left_elbow_angle'] = self._angle_between_points(
                keypoints['left_shoulder'],
                keypoints['left_elbow'],
                keypoints['left_wrist']
            )
        
        # Угол правого локтя
        if all(k in keypoints for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
            angles['right_elbow_angle'] = self._angle_between_points(
                keypoints['right_shoulder'],
                keypoints['right_elbow'],
                keypoints['right_wrist']
            )
        
        # Угол левого колена
        if all(k in keypoints for k in ['left_hip', 'left_knee', 'left_ankle']):
            angles['left_knee_angle'] = self._angle_between_points(
                keypoints['left_hip'],
                keypoints['left_knee'],
                keypoints['left_ankle']
            )
        
        # Угол правого колена
        if all(k in keypoints for k in ['right_hip', 'right_knee', 'right_ankle']):
            angles['right_knee_angle'] = self._angle_between_points(
                keypoints['right_hip'],
                keypoints['right_knee'],
                keypoints['right_ankle']
            )
        
        # Углы бедер
        if all(k in keypoints for k in ['left_shoulder', 'left_hip', 'left_knee']):
            angles['left_hip_angle'] = self._angle_between_points(
                keypoints['left_shoulder'],
                keypoints['left_hip'],
                keypoints['left_knee']
            )
        
        if all(k in keypoints for k in ['right_shoulder', 'right_hip', 'right_knee']):
            angles['right_hip_angle'] = self._angle_between_points(
                keypoints['right_shoulder'],
                keypoints['right_hip'],
                keypoints['right_knee']
            )
        
        # Углы плеч
        if all(k in keypoints for k in ['left_hip', 'left_shoulder', 'left_elbow']):
            angles['left_shoulder_angle'] = self._angle_between_points(
                keypoints['left_hip'],
                keypoints['left_shoulder'],
                keypoints['left_elbow']
            )
        
        if all(k in keypoints for k in ['right_hip', 'right_shoulder', 'right_elbow']):
            angles['right_shoulder_angle'] = self._angle_between_points(
                keypoints['right_hip'],
                keypoints['right_shoulder'],
                keypoints['right_elbow']
            )
        
        return angles
    
    def _calculate_distances(self, keypoints: Dict) -> Dict:
        """Вычисляет расстояния между ключевыми точками"""
        distances = {}
        
        # Ширина плеч
        if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
            distances['shoulder_width'] = self._distance(
                keypoints['left_shoulder'],
                keypoints['right_shoulder']
            )
        
        # Ширина бедер
        if 'left_hip' in keypoints and 'right_hip' in keypoints:
            distances['hip_width'] = self._distance(
                keypoints['left_hip'],
                keypoints['right_hip']
            )
        
        # Длина торса
        if 'left_shoulder' in keypoints and 'left_hip' in keypoints:
            distances['torso_length'] = self._distance(
                keypoints['left_shoulder'],
                keypoints['left_hip']
            )
        
        # Длина левой руки
        if all(k in keypoints for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
            d1 = self._distance(keypoints['left_shoulder'], keypoints['left_elbow'])
            d2 = self._distance(keypoints['left_elbow'], keypoints['left_wrist'])
            distances['left_arm_length'] = d1 + d2
        
        # Длина правой руки
        if all(k in keypoints for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
            d1 = self._distance(keypoints['right_shoulder'], keypoints['right_elbow'])
            d2 = self._distance(keypoints['right_elbow'], keypoints['right_wrist'])
            distances['right_arm_length'] = d1 + d2
        
        # Длина левой ноги
        if all(k in keypoints for k in ['left_hip', 'left_knee', 'left_ankle']):
            d1 = self._distance(keypoints['left_hip'], keypoints['left_knee'])
            d2 = self._distance(keypoints['left_knee'], keypoints['left_ankle'])
            distances['left_leg_length'] = d1 + d2
        
        # Длина правой ноги
        if all(k in keypoints for k in ['right_hip', 'right_knee', 'right_ankle']):
            d1 = self._distance(keypoints['right_hip'], keypoints['right_knee'])
            d2 = self._distance(keypoints['right_knee'], keypoints['right_ankle'])
            distances['right_leg_length'] = d1 + d2
        
        # Высота тела (от ankles до nose)
        if 'nose' in keypoints and 'left_ankle' in keypoints and 'right_ankle' in keypoints:
            ankle_y = (keypoints['left_ankle'].get('y', 0) + keypoints['right_ankle'].get('y', 0)) / 2
            nose_y = keypoints['nose'].get('y', 0)
            if not np.isnan(ankle_y) and not np.isnan(nose_y):
                distances['body_height'] = abs(nose_y - ankle_y)
        
        # Центр тела (среднее между плечами и бедрами)
        if all(k in keypoints for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
            center_x = (
                keypoints['left_shoulder'].get('x', 0) + 
                keypoints['right_shoulder'].get('x', 0) +
                keypoints['left_hip'].get('x', 0) +
                keypoints['right_hip'].get('x', 0)
            ) / 4
            center_y = (
                keypoints['left_shoulder'].get('y', 0) + 
                keypoints['right_shoulder'].get('y', 0) +
                keypoints['left_hip'].get('y', 0) +
                keypoints['right_hip'].get('y', 0)
            ) / 4
            distances['center_x'] = center_x
            distances['center_y'] = center_y
        
        return distances
    
    def _angle_between_points(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """Вычисляет угол между тремя точками (p2 - вершина)"""
        x1, y1 = p1.get('x', np.nan), p1.get('y', np.nan)
        x2, y2 = p2.get('x', np.nan), p2.get('y', np.nan)
        x3, y3 = p3.get('x', np.nan), p3.get('y', np.nan)
        
        if any(np.isnan([x1, y1, x2, y2, x3, y3])):
            return np.nan
        
        # Векторы
        v1 = np.array([x1 - x2, y1 - y2])
        v2 = np.array([x3 - x2, y3 - y2])
        
        # Угол
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def _distance(self, p1: Dict, p2: Dict) -> float:
        """Вычисляет евклидово расстояние между двумя точками"""
        x1, y1 = p1.get('x', np.nan), p1.get('y', np.nan)
        x2, y2 = p2.get('x', np.nan), p2.get('y', np.nan)
        
        if any(np.isnan([x1, y1, x2, y2])):
            return np.nan
        
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _is_frame_valid(self, features: Dict) -> bool:
        """Проверяет, валиден ли кадр (есть ли основные точки)"""
        required_features = [
            'left_shoulder_x', 'left_shoulder_y',
            'right_shoulder_x', 'right_shoulder_y',
            'left_hip_x', 'left_hip_y',
            'right_hip_x', 'right_hip_y'
        ]
        
        for feat in required_features:
            if feat not in features or np.isnan(features[feat]):
                return False
        
        return True
    
    def features_to_array(self, feature_list: List[Dict], feature_names: List[str]) -> Tuple[np.ndarray, List[str], List[int]]:
        """
        Преобразует список словарей признаков в numpy массив
        
        Args:
            feature_list: список словарей с признаками
            feature_names: список имен признаков для извлечения
        
        Returns:
            feature_array: массив признаков (num_frames, num_features)
            actual_feature_names: список имен признаков в массиве
            valid_indices: индексы валидных кадров
        """
        if not feature_list:
            return np.array([]), [], []
        
        # Определяем, какие признаки доступны
        available_features = set()
        for features in feature_list:
            available_features.update(features.keys())
        
        # Используем только доступные признаки
        actual_feature_names = [f for f in feature_names if f in available_features]
        
        if not actual_feature_names:
            return np.array([]), [], []
        
        # Создаем массив
        num_frames = len(feature_list)
        num_features = len(actual_feature_names)
        feature_array = np.full((num_frames, num_features), np.nan, dtype=np.float32)
        
        valid_indices = []
        for i, features in enumerate(feature_list):
            for j, feat_name in enumerate(actual_feature_names):
                if feat_name in features:
                    feature_array[i, j] = features[feat_name]
            
            # Кадр валиден, если хотя бы половина признаков не NaN
            if np.sum(~np.isnan(feature_array[i])) >= num_features // 2:
                valid_indices.append(i)
        
        return feature_array, actual_feature_names, valid_indices

