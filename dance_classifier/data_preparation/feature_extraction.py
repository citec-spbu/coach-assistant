"""
Модуль извлечения признаков из поз для классификации танцевальных фигур
"""
import numpy as np
from typing import List, Dict, Tuple


# Индексы ключевых точек YOLO-Pose (COCO формат)
# 0: нос, 1-2: глаза, 3-4: уши
# 5-6: плечи, 7-8: локти, 9-10: запястья
# 11-12: бедра, 13-14: колени, 15-16: лодыжки
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Индексы для удобного доступа
KP = {name: idx for idx, name in enumerate(KEYPOINT_NAMES)}


class FeatureExtractor:
    """Класс для извлечения признаков из поз"""
    
    def __init__(self, min_confidence=0.3):
        """
        Args:
            min_confidence: минимальная уверенность для использования ключевой точки
        """
        self.min_confidence = min_confidence
    
    def compute_angle(self, p1, p2, p3):
        """
        Вычисляет угол между тремя точками (p2 - вершина угла)
        
        Args:
            p1, p2, p3: точки в формате [x, y, confidence]
        
        Returns:
            float: угол в градусах или None если точки невалидны
        """
        # Проверяем уверенность
        if (p1[2] < self.min_confidence or 
            p2[2] < self.min_confidence or 
            p3[2] < self.min_confidence):
            return None
        
        # Векторы
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        # Вычисляем угол
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def compute_distance(self, p1, p2, normalize=True, height=1.0):
        """
        Вычисляет расстояние между двумя точками
        
        Args:
            p1, p2: точки в формате [x, y, confidence]
            normalize: нормализовать по высоте человека
            height: высота для нормализации
        
        Returns:
            float: расстояние или None если точки невалидны
        """
        if p1[2] < self.min_confidence or p2[2] < self.min_confidence:
            return None
        
        dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        if normalize and height > 0:
            dist = dist / height
        
        return dist
    
    def compute_velocity(self, p1_curr, p1_prev, fps=25.0):
        """
        Вычисляет скорость движения точки между кадрами
        
        Args:
            p1_curr: текущая позиция [x, y, confidence]
            p1_prev: предыдущая позиция [x, y, confidence]
            fps: частота кадров
        
        Returns:
            float: скорость (пикселей/сек) или None если точки невалидны
        """
        if (p1_curr[2] < self.min_confidence or 
            p1_prev[2] < self.min_confidence):
            return None
        
        dx = p1_curr[0] - p1_prev[0]
        dy = p1_curr[1] - p1_prev[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        velocity = dist * fps
        return velocity
    
    def estimate_person_height(self, keypoints):
        """
        Оценивает высоту человека для нормализации
        
        Args:
            keypoints: массив ключевых точек [[x, y, conf], ...]
        
        Returns:
            float: оценка высоты
        """
        if len(keypoints) < 17:
            return 1.0
        
        # Используем расстояние от носа до лодыжки
        nose = keypoints[KP['nose']]
        left_ankle = keypoints[KP['left_ankle']]
        right_ankle = keypoints[KP['right_ankle']]
        
        # Выбираем более уверенную лодыжку
        ankle = left_ankle if left_ankle[2] > right_ankle[2] else right_ankle
        
        if nose[2] < self.min_confidence or ankle[2] < self.min_confidence:
            # Альтернативная оценка: от плеча до лодыжки
            shoulder = keypoints[KP['left_shoulder']]
            if shoulder[2] >= self.min_confidence and ankle[2] >= self.min_confidence:
                return np.sqrt((shoulder[0] - ankle[0])**2 + (shoulder[1] - ankle[1])**2) * 1.2
            return 1.0
        
        height = np.sqrt((nose[0] - ankle[0])**2 + (nose[1] - ankle[1])**2)
        return max(height, 1.0)
    
    def extract_frame_features(self, keypoints, prev_keypoints=None, fps=25.0):
        """
        Извлекает признаки из одного кадра
        
        Args:
            keypoints: массив ключевых точек текущего кадра [[x, y, conf], ...]
            prev_keypoints: массив ключевых точек предыдущего кадра (для скоростей)
            fps: частота кадров
        
        Returns:
            dict: словарь с признаками
        """
        if len(keypoints) < 17:
            return None
        
        kps = np.array(keypoints)
        height = self.estimate_person_height(kps)
        
        features = {}
        
        # ========== УГЛЫ СУСТАВОВ ==========
        # Левый локоть
        features['left_elbow_angle'] = self.compute_angle(
            kps[KP['left_shoulder']], kps[KP['left_elbow']], kps[KP['left_wrist']]
        )
        
        # Правый локоть
        features['right_elbow_angle'] = self.compute_angle(
            kps[KP['right_shoulder']], kps[KP['right_elbow']], kps[KP['right_wrist']]
        )
        
        # Левое колено
        features['left_knee_angle'] = self.compute_angle(
            kps[KP['left_hip']], kps[KP['left_knee']], kps[KP['left_ankle']]
        )
        
        # Правое колено
        features['right_knee_angle'] = self.compute_angle(
            kps[KP['right_hip']], kps[KP['right_knee']], kps[KP['right_ankle']]
        )
        
        # Левое бедро
        features['left_hip_angle'] = self.compute_angle(
            kps[KP['left_shoulder']], kps[KP['left_hip']], kps[KP['left_knee']]
        )
        
        # Правое бедро
        features['right_hip_angle'] = self.compute_angle(
            kps[KP['right_shoulder']], kps[KP['right_hip']], kps[KP['right_knee']]
        )
        
        # ========== РАССТОЯНИЯ ==========
        # Расстояние между запястьями (ширина рук)
        features['wrists_distance'] = self.compute_distance(
            kps[KP['left_wrist']], kps[KP['right_wrist']], 
            normalize=True, height=height
        )
        
        # Расстояние между лодыжками (ширина ног)
        features['ankles_distance'] = self.compute_distance(
            kps[KP['left_ankle']], kps[KP['right_ankle']], 
            normalize=True, height=height
        )
        
        # Высота левой руки относительно плеча
        left_wrist_height = None
        if kps[KP['left_wrist']][2] >= self.min_confidence and kps[KP['left_shoulder']][2] >= self.min_confidence:
            left_wrist_height = (kps[KP['left_shoulder']][1] - kps[KP['left_wrist']][1]) / height
        features['left_wrist_height'] = left_wrist_height
        
        # Высота правой руки относительно плеча
        right_wrist_height = None
        if kps[KP['right_wrist']][2] >= self.min_confidence and kps[KP['right_shoulder']][2] >= self.min_confidence:
            right_wrist_height = (kps[KP['right_shoulder']][1] - kps[KP['right_wrist']][1]) / height
        features['right_wrist_height'] = right_wrist_height
        
        # ========== СКОРОСТИ (если есть предыдущий кадр) ==========
        if prev_keypoints is not None and len(prev_keypoints) == 17:
            prev_kps = np.array(prev_keypoints)
            
            # Скорость левой руки
            features['left_wrist_velocity'] = self.compute_velocity(
                kps[KP['left_wrist']], prev_kps[KP['left_wrist']], fps
            )
            
            # Скорость правой руки
            features['right_wrist_velocity'] = self.compute_velocity(
                kps[KP['right_wrist']], prev_kps[KP['right_wrist']], fps
            )
            
            # Скорость левой ноги
            features['left_ankle_velocity'] = self.compute_velocity(
                kps[KP['left_ankle']], prev_kps[KP['left_ankle']], fps
            )
            
            # Скорость правой ноги
            features['right_ankle_velocity'] = self.compute_velocity(
                kps[KP['right_ankle']], prev_kps[KP['right_ankle']], fps
            )
        
        return features
    
    def extract_sequence_features(self, poses_data):
        """
        Извлекает признаки из последовательности кадров
        
        Args:
            poses_data: данные поз из JSON файла
        
        Returns:
            tuple: (features_array, valid_mask) где
                features_array: numpy array размера (num_frames, num_features)
                valid_mask: булев массив валидных кадров
        """
        frames = poses_data['frames']
        fps = poses_data.get('fps', 25.0)
        
        feature_list = []
        valid_mask = []
        
        prev_keypoints = None
        
        for frame_data in frames:
            if not frame_data['valid'] or 'keypoints' not in frame_data:
                feature_list.append(None)
                valid_mask.append(False)
                prev_keypoints = None
                continue
            
            keypoints = frame_data['keypoints']
            features = self.extract_frame_features(keypoints, prev_keypoints, fps)
            
            if features is None:
                feature_list.append(None)
                valid_mask.append(False)
            else:
                feature_list.append(features)
                valid_mask.append(True)
            
            prev_keypoints = keypoints
        
        return feature_list, valid_mask
    
    def features_to_array(self, feature_list, feature_names=None):
        """
        Преобразует список словарей признаков в numpy array
        
        Args:
            feature_list: список словарей с признаками
            feature_names: список имен признаков для использования (по умолчанию все)
        
        Returns:
            tuple: (feature_array, feature_names, valid_indices)
        """
        # Собираем все возможные имена признаков
        if feature_names is None:
            feature_names = set()
            for features in feature_list:
                if features is not None:
                    feature_names.update(features.keys())
            feature_names = sorted(list(feature_names))
        
        # Создаем массив
        num_frames = len(feature_list)
        num_features = len(feature_names)
        feature_array = np.full((num_frames, num_features), np.nan)
        valid_indices = []
        
        for i, features in enumerate(feature_list):
            if features is None:
                continue
            
            valid = True
            for j, name in enumerate(feature_names):
                value = features.get(name)
                if value is None:
                    valid = False
                else:
                    feature_array[i, j] = value
            
            if valid:
                valid_indices.append(i)
        
        return feature_array, feature_names, valid_indices


def get_simple_features():
    """Возвращает список простых показательных признаков для первичного анализа"""
    return [
        'left_elbow_angle',
        'right_elbow_angle', 
        'left_knee_angle',
        'right_knee_angle',
        'wrists_distance',
        'ankles_distance',
        'left_wrist_height',
        'right_wrist_height'
    ]


if __name__ == "__main__":
    # Пример использования
    print("Доступные признаки:")
    print("\nУглы суставов:")
    print("- left_elbow_angle, right_elbow_angle")
    print("- left_knee_angle, right_knee_angle")
    print("- left_hip_angle, right_hip_angle")
    print("\nРасстояния:")
    print("- wrists_distance, ankles_distance")
    print("- left_wrist_height, right_wrist_height")
    print("\nСкорости:")
    print("- left_wrist_velocity, right_wrist_velocity")
    print("- left_ankle_velocity, right_ankle_velocity")
    print("\nПростые признаки для первичного анализа:")
    print(get_simple_features())


Модуль извлечения признаков из поз для классификации танцевальных фигур
"""
import numpy as np
from typing import List, Dict, Tuple


# Индексы ключевых точек YOLO-Pose (COCO формат)
# 0: нос, 1-2: глаза, 3-4: уши
# 5-6: плечи, 7-8: локти, 9-10: запястья
# 11-12: бедра, 13-14: колени, 15-16: лодыжки
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Индексы для удобного доступа
KP = {name: idx for idx, name in enumerate(KEYPOINT_NAMES)}


class FeatureExtractor:
    """Класс для извлечения признаков из поз"""
    
    def __init__(self, min_confidence=0.3):
        """
        Args:
            min_confidence: минимальная уверенность для использования ключевой точки
        """
        self.min_confidence = min_confidence
    
    def compute_angle(self, p1, p2, p3):
        """
        Вычисляет угол между тремя точками (p2 - вершина угла)
        
        Args:
            p1, p2, p3: точки в формате [x, y, confidence]
        
        Returns:
            float: угол в градусах или None если точки невалидны
        """
        # Проверяем уверенность
        if (p1[2] < self.min_confidence or 
            p2[2] < self.min_confidence or 
            p3[2] < self.min_confidence):
            return None
        
        # Векторы
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        # Вычисляем угол
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def compute_distance(self, p1, p2, normalize=True, height=1.0):
        """
        Вычисляет расстояние между двумя точками
        
        Args:
            p1, p2: точки в формате [x, y, confidence]
            normalize: нормализовать по высоте человека
            height: высота для нормализации
        
        Returns:
            float: расстояние или None если точки невалидны
        """
        if p1[2] < self.min_confidence or p2[2] < self.min_confidence:
            return None
        
        dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        if normalize and height > 0:
            dist = dist / height
        
        return dist
    
    def compute_velocity(self, p1_curr, p1_prev, fps=25.0):
        """
        Вычисляет скорость движения точки между кадрами
        
        Args:
            p1_curr: текущая позиция [x, y, confidence]
            p1_prev: предыдущая позиция [x, y, confidence]
            fps: частота кадров
        
        Returns:
            float: скорость (пикселей/сек) или None если точки невалидны
        """
        if (p1_curr[2] < self.min_confidence or 
            p1_prev[2] < self.min_confidence):
            return None
        
        dx = p1_curr[0] - p1_prev[0]
        dy = p1_curr[1] - p1_prev[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        velocity = dist * fps
        return velocity
    
    def estimate_person_height(self, keypoints):
        """
        Оценивает высоту человека для нормализации
        
        Args:
            keypoints: массив ключевых точек [[x, y, conf], ...]
        
        Returns:
            float: оценка высоты
        """
        if len(keypoints) < 17:
            return 1.0
        
        # Используем расстояние от носа до лодыжки
        nose = keypoints[KP['nose']]
        left_ankle = keypoints[KP['left_ankle']]
        right_ankle = keypoints[KP['right_ankle']]
        
        # Выбираем более уверенную лодыжку
        ankle = left_ankle if left_ankle[2] > right_ankle[2] else right_ankle
        
        if nose[2] < self.min_confidence or ankle[2] < self.min_confidence:
            # Альтернативная оценка: от плеча до лодыжки
            shoulder = keypoints[KP['left_shoulder']]
            if shoulder[2] >= self.min_confidence and ankle[2] >= self.min_confidence:
                return np.sqrt((shoulder[0] - ankle[0])**2 + (shoulder[1] - ankle[1])**2) * 1.2
            return 1.0
        
        height = np.sqrt((nose[0] - ankle[0])**2 + (nose[1] - ankle[1])**2)
        return max(height, 1.0)
    
    def extract_frame_features(self, keypoints, prev_keypoints=None, fps=25.0):
        """
        Извлекает признаки из одного кадра
        
        Args:
            keypoints: массив ключевых точек текущего кадра [[x, y, conf], ...]
            prev_keypoints: массив ключевых точек предыдущего кадра (для скоростей)
            fps: частота кадров
        
        Returns:
            dict: словарь с признаками
        """
        if len(keypoints) < 17:
            return None
        
        kps = np.array(keypoints)
        height = self.estimate_person_height(kps)
        
        features = {}
        
        # ========== УГЛЫ СУСТАВОВ ==========
        # Левый локоть
        features['left_elbow_angle'] = self.compute_angle(
            kps[KP['left_shoulder']], kps[KP['left_elbow']], kps[KP['left_wrist']]
        )
        
        # Правый локоть
        features['right_elbow_angle'] = self.compute_angle(
            kps[KP['right_shoulder']], kps[KP['right_elbow']], kps[KP['right_wrist']]
        )
        
        # Левое колено
        features['left_knee_angle'] = self.compute_angle(
            kps[KP['left_hip']], kps[KP['left_knee']], kps[KP['left_ankle']]
        )
        
        # Правое колено
        features['right_knee_angle'] = self.compute_angle(
            kps[KP['right_hip']], kps[KP['right_knee']], kps[KP['right_ankle']]
        )
        
        # Левое бедро
        features['left_hip_angle'] = self.compute_angle(
            kps[KP['left_shoulder']], kps[KP['left_hip']], kps[KP['left_knee']]
        )
        
        # Правое бедро
        features['right_hip_angle'] = self.compute_angle(
            kps[KP['right_shoulder']], kps[KP['right_hip']], kps[KP['right_knee']]
        )
        
        # ========== РАССТОЯНИЯ ==========
        # Расстояние между запястьями (ширина рук)
        features['wrists_distance'] = self.compute_distance(
            kps[KP['left_wrist']], kps[KP['right_wrist']], 
            normalize=True, height=height
        )
        
        # Расстояние между лодыжками (ширина ног)
        features['ankles_distance'] = self.compute_distance(
            kps[KP['left_ankle']], kps[KP['right_ankle']], 
            normalize=True, height=height
        )
        
        # Высота левой руки относительно плеча
        left_wrist_height = None
        if kps[KP['left_wrist']][2] >= self.min_confidence and kps[KP['left_shoulder']][2] >= self.min_confidence:
            left_wrist_height = (kps[KP['left_shoulder']][1] - kps[KP['left_wrist']][1]) / height
        features['left_wrist_height'] = left_wrist_height
        
        # Высота правой руки относительно плеча
        right_wrist_height = None
        if kps[KP['right_wrist']][2] >= self.min_confidence and kps[KP['right_shoulder']][2] >= self.min_confidence:
            right_wrist_height = (kps[KP['right_shoulder']][1] - kps[KP['right_wrist']][1]) / height
        features['right_wrist_height'] = right_wrist_height
        
        # ========== СКОРОСТИ (если есть предыдущий кадр) ==========
        if prev_keypoints is not None and len(prev_keypoints) == 17:
            prev_kps = np.array(prev_keypoints)
            
            # Скорость левой руки
            features['left_wrist_velocity'] = self.compute_velocity(
                kps[KP['left_wrist']], prev_kps[KP['left_wrist']], fps
            )
            
            # Скорость правой руки
            features['right_wrist_velocity'] = self.compute_velocity(
                kps[KP['right_wrist']], prev_kps[KP['right_wrist']], fps
            )
            
            # Скорость левой ноги
            features['left_ankle_velocity'] = self.compute_velocity(
                kps[KP['left_ankle']], prev_kps[KP['left_ankle']], fps
            )
            
            # Скорость правой ноги
            features['right_ankle_velocity'] = self.compute_velocity(
                kps[KP['right_ankle']], prev_kps[KP['right_ankle']], fps
            )
        
        return features
    
    def extract_sequence_features(self, poses_data):
        """
        Извлекает признаки из последовательности кадров
        
        Args:
            poses_data: данные поз из JSON файла
        
        Returns:
            tuple: (features_array, valid_mask) где
                features_array: numpy array размера (num_frames, num_features)
                valid_mask: булев массив валидных кадров
        """
        frames = poses_data['frames']
        fps = poses_data.get('fps', 25.0)
        
        feature_list = []
        valid_mask = []
        
        prev_keypoints = None
        
        for frame_data in frames:
            if not frame_data['valid'] or 'keypoints' not in frame_data:
                feature_list.append(None)
                valid_mask.append(False)
                prev_keypoints = None
                continue
            
            keypoints = frame_data['keypoints']
            features = self.extract_frame_features(keypoints, prev_keypoints, fps)
            
            if features is None:
                feature_list.append(None)
                valid_mask.append(False)
            else:
                feature_list.append(features)
                valid_mask.append(True)
            
            prev_keypoints = keypoints
        
        return feature_list, valid_mask
    
    def features_to_array(self, feature_list, feature_names=None):
        """
        Преобразует список словарей признаков в numpy array
        
        Args:
            feature_list: список словарей с признаками
            feature_names: список имен признаков для использования (по умолчанию все)
        
        Returns:
            tuple: (feature_array, feature_names, valid_indices)
        """
        # Собираем все возможные имена признаков
        if feature_names is None:
            feature_names = set()
            for features in feature_list:
                if features is not None:
                    feature_names.update(features.keys())
            feature_names = sorted(list(feature_names))
        
        # Создаем массив
        num_frames = len(feature_list)
        num_features = len(feature_names)
        feature_array = np.full((num_frames, num_features), np.nan)
        valid_indices = []
        
        for i, features in enumerate(feature_list):
            if features is None:
                continue
            
            valid = True
            for j, name in enumerate(feature_names):
                value = features.get(name)
                if value is None:
                    valid = False
                else:
                    feature_array[i, j] = value
            
            if valid:
                valid_indices.append(i)
        
        return feature_array, feature_names, valid_indices


def get_simple_features():
    """Возвращает список простых показательных признаков для первичного анализа"""
    return [
        'left_elbow_angle',
        'right_elbow_angle', 
        'left_knee_angle',
        'right_knee_angle',
        'wrists_distance',
        'ankles_distance',
        'left_wrist_height',
        'right_wrist_height'
    ]


if __name__ == "__main__":
    # Пример использования
    print("Доступные признаки:")
    print("\nУглы суставов:")
    print("- left_elbow_angle, right_elbow_angle")
    print("- left_knee_angle, right_knee_angle")
    print("- left_hip_angle, right_hip_angle")
    print("\nРасстояния:")
    print("- wrists_distance, ankles_distance")
    print("- left_wrist_height, right_wrist_height")
    print("\nСкорости:")
    print("- left_wrist_velocity, right_wrist_velocity")
    print("- left_ankle_velocity, right_ankle_velocity")
    print("\nПростые признаки для первичного анализа:")
    print(get_simple_features())


