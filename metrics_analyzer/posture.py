"""
Posture Analyzer - анализ осанки и положения корпуса
"""

import numpy as np
from typing import List, Dict


class PostureAnalyzer:
    """Анализатор осанки танцора"""
    
    def __init__(self):
        self.spine_angles = []
        self.head_heights = []
    
    def calculate_spine_angle(self, keypoints: List[List[float]]) -> float:
        """
        Вычисляет угол наклона позвоночника
        
        Args:
            keypoints: Координаты ключевых точек
            
        Returns:
            Угол наклона позвоночника в градусах (0 = прямо, + = вперёд, - = назад)
        """
        if len(keypoints) < 17:
            return 0.0
        
        # Среднее плеч (берём только x, y)
        left_shoulder = np.array(keypoints[5][:2])
        right_shoulder = np.array(keypoints[6][:2])
        shoulders_center = (left_shoulder + right_shoulder) / 2.0
        
        # Среднее бёдер (берём только x, y)
        left_hip = np.array(keypoints[11][:2])
        right_hip = np.array(keypoints[12][:2])
        hips_center = (left_hip + right_hip) / 2.0
        
        # Вектор позвоночника
        spine_vector = shoulders_center - hips_center
        
        # Угол относительно вертикали
        vertical = np.array([0, -1])  # Y вниз в системе координат изображения
        
        # Нормализуем
        spine_norm = np.linalg.norm(spine_vector)
        if spine_norm == 0:
            return 0.0
        
        spine_vector_norm = spine_vector / spine_norm
        
        # Косинус угла
        cos_angle = np.dot(spine_vector_norm, vertical)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = np.degrees(angle)
        
        # Определяем направление (вперёд/назад)
        if spine_vector[0] > 0:  # X > 0 = наклон вперёд
            return float(angle_deg)
        else:
            return float(-angle_deg)
    
    def calculate_head_height(self, keypoints: List[List[float]]) -> float:
        """
        Вычисляет относительную высоту головы
        
        Args:
            keypoints: Координаты ключевых точек
            
        Returns:
            Относительная высота (0-1, где 1 = максимально поднята)
        """
        if len(keypoints) < 17:
            return 0.0
        
        nose = np.array(keypoints[0][:2])
        
        # Среднее плеч (берём только y)
        left_shoulder = np.array(keypoints[5][:2])
        right_shoulder = np.array(keypoints[6][:2])
        shoulders_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
        
        # Расстояние от носа до плеч (чем больше, тем голова выше)
        head_height = shoulders_y - nose[1]  # Y вниз, поэтому вычитаем
        
        # Нормализуем (нормальное расстояние ~50-80 пикселей)
        normalized_height = head_height / 65.0
        normalized_height = np.clip(normalized_height, 0.0, 1.5)
        
        return float(normalized_height)
    
    def calculate_chest_openness(self, keypoints: List[List[float]]) -> float:
        """
        Вычисляет раскрытость грудной клетки (ширину плеч)
        
        Args:
            keypoints: Координаты ключевых точек
            
        Returns:
            Ширина плеч (нормализованная)
        """
        if len(keypoints) < 17:
            return 0.0
        
        left_shoulder = np.array(keypoints[5][:2])
        right_shoulder = np.array(keypoints[6][:2])
        
        # Расстояние между плечами
        distance = np.linalg.norm(left_shoulder - right_shoulder)
        
        # Нормализуем (нормальная ширина ~100-150 пикселей)
        normalized = distance / 125.0
        normalized = np.clip(normalized, 0.0, 1.5)
        
        return float(normalized)
    
    def analyze_posture(self, poses: List[Dict]) -> Dict:
        """
        Полный анализ осанки
        
        Args:
            poses: Список поз
            
        Returns:
            Результаты анализа
        """
        if len(poses) == 0:
            return {
                'score': 0,
                'status': 'no_data',
                'details': {}
            }
        
        spine_angles = []
        head_heights = []
        chest_openness = []
        
        for pose in poses:
            if not pose.get('valid', False):
                continue
            
            keypoints = pose.get('keypoints', [])
            if len(keypoints) < 17:
                continue
            
            spine_angle = self.calculate_spine_angle(keypoints)
            head_height = self.calculate_head_height(keypoints)
            chest = self.calculate_chest_openness(keypoints)
            
            spine_angles.append(spine_angle)
            head_heights.append(head_height)
            chest_openness.append(chest)
        
        if len(spine_angles) == 0:
            return {
                'score': 0,
                'status': 'no_data',
                'details': {}
            }
        
        self.spine_angles = np.array(spine_angles)
        self.head_heights = np.array(head_heights)
        
        # Средние значения
        avg_spine_angle = np.mean(spine_angles)
        avg_head_height = np.mean(head_heights)
        avg_chest = np.mean(chest_openness)
        
        # Оценка позвоночника (чем ближе к 0°, тем лучше)
        spine_score = max(0, 100 - abs(avg_spine_angle) * 5)  # 20° = -100 очков
        
        # Оценка высоты головы (должна быть ~1.0)
        head_score = max(0, 100 - abs(avg_head_height - 1.0) * 100)
        
        # Оценка раскрытости груди (должна быть ~1.0)
        chest_score = max(0, 100 - abs(avg_chest - 1.0) * 100)
        
        # Общая оценка
        score = 0.4 * spine_score + 0.3 * head_score + 0.3 * chest_score
        
        # Статус
        if score >= 90:
            status = 'excellent'
        elif score >= 75:
            status = 'good'
        elif score >= 60:
            status = 'fair'
        else:
            status = 'poor'
        
        return {
            'score': float(score),
            'status': status,
            'details': {
                'spine_angle': float(avg_spine_angle),
                'head_height': float(avg_head_height),
                'chest_openness': float(avg_chest),
                'spine_score': float(spine_score),
                'head_score': float(head_score),
                'chest_score': float(chest_score)
            }
        }
    
    def find_posture_issues(
        self, 
        poses: List[Dict],
        spine_threshold: float = 15.0,
        head_threshold: float = 0.3
    ) -> List[Dict]:
        """
        Находит моменты с проблемами осанки
        
        Args:
            poses: Список поз
            spine_threshold: Допустимый угол наклона позвоночника (градусы)
            head_threshold: Допустимое отклонение высоты головы
            
        Returns:
            Список ошибок
        """
        if len(self.spine_angles) == 0:
            self.analyze_posture(poses)
        
        if len(self.spine_angles) == 0:
            return []
        
        errors = []
        
        for frame_idx in range(len(self.spine_angles)):
            spine_angle = self.spine_angles[frame_idx]
            head_height = self.head_heights[frame_idx]
            
            # Проверка позвоночника
            if abs(spine_angle) > spine_threshold:
                errors.append({
                    'frame': frame_idx,
                    'timestamp': frame_idx / 30.0,
                    'category': 'posture',
                    'joint': 'spine',
                    'severity': 'high' if abs(spine_angle) > 25 else 'medium',
                    'issue': 'Body tilted (spine angle deviation)',
                    'current_value': float(spine_angle),
                    'expected_value': 0.0,
                    'deviation': float(abs(spine_angle))
                })
            
            # Проверка головы
            if abs(head_height - 1.0) > head_threshold:
                direction = 'down' if head_height < 1.0 else 'up'
                errors.append({
                    'frame': frame_idx,
                    'timestamp': frame_idx / 30.0,
                    'category': 'posture',
                    'joint': 'head',
                    'severity': 'low',
                    'issue': f'Head tilted {direction}',
                    'current_value': float(head_height),
                    'expected_value': 1.0,
                    'deviation': float(abs(head_height - 1.0))
                })
        
        return errors


def analyze_posture_from_file(poses_file: str) -> Dict:
    """
    Удобная функция для анализа осанки из файла
    
    Args:
        poses_file: Путь к poses.jsonl
        
    Returns:
        Результаты анализа
    """
    import json
    
    poses = []
    with open(poses_file, 'r') as f:
        for line in f:
            poses.append(json.loads(line))
    
    analyzer = PostureAnalyzer()
    result = analyzer.analyze_posture(poses)
    errors = analyzer.find_posture_issues(poses)
    
    return {
        'score': result,
        'errors': errors
    }

