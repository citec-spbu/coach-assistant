"""
Balance Analyzer - анализ баланса и стабильности
"""

import numpy as np
from typing import List, Dict


class BalanceAnalyzer:
    """Анализатор баланса танцора"""
    
    def __init__(self):
        self.center_of_mass_history = []
    
    def calculate_center_of_mass(self, keypoints: List[List[float]]) -> np.ndarray:
        """
        Вычисляет центр тяжести (приблизительно - среднее координат бёдер)
        
        Args:
            keypoints: Координаты ключевых точек [[x1,y1], ...]
            
        Returns:
            Координаты центра тяжести [x, y]
        """
        if len(keypoints) < 17:
            return np.array([0.0, 0.0])
        
        # Центр тяжести ≈ среднее между левым и правым бедром (берём только x, y)
        left_hip = np.array(keypoints[11][:2])
        right_hip = np.array(keypoints[12][:2])
        
        center = (left_hip + right_hip) / 2.0
        return center
    
    def analyze_balance(self, poses: List[Dict]) -> Dict:
        """
        Анализирует баланс на протяжении всего видео
        
        Args:
            poses: Список поз из poses.jsonl
            
        Returns:
            Результаты анализа баланса
        """
        centers = []
        body_heights = []  # Для нормализации
        
        for pose in poses:
            if not pose.get('valid', False):
                continue
            
            keypoints = pose.get('keypoints', [])
            if len(keypoints) < 17:
                continue
            
            center = self.calculate_center_of_mass(keypoints)
            centers.append(center)
            
            # Вычисляем высоту тела для нормализации (от носа до бёдер)
            nose = np.array(keypoints[0][:2])
            hips = (np.array(keypoints[11][:2]) + np.array(keypoints[12][:2])) / 2.0
            body_height = abs(nose[1] - hips[1])
            body_heights.append(body_height)
        
        if len(centers) == 0:
            return {
                'score': 0,
                'status': 'no_data',
                'details': {}
            }
        
        self.center_of_mass_history = np.array(centers)
        avg_body_height = np.mean(body_heights)
        
        # ИСПРАВЛЕНО: Вместо отклонения от средней позиции всего видео,
        # измеряем локальную стабильность (изменение между соседними кадрами)
        # Это более правильно, т.к. человек может перемещаться по кадру
        
        frame_to_frame_movement = []
        for i in range(1, len(self.center_of_mass_history)):
            movement = np.linalg.norm(self.center_of_mass_history[i] - self.center_of_mass_history[i-1])
            frame_to_frame_movement.append(movement)
        
        deviations = np.array(frame_to_frame_movement)
        
        avg_deviation = np.mean(deviations)
        max_deviation = np.max(deviations)
        std_deviation = np.std(deviations)
        
        # ИСПРАВЛЕНО: Нормализуем относительно высоты тела
        # Используем более мягкую шкалу:
        # - Отклонение < 20% высоты тела = отлично (100)
        # - Отклонение 20-50% = хорошо (80-100)
        # - Отклонение 50-100% = удовлетворительно (50-80)
        # - Отклонение > 100% = плохо (0-50)
        
        if avg_body_height > 0:
            deviation_ratio = avg_deviation / avg_body_height
            
            # Используем сигмоиду для плавной оценки
            # score = 100 / (1 + exp(k * (deviation_ratio - threshold)))
            # При deviation_ratio = 0.5 (50%), score ≈ 50
            k = 8  # крутизна
            threshold = 0.5
            score = 100 / (1 + np.exp(k * (deviation_ratio - threshold)))
            
            stability = score / 100
            normalized_deviation = deviation_ratio
        else:
            score = 0
            stability = 0
            normalized_deviation = 0
        
        # Статус
        if score >= 90:
            status = 'excellent'
        elif score >= 75:
            status = 'good'
        elif score >= 60:
            status = 'fair'
        else:
            status = 'poor'
        
        # Проверка симметрии (левое/правое плечо на одной высоте)
        symmetry_scores = []
        
        for pose in poses:
            if not pose.get('valid', False):
                continue
            
            keypoints = pose.get('keypoints', [])
            if len(keypoints) < 17:
                continue
            
            left_shoulder = np.array(keypoints[5][:2])
            right_shoulder = np.array(keypoints[6][:2])
            
            # Разница по высоте (Y координата)
            height_diff = abs(left_shoulder[1] - right_shoulder[1])
            
            # Нормализуем (разница < 20 пикселей = хорошо)
            symmetry = 1 - min(height_diff / 20.0, 1.0)
            symmetry_scores.append(symmetry)
        
        avg_symmetry = np.mean(symmetry_scores) if symmetry_scores else 0
        
        return {
            'score': float(score),
            'status': status,
            'details': {
                'avg_deviation': float(avg_deviation),
                'max_deviation': float(max_deviation),
                'std_deviation': float(std_deviation),
                'stability': float(stability),
                'symmetry': float(avg_symmetry),
                'total_frames': len(centers),
                'avg_body_height': float(avg_body_height),
                'normalized_deviation': float(normalized_deviation)
            }
        }
    
    def find_balance_issues(
        self, 
        poses: List[Dict], 
        threshold: float = 2.5  # ИСПРАВЛЕНО: увеличен порог
    ) -> List[Dict]:
        """
        Находит моменты с проблемами баланса
        
        Args:
            poses: Список поз
            threshold: Порог в стандартных отклонениях
            
        Returns:
            Список ошибок с временными метками
        """
        if len(self.center_of_mass_history) == 0:
            self.analyze_balance(poses)
        
        if len(self.center_of_mass_history) == 0:
            return []
        
        mean_center = np.mean(self.center_of_mass_history, axis=0)
        std_dev = np.std(np.linalg.norm(self.center_of_mass_history - mean_center, axis=1))
        
        errors = []
        
        for frame_idx, center in enumerate(self.center_of_mass_history):
            deviation = np.linalg.norm(center - mean_center)
            z_score = deviation / std_dev if std_dev > 0 else 0
            
            if z_score > threshold:
                errors.append({
                    'frame': frame_idx,
                    'timestamp': frame_idx / 30.0,
                    'category': 'balance',
                    'severity': 'high' if z_score > 3.5 else 'medium',  # ИСПРАВЛЕНО: увеличен порог
                    'issue': 'Swaying (instability of center of gravity)',
                    'deviation': float(deviation),
                    'z_score': float(z_score)
                })
        
        return errors


def analyze_balance_from_file(poses_file: str) -> Dict:
    """
    Удобная функция для анализа баланса из файла
    
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
    
    analyzer = BalanceAnalyzer()
    result = analyzer.analyze_balance(poses)
    errors = analyzer.find_balance_issues(poses)
    
    return {
        'score': result,
        'errors': errors
    }

