"""
Technique Analyzer - анализ техники исполнения через углы суставов
"""

import numpy as np
from typing import List, Dict, Tuple


class TechniqueAnalyzer:
    """Анализатор техники танца на основе углов суставов"""
    
    # Индексы ключевых точек YOLO Pose (17 точек)
    KEYPOINT_INDICES = {
        'nose': 0,
        'left_eye': 1, 'right_eye': 2,
        'left_ear': 3, 'right_ear': 4,
        'left_shoulder': 5, 'right_shoulder': 6,
        'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10,
        'left_hip': 11, 'right_hip': 12,
        'left_knee': 13, 'right_knee': 14,
        'left_ankle': 15, 'right_ankle': 16
    }
    
    def __init__(self):
        self.angles_history = []
    
    @staticmethod
    def calculate_angle(point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
        """
        Вычисляет угол в point2 между point1-point2-point3
        
        Args:
            point1: Первая точка [x, y]
            point2: Вершина угла [x, y]
            point3: Третья точка [x, y]
            
        Returns:
            Угол в градусах (0-180)
        """
        vector1 = np.array(point1) - np.array(point2)
        vector2 = np.array(point3) - np.array(point2)
        
        # Проверка на нулевые векторы
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cos_angle = np.dot(vector1, vector2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    def extract_angles_from_pose(self, keypoints: List[List[float]]) -> Dict[str, float]:
        """
        Извлекает все важные углы из одной позы
        
        Args:
            keypoints: Список координат [[x1,y1], [x2,y2], ...] - 17 точек
            
        Returns:
            Словарь с углами: {'left_elbow': 145.2, 'right_knee': 165.8, ...}
        """
        if len(keypoints) < 17:
            return {}
        
        angles = {}
        
        try:
            # Локти (плечо-локоть-запястье)
            angles['left_elbow'] = self.calculate_angle(
                keypoints[5], keypoints[7], keypoints[9]
            )
            angles['right_elbow'] = self.calculate_angle(
                keypoints[6], keypoints[8], keypoints[10]
            )
            
            # Колени (бедро-колено-лодыжка)
            angles['left_knee'] = self.calculate_angle(
                keypoints[11], keypoints[13], keypoints[15]
            )
            angles['right_knee'] = self.calculate_angle(
                keypoints[12], keypoints[14], keypoints[16]
            )
            
            # Бёдра (плечо-бедро-колено)
            angles['left_hip'] = self.calculate_angle(
                keypoints[5], keypoints[11], keypoints[13]
            )
            angles['right_hip'] = self.calculate_angle(
                keypoints[6], keypoints[12], keypoints[14]
            )
            
            # Плечи - подъём рук
            # Вычисляем угол между линией плеча-локоть и вертикалью
            # Для этого используем вектор плечо->локоть
            
            # Левое плечо
            shoulder_left = np.array(keypoints[5][:2])
            elbow_left = np.array(keypoints[7][:2])
            arm_vector_left = elbow_left - shoulder_left
            # Угол с вертикалью (вниз)
            vertical = np.array([0, 1])  # Y вниз
            if np.linalg.norm(arm_vector_left) > 0:
                cos_angle = np.dot(arm_vector_left, vertical) / np.linalg.norm(arm_vector_left)
                angles['left_shoulder'] = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
            else:
                angles['left_shoulder'] = 0.0
            
            # Правое плечо
            shoulder_right = np.array(keypoints[6][:2])
            elbow_right = np.array(keypoints[8][:2])
            arm_vector_right = elbow_right - shoulder_right
            if np.linalg.norm(arm_vector_right) > 0:
                cos_angle = np.dot(arm_vector_right, vertical) / np.linalg.norm(arm_vector_right)
                angles['right_shoulder'] = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
            else:
                angles['right_shoulder'] = 0.0
            
        except (IndexError, ValueError) as e:
            print(f"Warning: Error calculating angles: {e}")
            return {}
        
        return angles
    
    def extract_angles_from_video(self, poses: List[Dict]) -> List[Dict[str, float]]:
        """
        Извлекает углы из всех кадров видео
        
        Args:
            poses: Список поз из poses.jsonl
            
        Returns:
            Список словарей с углами для каждого кадра
        """
        all_angles = []
        
        for pose in poses:
            if not pose.get('valid', False):
                continue
            
            keypoints = pose.get('keypoints', [])
            if len(keypoints) < 17:
                continue
            
            angles = self.extract_angles_from_pose(keypoints)
            if angles:
                all_angles.append(angles)
        
        self.angles_history = all_angles
        return all_angles
    
    def compare_with_reference(
        self, 
        video_angles: List[Dict[str, float]], 
        reference_angles: Dict[str, Dict[str, float]],
        threshold: float = 2.5  # ИСПРАВЛЕНО: увеличен порог с 2.0 до 2.5
    ) -> List[Dict]:
        """
        Сравнивает углы видео с эталонными значениями
        
        Args:
            video_angles: Углы из видео (список словарей)
            reference_angles: Эталонные значения {'left_elbow': {'mean': 145, 'std': 15, ...}, ...}
            threshold: Порог в стандартных отклонениях (по умолчанию 2σ)
            
        Returns:
            Список ошибок с временными метками
        """
        errors = []
        
        for frame_idx, frame_angles in enumerate(video_angles):
            frame_errors = {}
            
            for joint, angle in frame_angles.items():
                if joint not in reference_angles:
                    continue
                
                ref = reference_angles[joint]
                
                # Z-score: сколько стандартных отклонений от среднего
                z_score = abs(angle - ref['mean']) / ref['std'] if ref['std'] > 0 else 0
                
                if z_score > threshold:
                    severity = 'high' if z_score > 3.5 else 'medium'  # ИСПРАВЛЕНО: увеличен порог high
                    
                    frame_errors[joint] = {
                        'current': float(angle),
                        'expected': float(ref['mean']),
                        'deviation': float(z_score),
                        'severity': severity,
                        'diff': float(angle - ref['mean'])
                    }
            
            if frame_errors:
                errors.append({
                    'frame': frame_idx,
                    'timestamp': frame_idx / 30.0,  # Предполагаем 30 FPS
                    'errors': frame_errors
                })
        
        return errors
    
    def calculate_technique_score(
        self, 
        video_angles: List[Dict[str, float]], 
        reference_angles: Dict[str, Dict[str, float]]
    ) -> Dict:
        """
        Вычисляет общую оценку техники (0-100)
        
        Args:
            video_angles: Углы из видео
            reference_angles: Эталонные значения
            
        Returns:
            Словарь с оценкой и деталями
        """
        if not video_angles or not reference_angles:
            return {'score': 0, 'status': 'error', 'details': {}}
        
        # Подсчёт средних отклонений для каждого сустава
        joint_scores = {}
        
        for joint in reference_angles.keys():
            deviations = []
            
            for frame_angles in video_angles:
                if joint in frame_angles:
                    ref = reference_angles[joint]
                    angle = frame_angles[joint]
                    
                    # Нормализованное отклонение (0-1)
                    z_score = abs(angle - ref['mean']) / ref['std'] if ref['std'] > 0 else 0
                    deviation = min(z_score / 3.0, 1.0)  # Ограничиваем до 1.0
                    deviations.append(deviation)
            
            if deviations:
                avg_deviation = np.mean(deviations)
                joint_score = max(0, 100 * (1 - avg_deviation))
                
                joint_scores[joint] = {
                    'score': float(joint_score),
                    'avg_deviation': float(avg_deviation)
                }
        
        # Общая оценка - среднее по всем суставам
        if joint_scores:
            overall_score = np.mean([js['score'] for js in joint_scores.values()])
        else:
            overall_score = 0
        
        # Статус
        if overall_score >= 90:
            status = 'excellent'
        elif overall_score >= 75:
            status = 'good'
        elif overall_score >= 60:
            status = 'fair'
        else:
            status = 'poor'
        
        return {
            'score': float(overall_score),
            'status': status,
            'details': joint_scores
        }


def analyze_technique(poses_file: str, reference_angles: Dict) -> Dict:
    """
    Удобная функция для анализа техники из файла poses.jsonl
    
    Args:
        poses_file: Путь к poses.jsonl
        reference_angles: Эталонные углы
        
    Returns:
        Результаты анализа
    """
    import json
    
    # Читаем позы
    poses = []
    with open(poses_file, 'r') as f:
        for line in f:
            poses.append(json.loads(line))
    
    # Анализируем
    analyzer = TechniqueAnalyzer()
    video_angles = analyzer.extract_angles_from_video(poses)
    
    score = analyzer.calculate_technique_score(video_angles, reference_angles)
    errors = analyzer.compare_with_reference(video_angles, reference_angles)
    
    return {
        'score': score,
        'errors': errors,
        'total_frames': len(video_angles),
        'error_frames': len(errors)
    }

