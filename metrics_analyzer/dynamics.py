"""
Dynamics Analyzer - анализ динамики и амплитуды движений
"""

import numpy as np
from typing import List, Dict


class DynamicsAnalyzer:
    """Анализатор динамики танца"""
    
    def __init__(self):
        self.amplitudes = []
        self.velocities = []
    
    def calculate_amplitude(self, poses: List[Dict]) -> np.ndarray:
        """
        Вычисляет амплитуду движений (размах)
        
        Args:
            poses: Список поз
            
        Returns:
            Амплитуды для каждого сустава
        """
        all_keypoints = []
        
        for pose in poses:
            if not pose.get('valid', False):
                continue
            
            keypoints = pose.get('keypoints', [])
            if len(keypoints) >= 17:
                all_keypoints.append(np.array(keypoints))
        
        if len(all_keypoints) == 0:
            return np.array([])
        
        all_keypoints = np.array(all_keypoints)
        
        # Амплитуда = макс. расстояние от средней позиции для каждой точки
        mean_positions = np.mean(all_keypoints, axis=0)
        
        amplitudes = []
        for keypoints in all_keypoints:
            distances = np.linalg.norm(keypoints - mean_positions, axis=1)
            max_distance = np.max(distances)
            amplitudes.append(max_distance)
        
        self.amplitudes = np.array(amplitudes)
        return self.amplitudes
    
    def calculate_energy(self, poses: List[Dict]) -> float:
        """
        Вычисляет энергию движения (сумма всех смещений)
        
        Args:
            poses: Список поз
            
        Returns:
            Значение энергии
        """
        total_movement = 0.0
        prev_keypoints = None
        
        for pose in poses:
            if not pose.get('valid', False):
                continue
            
            keypoints = np.array(pose.get('keypoints', []))
            
            if prev_keypoints is not None and len(keypoints) == len(prev_keypoints):
                movement = np.linalg.norm(keypoints - prev_keypoints, axis=1)
                total_movement += np.sum(movement)
            
            prev_keypoints = keypoints
        
        return float(total_movement)
    
    def calculate_contrast(self, velocities: np.ndarray) -> float:
        """
        Вычисляет контраст между быстрыми и медленными движениями
        
        Args:
            velocities: Скорости движения
            
        Returns:
            Значение контраста (0-1)
        """
        if len(velocities) == 0:
            return 0.0
        
        # Стандартное отклонение скоростей = насколько разнообразна динамика
        std_velocity = np.std(velocities)
        mean_velocity = np.mean(velocities)
        
        # Нормализованный контраст
        contrast = std_velocity / (mean_velocity + 1e-6)
        
        # Ограничиваем до 1.0
        contrast = min(contrast, 1.0)
        
        return float(contrast)
    
    def analyze_dynamics(self, poses: List[Dict]) -> Dict:
        """
        Полный анализ динамики
        
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
        
        # Амплитуда
        amplitudes = self.calculate_amplitude(poses)
        avg_amplitude = np.mean(amplitudes) if len(amplitudes) > 0 else 0
        
        # Энергия
        energy = self.calculate_energy(poses)
        
        # Вычисляем скорости для контраста
        velocities = []
        prev_pose = None
        
        for pose in poses:
            if not pose.get('valid', False):
                if prev_pose is not None:
                    velocities.append(0.0)
                continue
            
            keypoints = np.array(pose.get('keypoints', []))
            
            if prev_pose is not None:
                prev_keypoints = np.array(prev_pose.get('keypoints', []))
                
                if len(keypoints) == len(prev_keypoints):
                    movement = np.linalg.norm(keypoints - prev_keypoints, axis=1)
                    avg_velocity = np.mean(movement)
                    velocities.append(float(avg_velocity))
            
            prev_pose = pose
        
        self.velocities = np.array(velocities)
        
        # Контраст
        contrast = self.calculate_contrast(self.velocities)
        
        # Оценка динамики (комбинация амплитуды, энергии и контраста)
        # Нормализуем амплитуду (200 пикселей = хорошо)
        amplitude_score = min(avg_amplitude / 200.0, 1.0) * 100
        
        # Нормализуем энергию (зависит от длины видео)
        energy_per_frame = energy / len(poses)
        energy_score = min(energy_per_frame / 50.0, 1.0) * 100
        
        # Контраст уже нормализован (0-1)
        contrast_score = contrast * 100
        
        # Общая оценка (взвешенное среднее)
        score = 0.4 * amplitude_score + 0.3 * energy_score + 0.3 * contrast_score
        
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
                'avg_amplitude': float(avg_amplitude),
                'energy': float(energy),
                'energy_per_frame': float(energy_per_frame),
                'contrast': float(contrast),
                'amplitude_score': float(amplitude_score),
                'energy_score': float(energy_score),
                'contrast_score': float(contrast_score)
            }
        }


def analyze_dynamics_from_file(poses_file: str) -> Dict:
    """
    Удобная функция для анализа динамики из файла
    
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
    
    analyzer = DynamicsAnalyzer()
    result = analyzer.analyze_dynamics(poses)
    
    return result









