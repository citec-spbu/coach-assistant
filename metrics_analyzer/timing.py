"""
Timing Analyzer - анализ попадания движений в музыкальный ритм
"""

import numpy as np
from typing import List, Dict, Optional
from scipy.signal import find_peaks


class TimingAnalyzer:
    """Анализатор тайминга и попадания в музыкальный ритм"""
    
    def __init__(self, fps: int = 30):
        self.fps = fps
        self.velocities = []
        self.peak_times = []
    
    def calculate_movement_velocity(self, poses: List[Dict]) -> np.ndarray:
        """
        Вычисляет скорость движения для каждого кадра
        
        Args:
            poses: Список поз из poses.jsonl
            
        Returns:
            Массив со скоростями движения
        """
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
                    # Среднее смещение всех точек
                    movement = np.linalg.norm(keypoints - prev_keypoints, axis=1)
                    avg_velocity = np.mean(movement)
                    velocities.append(float(avg_velocity))
                else:
                    velocities.append(0.0)
            
            prev_pose = pose
        
        self.velocities = np.array(velocities)
        return self.velocities
    
    def find_movement_peaks(self, velocities: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Находит пики движения (резкие/сильные движения)
        
        Args:
            velocities: Скорости движений (если None, используются ранее вычисленные)
            
        Returns:
            Массив индексов пиков
        """
        if velocities is None:
            velocities = self.velocities
        
        if len(velocities) == 0:
            return np.array([])
        
        # Находим пики выше 80-го перцентиля
        threshold = np.percentile(velocities, 80)
        peaks, _ = find_peaks(velocities, height=threshold, distance=5)
        
        # Конвертируем в временные метки
        self.peak_times = peaks / self.fps
        
        return peaks
    
    def analyze_with_music(
        self, 
        poses: List[Dict], 
        music_info: Dict,
        tolerance: float = 0.2
    ) -> Dict:
        """
        Анализ попадания движений в музыкальные биты
        
        Args:
            poses: Список поз
            music_info: Информация о музыке {'bpm': 128, 'beats': [0.5, 1.0, ...]}
            tolerance: Допустимое отклонение в секундах (±0.2 сек по умолчанию)
            
        Returns:
            Результаты анализа тайминга
        """
        # Вычисляем скорости движения
        velocities = self.calculate_movement_velocity(poses)
        
        # Находим пики движения
        peaks = self.find_movement_peaks(velocities)
        peak_times = peaks / self.fps
        
        if len(peak_times) == 0:
            return {
                'score': 0,
                'status': 'no_peaks',
                'details': {
                    'total_peaks': 0,
                    'matched_peaks': 0,
                    'bpm': music_info.get('bpm', 0)
                }
            }
        
        # Проверяем попадание в биты
        beats = np.array(music_info.get('beats', []))
        
        if len(beats) == 0:
            # Если битов нет, оцениваем только по постоянству темпа
            return self._analyze_tempo_consistency(velocities, music_info.get('bpm', 120))
        
        matched = 0
        mismatches = []
        
        for peak_time in peak_times:
            # Находим ближайший бит
            if len(beats) > 0:
                closest_beat_idx = np.argmin(np.abs(beats - peak_time))
                closest_beat = beats[closest_beat_idx]
                
                if abs(peak_time - closest_beat) < tolerance:
                    matched += 1
                else:
                    mismatches.append({
                        'timestamp': float(peak_time),
                        'expected_beat': float(closest_beat),
                        'lag': float(peak_time - closest_beat)
                    })
        
        # Процент попадания
        timing_score = (matched / len(peak_times)) * 100 if len(peak_times) > 0 else 0
        
        # Статус
        if timing_score >= 90:
            status = 'excellent'
        elif timing_score >= 75:
            status = 'good'
        elif timing_score >= 60:
            status = 'fair'
        else:
            status = 'poor'
        
        return {
            'score': float(timing_score),
            'status': status,
            'details': {
                'total_peaks': int(len(peak_times)),
                'matched_peaks': int(matched),
                'bpm': float(music_info.get('bpm', 0)),
                'mismatches': mismatches[:5]  # Топ-5 ошибок
            }
        }
    
    def _analyze_tempo_consistency(self, velocities: np.ndarray, target_bpm: float) -> Dict:
        """
        Анализ постоянства темпа (если нет музыкальных битов)
        
        Args:
            velocities: Скорости движения
            target_bpm: Целевой BPM
            
        Returns:
            Результаты анализа
        """
        # Находим периодичность движений
        peaks = self.find_movement_peaks(velocities)
        
        if len(peaks) < 2:
            return {
                'score': 0,
                'status': 'insufficient_data',
                'details': {'total_peaks': len(peaks)}
            }
        
        # Вычисляем интервалы между пиками
        intervals = np.diff(peaks) / self.fps
        
        # Ожидаемый интервал из BPM
        expected_interval = 60.0 / target_bpm
        
        # Отклонения от ожидаемого интервала
        deviations = np.abs(intervals - expected_interval) / expected_interval
        avg_deviation = np.mean(deviations)
        
        # Оценка (0-100)
        score = max(0, 100 * (1 - avg_deviation))
        
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
                'total_peaks': int(len(peaks)),
                'avg_interval': float(np.mean(intervals)),
                'expected_interval': float(expected_interval),
                'avg_deviation': float(avg_deviation),
                'bpm': float(target_bpm)
            }
        }
    
    def analyze_without_music(self, poses: List[Dict]) -> Dict:
        """
        Анализ тайминга без музыки (только плавность и темп)
        
        Args:
            poses: Список поз
            
        Returns:
            Результаты анализа
        """
        velocities = self.calculate_movement_velocity(poses)
        
        if len(velocities) == 0:
            return {
                'score': 0,
                'status': 'no_data',
                'details': {}
            }
        
        # Вычисляем ускорение (плавность)
        acceleration = np.diff(velocities)
        
        # Резкие рывки = высокое ускорение
        jerks = np.abs(acceleration)
        avg_jerk = np.mean(jerks)
        max_jerk = np.max(jerks)
        
        # Оценка плавности (меньше рывков = лучше)
        # Нормализуем относительно средней скорости
        avg_velocity = np.mean(velocities)
        smoothness = 1 - min(avg_jerk / (avg_velocity + 1e-6), 1.0)
        
        score = smoothness * 100
        
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
                'avg_velocity': float(avg_velocity),
                'avg_jerk': float(avg_jerk),
                'max_jerk': float(max_jerk),
                'smoothness': float(smoothness)
            }
        }


def extract_audio_and_analyze(video_path: str, poses: List[Dict]) -> Dict:
    """
    Извлекает аудио из видео и анализирует тайминг
    
    Args:
        video_path: Путь к видео
        poses: Список поз
        
    Returns:
        Результаты анализа тайминга
    """
    try:
        from moviepy.editor import VideoFileClip
        import librosa
        import tempfile
        import os
        
        # Извлекаем аудио
        clip = VideoFileClip(video_path)
        
        if clip.audio is None:
            print("Warning: No audio in video, analyzing without music")
            analyzer = TimingAnalyzer()
            return analyzer.analyze_without_music(poses)
        
        # Сохраняем аудио во временный файл
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_path = tmp.name
            clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
        
        clip.close()
        
        # Анализируем музыку
        y, sr = librosa.load(audio_path)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        music_info = {
            'bpm': float(tempo),
            'beats': beat_times.tolist(),
            'duration': len(y) / sr
        }
        
        # Удаляем временный файл
        os.unlink(audio_path)
        
        # Анализируем тайминг
        analyzer = TimingAnalyzer()
        result = analyzer.analyze_with_music(poses, music_info)
        
        return result
        
    except ImportError:
        print("Warning: moviepy or librosa not installed, analyzing without music")
        analyzer = TimingAnalyzer()
        return analyzer.analyze_without_music(poses)
    
    except Exception as e:
        print(f"Warning: Error analyzing audio: {e}, analyzing without music")
        analyzer = TimingAnalyzer()
        return analyzer.analyze_without_music(poses)









