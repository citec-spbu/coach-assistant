"""
Метрика тайминга и ритма: шаги относительно музыки
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np


@dataclass
class TimingMetricResult:
    bpm: float
    mean_abs_offset: float
    onbeat_ratio: float
    score: float
    # времена битов (сек)
    beat_times: np.ndarray
    # времена шагов (сек)
    step_times: np.ndarray
    # |dt| для каждого шага
    abs_offsets: np.ndarray
    # сегменты, где шаги сильно "мимо" бита
    error_segments: List[Dict]


# ---------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ----------

def _load_audio_and_beats(
    audio_path: Path,
) -> Tuple[float, np.ndarray]:
    """
    Загружаем аудио и находим биты.
    Возвращает:
        bpm, beat_times (сек).
    """
    import librosa
    import subprocess
    import shutil
    import tempfile
    import os
    import warnings

    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Файл аудио/видео не найден: {audio_path}")

    # Подавляем предупреждения librosa о PySoundFile и audioread
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
        warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
        
        # Пробуем загрузить напрямую через librosa
        try:
            y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        except Exception as e:
            # Если не получилось (проблема с audioread/ffmpeg), извлекаем аудио через subprocess
            ffmpeg_path = shutil.which('ffmpeg')
            if not ffmpeg_path:
                raise RuntimeError(
                    "ffmpeg не найден в PATH. "
                    "Установите ffmpeg и перезапустите терминал."
                ) from e
            
            # Извлекаем аудио во временный файл
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_audio_path = tmp_file.name
            
            try:
                # Сначала проверяем, есть ли аудио в видео
                probe_cmd = [
                    ffmpeg_path,
                    '-i', str(audio_path),
                    '-show_streams',
                    '-select_streams', 'a',  # только аудио потоки
                    '-loglevel', 'error'
                ]
                
                probe_result = subprocess.run(
                    probe_cmd,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                # Если нет аудио потоков, возвращаем ошибку
                if 'codec_type=audio' not in probe_result.stdout:
                    raise RuntimeError(
                        "В видео файле нет аудио потока. "
                        "Метрика Timing не может быть вычислена."
                    ) from e
                
                # Извлекаем аудио из видео через ffmpeg
                cmd = [
                    ffmpeg_path,
                    '-i', str(audio_path),
                    '-vn',  # без видео
                    '-acodec', 'pcm_s16le',  # PCM 16-bit
                    '-ar', '22050',  # частота дискретизации
                    '-ac', '1',  # моно
                    '-y',  # перезаписать выходной файл
                    tmp_audio_path
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    # Проверяем, может быть в видео просто нет аудио
                    if 'does not contain any stream' in result.stderr or 'no audio' in result.stderr.lower():
                        raise RuntimeError(
                            "В видео файле нет аудио потока. "
                            "Метрика Timing не может быть вычислена."
                        ) from e
                    raise RuntimeError(
                        f"Ошибка извлечения аудио через ffmpeg: {result.stderr}"
                    ) from e
                
                # Загружаем извлеченное аудио (подавляем предупреждения)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
                    warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
                    y, sr = librosa.load(tmp_audio_path, sr=None, mono=True)
            finally:
                # Удаляем временный файл
                if os.path.exists(tmp_audio_path):
                    try:
                        os.unlink(tmp_audio_path)
                    except:
                        pass
    
    # Вычисляем биты (подавляем предупреждения)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
        warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    return float(tempo), beat_times


def _extract_ankle_positions(poses_data: Dict) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Вынимает траектории лодыжек из poses_data.

    Предположение:
        poses_data['frames'] – список кадров,
        в каждом кадре есть либо 'pose_world_landmarks', либо 'pose_landmarks'
        – список landmark'ов длиной >= 29 (индексы 27, 28 – лодыжки).
    """
    frames = poses_data.get("frames", [])
    if not frames:
        raise ValueError("В poses_data нет frames")

    fps = float(poses_data.get("fps", 25.0))

    left_ankle = []
    right_ankle = []

    for f in frames:
        lm = f.get("pose_world_landmarks") or f.get("pose_landmarks")
        if lm is None or len(lm) < 29:
            # если не хватает landmark'ов – заполняем NaN
            left_ankle.append([np.nan, np.nan, np.nan])
            right_ankle.append([np.nan, np.nan, np.nan])
            continue

        la = lm[27]
        ra = lm[28]

        # формат landmark'ов может отличаться (dict или list/tuple)
        def _get_coord(point, key):
            if isinstance(point, dict):
                return float(point.get(key, np.nan))
            elif isinstance(point, (list, tuple)) and len(point) >= 3:
                idx = {"x": 0, "y": 1, "z": 2}[key]
                return float(point[idx])
            else:
                return np.nan

        left_ankle.append([
            _get_coord(la, "x"),
            _get_coord(la, "y"),
            _get_coord(la, "z"),
        ])
        right_ankle.append([
            _get_coord(ra, "x"),
            _get_coord(ra, "y"),
            _get_coord(ra, "z"),
        ])

    left_ankle = np.asarray(left_ankle, dtype=np.float32)
    right_ankle = np.asarray(right_ankle, dtype=np.float32)
    return left_ankle, right_ankle, fps


def _detect_steps_from_ankles(
    left_ankle: np.ndarray,
    right_ankle: np.ndarray,
    fps: float,
    speed_threshold: float = 0.01,
    min_interval: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Простейший детектор шагов:
    - считаем скорость каждой лодыжки в кадр/сек;
    - шаг = момент, когда скорость резко падает (нога "встала"),
      а до этого была заметная активность.

    Возвращает:
        step_frame_indices: np.ndarray[int],
        step_times: np.ndarray[float] (сек).
    """
    T = left_ankle.shape[0]
    if T < 3:
        return np.array([], dtype=int), np.array([], dtype=float)

    # скорость = разность координат * fps
    def _speed(traj: np.ndarray) -> np.ndarray:
        vel = np.diff(traj, axis=0) * fps
        speed = np.linalg.norm(vel, axis=1)
        # дополняем до длины T (добавим 0 в начало)
        return np.concatenate([[0.0], speed])

    speed_L = _speed(left_ankle)
    speed_R = _speed(right_ankle)

    # Немного сгладим
    def _smooth(x: np.ndarray, w: int = 3) -> np.ndarray:
        if len(x) < w:
            return x
        kernel = np.ones(w) / w
        return np.convolve(x, kernel, mode="same")

    speed_L = _smooth(speed_L)
    speed_R = _smooth(speed_R)

    # суммарная скорость ног
    speed_sum = speed_L + speed_R

    # шаг = локальный минимум скорости после локального максимума > порог
    candidates = []
    min_frames_between_steps = int(min_interval * fps)

    last_step_frame = -9999

    for t in range(1, T - 1):
        # локальный максимум скорости
        if speed_sum[t] > speed_sum[t - 1] and speed_sum[t] > speed_sum[t + 1] and speed_sum[t] > speed_threshold:
            # ищем ближайший последующий минимум
            # (момент, когда нога "остановилась" после маха)
            look_ahead = min(T - 1, t + int(0.3 * fps))  # ищем максимум 0.3 секунды вперёд
            min_t = t
            for k in range(t + 1, look_ahead):
                if speed_sum[k] < speed_sum[min_t]:
                    min_t = k
            # если минимум действительно "спокойный"
            if speed_sum[min_t] < speed_threshold and (min_t - last_step_frame) >= min_frames_between_steps:
                candidates.append(min_t)
                last_step_frame = min_t

    step_frame_indices = np.array(sorted(set(candidates)), dtype=int)
    step_times = step_frame_indices / fps
    return step_frame_indices, step_times


def _match_steps_to_beats(
    step_times: np.ndarray,
    beat_times: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Для каждого шага находит ближайший бит и возвращает:
        matched_beat_times, time_offsets (step_time - beat_time).
    """
    if len(step_times) == 0 or len(beat_times) == 0:
        return np.array([]), np.array([])

    matched_beats = []
    offsets = []

    for t in step_times:
        k = int(np.argmin(np.abs(beat_times - t)))
        bt = beat_times[k]
        matched_beats.append(bt)
        offsets.append(t - bt)

    return np.asarray(matched_beats, dtype=float), np.asarray(offsets, dtype=float)


# ---------- ОСНОВНАЯ ФУНКЦИЯ МЕТРИКИ ----------

def compute_timing_metric(
    poses_data: Dict,
    audio_path: Path,
    *,
    seq_frame_indices: Optional[np.ndarray] = None,
    bad_threshold: float = 0.20,   # > 200 мс – считаем заметным промахом
    good_threshold: float = 0.10,  # <= 100 мс – считаем "в такт"
    gamma: float = 4.0,           # насколько резко штрафуем большие ошибки
) -> TimingMetricResult:
    """
    Вычисляет метрику тайминга:
      - ищет биты по аудио,
      - детектирует шаги по позам,
      - сравнивает времена шагов с ближайшими битами.

    Args:
        poses_data: словарь с позами (как в predict_from_poses).
        audio_path: путь к видео/аудио файлу.
        seq_frame_indices: индексы кадров (в исходном видео), относящиеся к анализируемой фигуре.
                           Если передан, шаги будут отфильтрованы по этому диапазону.
        bad_threshold: порог (сек), после которого считаем шаг сильно промахнувшимся по времени.
        good_threshold: порог (сек), для подсчёта доли "on-beat" шагов.
        gamma: коэффициент для перевода ошибки в скор 0–100.

    Returns:
        TimingMetricResult
    """
    audio_path = Path(audio_path)

    # 1) Биты
    bpm, beat_times = _load_audio_and_beats(audio_path)

    # 2) Траектории лодыжек и шаги
    left_ankle, right_ankle, fps = _extract_ankle_positions(poses_data)
    step_frame_indices, step_times = _detect_steps_from_ankles(left_ankle, right_ankle, fps)

    # 3) Если передан диапазон кадров фигуры – фильтруем шаги по нему
    if seq_frame_indices is not None and len(seq_frame_indices) > 0:
        seq_frame_indices = np.asarray(seq_frame_indices, dtype=int)
        min_f = int(seq_frame_indices.min())
        max_f = int(seq_frame_indices.max())
        mask = (step_frame_indices >= min_f) & (step_frame_indices <= max_f)
        step_frame_indices = step_frame_indices[mask]
        step_times = step_times[mask]

    if len(step_times) == 0:
        # нет шагов – честно возвращаем "пустую" метрику
        return TimingMetricResult(
            bpm=bpm,
            mean_abs_offset=np.nan,
            onbeat_ratio=0.0,
            score=0.0,
            beat_times=beat_times,
            step_times=step_times,
            abs_offsets=np.array([], dtype=float),
            error_segments=[],
        )

    # 4) Матчинг шагов к битам
    _, offsets = _match_steps_to_beats(step_times, beat_times)
    abs_offsets = np.abs(offsets)

    mean_abs_offset = float(abs_offsets.mean())
    onbeat_ratio = float((abs_offsets <= good_threshold).mean())

    # 5) Перевод в скор 0–100: маленькая средняя ошибка и большая доля on-beat → большой скор
    # пример формулы:  score = onbeat_ratio * exp(-gamma * mean_abs_offset)
    score = float(onbeat_ratio * np.exp(-gamma * mean_abs_offset))
    score = float(np.clip(score * 100.0, 0.0, 100.0))

    # 6) Сегменты "плохого" тайминга
    error_segments: List[Dict] = []
    bad_mask = abs_offsets > bad_threshold

    in_seg = False
    seg_start = 0

    for i in range(len(step_times)):
        if bad_mask[i] and not in_seg:
            in_seg = True
            seg_start = i
        elif (not bad_mask[i]) and in_seg:
            in_seg = False
            seg_end = i - 1
            if seg_end >= seg_start:
                error_segments.append({
                    "start_step_idx": int(seg_start),
                    "end_step_idx": int(seg_end),
                    "start_time": float(step_times[seg_start]),
                    "end_time": float(step_times[seg_end]),
                    "mean_abs_offset": float(abs_offsets[seg_start:seg_end+1].mean()),
                })

    if in_seg:
        seg_end = len(step_times) - 1
        error_segments.append({
            "start_step_idx": int(seg_start),
            "end_step_idx": int(seg_end),
            "start_time": float(step_times[seg_start]),
            "end_time": float(step_times[seg_end]),
            "mean_abs_offset": float(abs_offsets[seg_start:seg_end+1].mean()),
        })

    return TimingMetricResult(
        bpm=bpm,
        mean_abs_offset=mean_abs_offset,
        onbeat_ratio=onbeat_ratio,
        score=score,
        beat_times=beat_times,
        step_times=step_times,
        abs_offsets=abs_offsets,
        error_segments=error_segments,
    )





