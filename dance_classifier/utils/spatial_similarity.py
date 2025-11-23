"""
Метрика пространственно-временной похожести фигуры
(сравнение с эталонной траекторией при помощи DTW).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np


@dataclass
class SpatialSimilarityResult:
    # Средняя дистанция вдоль оптимального DTW-пути
    mean_distance: float
    # Нормализованный скор 0–100 (чем больше, тем лучше)
    score: float
    # Локальные расстояния для каждого кадра sequence (после выравнивания)
    per_frame_distances: np.ndarray
    # Интервалы ошибок (по индексам кадров и, опционально, времени)
    error_segments: List[Dict]


def _dtw_distance_with_path(
    seq_a: np.ndarray,
    seq_b: np.ndarray
) -> Tuple[float, List[Tuple[int, int, float]]]:
    """
    Простейшая реализация DTW c евклидовой метрикой и восстановлением пути.

    Args:
        seq_a: numpy array (T1, D)
        seq_b: numpy array (T2, D)

    Returns:
        total_cost: полная стоимость оптимального пути
        path: список (i, j, dist_ij) по пути, отсортированный по времени
    """
    if seq_a.ndim != 2 or seq_b.ndim != 2:
        raise ValueError("seq_a и seq_b должны иметь форму (T, D)")

    T1, D1 = seq_a.shape
    T2, D2 = seq_b.shape
    if D1 != D2:
        raise ValueError(f"Несовпадение размерностей признаков: {D1} != {D2}")

    # Матрица кумулятивных стоимостей
    cost = np.full((T1 + 1, T2 + 1), np.inf, dtype=np.float32)
    cost[0, 0] = 0.0

    # Попарные расстояния
    dist_mat = np.zeros((T1, T2), dtype=np.float32)
    for i in range(T1):
        for j in range(T2):
            dist_mat[i, j] = np.linalg.norm(seq_a[i] - seq_b[j])

    # DP
    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            d = dist_mat[i - 1, j - 1]
            cost[i, j] = d + min(
                cost[i - 1, j],      # шаг по seq_a
                cost[i, j - 1],      # шаг по seq_b
                cost[i - 1, j - 1],  # диагональ
            )

    total_cost = float(cost[T1, T2])

    # Восстановление пути (backtracking)
    path: List[Tuple[int, int, float]] = []
    i, j = T1, T2
    while i > 0 and j > 0:
        d = dist_mat[i - 1, j - 1]
        path.append((i - 1, j - 1, float(d)))

        prev_costs = (
            cost[i - 1, j],      # вверх (по seq_a)
            cost[i, j - 1],      # влево (по seq_b)
            cost[i - 1, j - 1],  # диагональ
        )
        argmin = int(np.argmin(prev_costs))
        if argmin == 0:
            i -= 1
        elif argmin == 1:
            j -= 1
        else:
            i -= 1
            j -= 1

    path.reverse()
    return total_cost, path


def _per_frame_distances_from_path(
    path: List[Tuple[int, int, float]],
    T1: int
) -> np.ndarray:
    """
    Строит массив средних локальных расстояний для каждого кадра seq_a.
    """
    per_frame = np.zeros(T1, dtype=np.float32)
    counts = np.zeros(T1, dtype=np.int32)

    for i, j, d in path:
        per_frame[i] += d
        counts[i] += 1

    counts[counts == 0] = 1
    per_frame /= counts.astype(np.float32)
    return per_frame


def compute_spatial_similarity(
    sequence: np.ndarray,
    reference_sequence: np.ndarray,
    *,
    alpha: float = 3.0,
    error_z: float = 1.5,
    frame_indices: Optional[np.ndarray] = None,
    frame_timestamps: Optional[np.ndarray] = None,
) -> SpatialSimilarityResult:
    """
    Считает метрику похожести траектории на эталон.

    Args:
        sequence: (T, D) – последовательность признаков конкретного исполнения
                  (желательно НОРМАЛИЗОВАННАЯ через scaler!)
        reference_sequence: (T_ref, D) – эталонная траектория для этой фигуры
        alpha: коэффициент для перевода distance -> score (0–100)
        error_z: порог в σ для выделения "ошибочных" кадров
        frame_indices: (T,) индексы кадров в исходном видео (опционально)
        frame_timestamps: (T,) временные метки (сек), соответствующие кадрам sequence (опционально)

    Returns:
        SpatialSimilarityResult с глобальной оценкой и интервалами ошибок.
    """
    if sequence.ndim != 2:
        raise ValueError("sequence должен иметь форму (T, D)")
    if reference_sequence.ndim != 2:
        raise ValueError("reference_sequence должен иметь форму (T_ref, D)")

    # DTW
    total_cost, path = _dtw_distance_with_path(sequence, reference_sequence)
    path_len = max(len(path), 1)
    mean_distance = total_cost / path_len

    # Локальные расстояния по кадрам
    T, _ = sequence.shape
    per_frame = _per_frame_distances_from_path(path, T)

    # Перевод в скор 0–100
    score = float(100.0 * np.exp(-alpha * float(mean_distance)))
    score = float(np.clip(score, 0.0, 100.0))

    # Выделение "плохих" кадров
    mu = float(per_frame.mean())
    sigma = float(per_frame.std()) if per_frame.size > 1 else 0.0
    if sigma == 0.0:
        threshold = mu
    else:
        threshold = mu + error_z * sigma

    error_segments: List[Dict] = []
    in_segment = False
    seg_start_idx = 0

    for t in range(T):
        if per_frame[t] > threshold and not in_segment:
            in_segment = True
            seg_start_idx = t
        elif per_frame[t] <= threshold and in_segment:
            in_segment = False
            seg_end_idx = t - 1
            if seg_end_idx >= seg_start_idx:
                segment = {
                    "start_frame_idx": int(seg_start_idx),
                    "end_frame_idx": int(seg_end_idx),
                    "mean_distance": float(per_frame[seg_start_idx:seg_end_idx + 1].mean()),
                }
                if frame_indices is not None and len(frame_indices) == T:
                    segment["start_video_frame"] = int(frame_indices[seg_start_idx])
                    segment["end_video_frame"] = int(frame_indices[seg_end_idx])
                if frame_timestamps is not None and len(frame_timestamps) == T:
                    segment["start_time"] = float(frame_timestamps[seg_start_idx])
                    segment["end_time"] = float(frame_timestamps[seg_end_idx])
                error_segments.append(segment)

    # Хвостовой сегмент
    if in_segment:
        seg_end_idx = T - 1
        segment = {
            "start_frame_idx": int(seg_start_idx),
            "end_frame_idx": int(seg_end_idx),
            "mean_distance": float(per_frame[seg_start_idx:seg_end_idx + 1].mean()),
        }
        if frame_indices is not None and len(frame_indices) == T:
            segment["start_video_frame"] = int(frame_indices[seg_start_idx])
            segment["end_video_frame"] = int(frame_indices[seg_end_idx])
        if frame_timestamps is not None and len(frame_timestamps) == T:
            segment["start_time"] = float(frame_timestamps[seg_start_idx])
            segment["end_time"] = float(frame_timestamps[seg_end_idx])
        error_segments.append(segment)

    return SpatialSimilarityResult(
        mean_distance=float(mean_distance),
        score=score,
        per_frame_distances=per_frame,
        error_segments=error_segments,
    )


def build_reference_from_examples(
    sequences: List[np.ndarray],
    target_length: Optional[int] = None,
) -> np.ndarray:
    """
    Строит усреднённую эталонную траекторию из набора "правильных" исполнений.

    Args:
        sequences: список массивов (T_i, D) в ОДНОМ и том же признаковом пространстве
                   (желательно уже нормализованных scaler'ом)
        target_length: желаемая длина эталона. Если None – берётся
                       средняя длина по выборке и все последовательности
                       линейно ресэмплируются до этого значения.

    Returns:
        numpy array (target_length, D): эталонная траектория
    """
    if len(sequences) == 0:
        raise ValueError("sequences пустой – нечего усреднять")

    if target_length is None:
        lengths = [s.shape[0] for s in sequences]
        target_length = int(np.round(np.mean(lengths)))

    D = sequences[0].shape[1]
    ref = np.zeros((target_length, D), dtype=np.float32)

    for seq in sequences:
        if seq.ndim != 2 or seq.shape[1] != D:
            raise ValueError("Все последовательности должны иметь форму (T, D) с одинаковым D")

        T = seq.shape[0]
        src_t = np.linspace(0.0, 1.0, T)
        dst_t = np.linspace(0.0, 1.0, target_length)
        seq_resampled = np.zeros((target_length, D), dtype=np.float32)
        for d in range(D):
            seq_resampled[:, d] = np.interp(dst_t, src_t, seq[:, d])

        ref += seq_resampled

    ref /= float(len(sequences))
    return ref





