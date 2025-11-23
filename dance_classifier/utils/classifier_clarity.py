"""
Метрика "разборчивости" фигуры по выходам классификатора
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np


@dataclass
class ClassifierClarityResult:
    """
    Результат метрики "разборчивости" фигуры по выходам классификатора.
    """
    mean_confidence: float
    mean_margin: float
    mean_entropy: float
    score: float
    per_window_confidence: np.ndarray
    per_window_margin: np.ndarray
    per_window_entropy: np.ndarray
    window_centers: np.ndarray
    error_segments: List[Dict]


def compute_classifier_clarity(
    probabilities_over_time: np.ndarray,
    target_class_idx: int,
    *,
    window_centers: Optional[np.ndarray] = None,
    seq_frame_indices: Optional[np.ndarray] = None,
    seq_timestamps: Optional[np.ndarray] = None,
    low_conf_threshold: float = 0.6,
    low_margin_threshold: float = 0.2,
    entropy_z: float = 1.0,
) -> ClassifierClarityResult:
    """
    Метрика качества/разборчивости фигуры на основе распределений
    вероятностей по окнам последовательности.

    Args:
        probabilities_over_time: np.ndarray формы (W, C),
            где W – число окон, C – число классов.
        target_class_idx: индекс целевого класса (предсказанного фигурой).
        window_centers: np.ndarray формы (W,), индексы кадров
            (внутри sequence), соответствующие центру каждого окна.
        seq_frame_indices: np.ndarray формы (T,), индексы кадров исходного
            видео, соответствующие кадрам sequence (опционально).
        seq_timestamps: np.ndarray формы (T,), временные метки (секунды)
            для каждого кадра sequence (опционально).
        low_conf_threshold: порог "низкой уверенности" p(class).
        low_margin_threshold: порог "малого отрыва" от второго по величине класса.
        entropy_z: множитель для порога по энтропии (mu + z * sigma).

    Returns:
        ClassifierClarityResult
    """
    if probabilities_over_time.ndim != 2:
        raise ValueError("probabilities_over_time должен иметь форму (W, C)")

    W, C = probabilities_over_time.shape
    if not (0 <= target_class_idx < C):
        raise ValueError(f"Некорректный target_class_idx={target_class_idx} для C={C}")

    # Нормируем на всякий случай
    probs = probabilities_over_time.astype(np.float32)
    probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)

    # Уверенность в целевом классе
    conf = probs[:, target_class_idx]  # (W,)

    # Отрыв от ближайшего конкурента
    other_mask = np.ones(C, dtype=bool)
    other_mask[target_class_idx] = False
    best_other = probs[:, other_mask].max(axis=1)
    margin = conf - best_other  # (W,)

    # Энтропия
    entropy = -(probs * np.log(probs + 1e-8)).sum(axis=1)  # (W,)
    H_max = float(np.log(C))

    mean_conf = float(conf.mean())
    mean_margin = float(margin.mean())
    mean_entropy = float(entropy.mean())

    # Сводим в скор 0–100:
    #  - хотим высокую уверенность и низкую энтропию
    #  - обе компоненты в [0, 1]
    conf_component = mean_conf  # уже в [0, 1]
    entropy_component = 1.0 - (mean_entropy / (H_max + 1e-8))
    entropy_component = float(np.clip(entropy_component, 0.0, 1.0))

    combined = 0.7 * conf_component + 0.3 * entropy_component
    score = float(np.clip(combined * 100.0, 0.0, 100.0))

    # ==== Поиск "проблемных" окон и сбор сегментов ====
    # Порог по энтропии
    mu_e = float(entropy.mean())
    sigma_e = float(entropy.std()) if W > 1 else 0.0
    if sigma_e == 0.0:
        entropy_thr = mu_e + 0.0
    else:
        entropy_thr = mu_e + entropy_z * sigma_e

    is_bad = (conf < low_conf_threshold) | (margin < low_margin_threshold) | (entropy > entropy_thr)

    error_segments: List[Dict] = []
    in_segment = False
    seg_start = 0

    # Если нет window_centers – просто используем индексы окон как "оси времени" внутри sequence.
    if window_centers is None:
        window_centers = np.arange(W, dtype=int)
    else:
        window_centers = window_centers.astype(int)

    for w in range(W):
        if is_bad[w] and not in_segment:
            in_segment = True
            seg_start = w
        elif (not is_bad[w]) and in_segment:
            in_segment = False
            seg_end = w - 1
            if seg_end >= seg_start:
                seg = _build_segment_dict(
                    seg_start,
                    seg_end,
                    conf,
                    margin,
                    entropy,
                    window_centers,
                    seq_frame_indices,
                    seq_timestamps,
                )
                error_segments.append(seg)

    if in_segment:
        seg_end = W - 1
        seg = _build_segment_dict(
            seg_start,
            seg_end,
            conf,
            margin,
            entropy,
            window_centers,
            seq_frame_indices,
            seq_timestamps,
        )
        error_segments.append(seg)

    return ClassifierClarityResult(
        mean_confidence=mean_conf,
        mean_margin=mean_margin,
        mean_entropy=mean_entropy,
        score=score,
        per_window_confidence=conf,
        per_window_margin=margin,
        per_window_entropy=entropy,
        window_centers=window_centers,
        error_segments=error_segments,
    )


def _build_segment_dict(
    seg_start: int,
    seg_end: int,
    conf: np.ndarray,
    margin: np.ndarray,
    entropy: np.ndarray,
    window_centers: np.ndarray,
    seq_frame_indices: Optional[np.ndarray],
    seq_timestamps: Optional[np.ndarray],
) -> Dict:
    """
    Собирает словарь с информацией об одном "плохом" сегменте.
    """
    seg_slice = slice(seg_start, seg_end + 1)
    centers = window_centers[seg_slice]

    seg_dict: Dict = {
        "start_window_idx": int(seg_start),
        "end_window_idx": int(seg_end),
        "start_seq_idx": int(centers.min()),
        "end_seq_idx": int(centers.max()),
        "mean_confidence": float(conf[seg_slice].mean()),
        "mean_margin": float(margin[seg_slice].mean()),
        "mean_entropy": float(entropy[seg_slice].mean()),
    }

    if seq_frame_indices is not None and len(seq_frame_indices) > 0:
        T = len(seq_frame_indices)
        start_seq_idx = int(np.clip(seg_dict["start_seq_idx"], 0, T - 1))
        end_seq_idx = int(np.clip(seg_dict["end_seq_idx"], 0, T - 1))
        seg_dict["start_video_frame"] = int(seq_frame_indices[start_seq_idx])
        seg_dict["end_video_frame"] = int(seq_frame_indices[end_seq_idx])

    if seq_timestamps is not None and len(seq_timestamps) > 0:
        T = len(seq_timestamps)
        start_seq_idx = int(np.clip(seg_dict["start_seq_idx"], 0, T - 1))
        end_seq_idx = int(np.clip(seg_dict["end_seq_idx"], 0, T - 1))
        seg_dict["start_time"] = float(seq_timestamps[start_seq_idx])
        seg_dict["end_time"] = float(seq_timestamps[end_seq_idx])

    return seg_dict





