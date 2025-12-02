"""
Метрика баланса и стабильности корпуса
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import math


@dataclass
class BalanceMetricResult:
    """
    Результат метрики баланса и стабильности.
    """
    mean_tilt_deg: float                 # средний наклон корпуса (градусы)
    tilt_std_deg: float                  # стандартное отклонение наклона
    com_std_norm: float                  # std бокового смещения CoM в "ширинах таза"
    out_of_corridor_ratio: float         # доля кадров сильно вне "коридора"
    score: float                         # интегральный скор 0–100
    per_frame_tilt_deg: np.ndarray       # наклон по кадрам (градусы)
    per_frame_com_offset_norm: np.ndarray  # нормированное боковое смещение CoM
    timestamps: np.ndarray               # времена кадров (сек)
    error_segments: List[Dict]           # интервал(ы), где баланс явного плохой


def _get_landmarks_from_frame(frame: Dict) -> Optional[List[Dict]]:
    """
    Возвращает список landmark'ов для кадра:
    либо из pose_world_landmarks, либо из pose_landmarks.
    """
    lm = frame.get("pose_world_landmarks")
    if lm is None:
        lm = frame.get("pose_landmarks")

    if lm is None or len(lm) < 29:
        return None
    return lm


def _get_coord(point, key: str) -> float:
    """
    Достаёт координату (x/y/z) из landmark'а,
    который может быть dict или list/tuple.
    """
    if isinstance(point, dict):
        return float(point.get(key, np.nan))
    elif isinstance(point, (list, tuple)) and len(point) >= 3:
        idx_map = {"x": 0, "y": 1, "z": 2}
        idx = idx_map[key]
        return float(point[idx])
    else:
        return float("nan")


def _compute_body_params_for_indices(
    frames: List[Dict],
    seq_frame_indices: np.ndarray,
    fps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Для заданных индексов кадров считает:
      - наклон корпуса (градусы),
      - боковое смещение центра масс (нормировано на ширину таза),
      - timestamps (сек).

    Возвращает:
        tilt_deg: (T,) – наклон,
        com_offset_norm: (T,) – нормированное смещение,
        timestamps: (T,) – время кадра в сек.
    """
    tilt_list = []
    com_offset_norm_list = []
    ts_list = []

    for idx in seq_frame_indices:
        if idx < 0 or idx >= len(frames):
            continue

        f = frames[idx]
        lm = _get_landmarks_from_frame(f)
        if lm is None:
            continue

        # ключевые точки
        # плечи
        l_sh = lm[11]
        r_sh = lm[12]
        # бёдра/таз
        l_hip = lm[23]
        r_hip = lm[24]
        # лодыжки
        l_ank = lm[27]
        r_ank = lm[28]

        sh_cx = (_get_coord(l_sh, "x") + _get_coord(r_sh, "x")) * 0.5
        sh_cy = (_get_coord(l_sh, "y") + _get_coord(r_sh, "y")) * 0.5

        hip_cx = (_get_coord(l_hip, "x") + _get_coord(r_hip, "x")) * 0.5
        hip_cy = (_get_coord(l_hip, "y") + _get_coord(r_hip, "y")) * 0.5

        ank_cx = (_get_coord(l_ank, "x") + _get_coord(r_ank, "x")) * 0.5

        # ширина таза – для нормировки
        hip_w = abs(_get_coord(l_hip, "x") - _get_coord(r_hip, "x"))

        # проверяем NaN
        if any(math.isnan(v) for v in [sh_cx, sh_cy, hip_cx, hip_cy, ank_cx]) or math.isnan(hip_w) or hip_w == 0:
            continue

        # Вектор "таз -> плечи"
        vx = sh_cx - hip_cx
        vy = sh_cy - hip_cy

        # В Mediapipe y обычно растёт вниз; нам важен угол к вертикали, берём abs
        # угол между (vx, vy) и вертикалью (0,1): можно взять arctan(|vx| / |vy|)
        tilt_rad = math.atan2(abs(vx), abs(vy) + 1e-8)
        tilt_deg = math.degrees(tilt_rad)

        # Приближённый центр масс
        com_x = 0.6 * hip_cx + 0.4 * sh_cx  # немного больше веса на таз
        # Боковое смещение относительно середины стоп
        com_offset = com_x - ank_cx
        com_offset_norm = com_offset / hip_w  # в "ширинах таза"

        tilt_list.append(tilt_deg)
        com_offset_norm_list.append(com_offset_norm)

        ts = f.get("timestamp")
        if ts is None:
            ts = idx / fps
        ts_list.append(float(ts))

    if not tilt_list:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        )

    return (
        np.asarray(tilt_list, dtype=np.float32),
        np.asarray(com_offset_norm_list, dtype=np.float32),
        np.asarray(ts_list, dtype=np.float32),
    )


def compute_balance_metric(
    poses_data: Dict,
    seq_frame_indices: np.ndarray,
    *,
    tilt_limit_deg: float = 15.0,
    com_limit: float = 0.4,
    alpha: float = 1.5,
    beta: float = 1.0,
) -> BalanceMetricResult:
    """
    Основная функция метрики баланса и стабильности.

    Args:
        poses_data: словарь с позами (как в predict_from_poses).
        seq_frame_indices: индексы кадров исходного видео, относящиеся к анализируемой фигуре.
        tilt_limit_deg: порог "слишком большого" наклона корпуса (градусы).
        com_limit: допустимое нормированное смещение CoM (в ширинах таза).
        alpha, beta: коэффициенты для перевода ошибок в скор.

    Returns:
        BalanceMetricResult
    """
    frames = poses_data.get("frames", [])
    if not frames:
        return BalanceMetricResult(
            mean_tilt_deg=float("nan"),
            tilt_std_deg=float("nan"),
            com_std_norm=float("nan"),
            out_of_corridor_ratio=0.0,
            score=0.0,
            per_frame_tilt_deg=np.array([], dtype=np.float32),
            per_frame_com_offset_norm=np.array([], dtype=np.float32),
            timestamps=np.array([], dtype=np.float32),
            error_segments=[],
        )

    fps = float(poses_data.get("fps", 25.0))
    seq_frame_indices = np.asarray(seq_frame_indices, dtype=int)

    tilt_deg, com_offset_norm, timestamps = _compute_body_params_for_indices(
        frames, seq_frame_indices, fps
    )

    if tilt_deg.size == 0:
        return BalanceMetricResult(
            mean_tilt_deg=float("nan"),
            tilt_std_deg=float("nan"),
            com_std_norm=float("nan"),
            out_of_corridor_ratio=0.0,
            score=0.0,
            per_frame_tilt_deg=np.array([], dtype=np.float32),
            per_frame_com_offset_norm=np.array([], dtype=np.float32),
            timestamps=np.array([], dtype=np.float32),
            error_segments=[],
        )

    # Основные статистики
    mean_tilt = float(tilt_deg.mean())
    tilt_std = float(tilt_deg.std())
    com_std = float(com_offset_norm.std())

    # Доля кадров, где есть явное нарушение:
    #  - корпус сильно завален,
    #  - CoM выходит за допустимый "коридор".
    bad_mask = (tilt_deg > tilt_limit_deg) | (np.abs(com_offset_norm) > com_limit)
    out_of_corridor_ratio = float(bad_mask.mean())

    # Переводим в скор 0–100.
    # Чем меньше средний наклон/разброс CoM и меньше bad_mask, тем лучше:
    #   penalty = alpha * com_std + beta * out_of_corridor_ratio + (mean_tilt / 90)
    penalty = alpha * com_std + beta * out_of_corridor_ratio + (mean_tilt / 90.0)
    penalty = float(max(0.0, penalty))
    score = float(100.0 * math.exp(-penalty))
    score = float(np.clip(score, 0.0, 100.0))

    # Сегменты плохого баланса по времени
    error_segments: List[Dict] = []
    in_seg = False
    seg_start = 0

    for i in range(len(tilt_deg)):
        if bad_mask[i] and not in_seg:
            in_seg = True
            seg_start = i
        elif (not bad_mask[i]) and in_seg:
            in_seg = False
            seg_end = i - 1
            if seg_end >= seg_start:
                error_segments.append({
                    "start_idx": int(seg_start),
                    "end_idx": int(seg_end),
                    "start_time": float(timestamps[seg_start]),
                    "end_time": float(timestamps[seg_end]),
                    "mean_tilt_deg": float(tilt_deg[seg_start:seg_end+1].mean()),
                    "mean_com_offset_norm": float(com_offset_norm[seg_start:seg_end+1].mean()),
                })

    if in_seg:
        seg_end = len(tilt_deg) - 1
        error_segments.append({
            "start_idx": int(seg_start),
            "end_idx": int(seg_end),
            "start_time": float(timestamps[seg_start]),
            "end_time": float(timestamps[seg_end]),
            "mean_tilt_deg": float(tilt_deg[seg_start:seg_end+1].mean()),
            "mean_com_offset_norm": float(com_offset_norm[seg_start:seg_end+1].mean()),
        })

    return BalanceMetricResult(
        mean_tilt_deg=mean_tilt,
        tilt_std_deg=tilt_std,
        com_std_norm=com_std,
        out_of_corridor_ratio=out_of_corridor_ratio,
        score=score,
        per_frame_tilt_deg=tilt_deg,
        per_frame_com_offset_norm=com_offset_norm,
        timestamps=timestamps,
        error_segments=error_segments,
    )





