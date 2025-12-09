import math
import time
import numpy as np
from typing import List, Dict, Optional, Union

class LowPassFilter:
    """
    Базовый фильтр низких частот (Low Pass Filter).
    Используется для уменьшения высокочастотного шума, делая сигнал более плавным.
    """
    def __init__(self, alpha: float = 0.5):
        self.prev_value: Optional[float] = None
        self.alpha: float = alpha

    def __call__(self, value: float, alpha: float = None) -> float:
        if alpha is not None:
            self.alpha = alpha
            
        if self.prev_value is None:
            self.prev_value = value
            return value
            
        filtered_value = self.alpha * value + (1.0 - self.alpha) * self.prev_value
        self.prev_value = filtered_value
        return filtered_value


class OneEuroFilter:
    """
    Реализация алгоритма OneEuroFilter.
    Это адаптивный фильтр низких частот, который идеально подходит для оценки поз в реальном времени.
    Он устраняет дрожание (шум) при медленных движениях и минимизирует задержку при быстрых движениях.
    
    Reference: 
    Casiez, G., Roussel, N., & Vogel, D. (2012). 1 € filter: a simple speed-based low-pass filter for noisy input in interactive systems.
    """
    def __init__(self, t0: float, x0: float, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        """
        Инициализация фильтра.

        Args:
            t0 (float): Начальная метка времени.
            x0 (float): Начальное значение.
            min_cutoff (float): Минимальная частота среза (Гц). Чем меньше значение, тем плавне медленные движения.
            beta (float): Коэффициент скорости. Чем больше значение, тем быстрее реакция на быстрые движения (меньше задержка).
            d_cutoff (float): Частота среза для производной (Гц).
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        
        self.x_prev = float(x0)
        self.t_prev = float(t0)

    def _smoothing_factor(self, t_e: float, cutoff: float) -> float:
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def _exponential_smoothing(self, alpha: float, x: float, x_prev: float) -> float:
        return alpha * x + (1 - alpha) * x_prev

    def __call__(self, t: float, x: float) -> float:
        """
        Фильтрация нового входящего значения.

        Args:
            t (float): Текущая метка времени.
            x (float): Текущее наблюдаемое значение.

        Returns:
            float: Отфильтрованное значение.
        """
        t_e = t - self.t_prev

        # Избегаем деления на ноль или обратного хода времени
        if t_e <= 0.0:
            return self.x_prev

        # Вычисляем скорость изменения сигнала (производную)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.dx_filter(dx, alpha=self._smoothing_factor(t_e, self.d_cutoff))

        # Динамически настраиваем частоту среза в зависимости от скорости
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # Фильтруем сам сигнал
        x_hat = self.x_filter(x, alpha=self._smoothing_factor(t_e, cutoff))

        self.x_prev = x_hat
        self.t_prev = t
        return x_hat


class PoseStabilizer:
    """
    Стабилизатор позы.
    Управляет группой фильтров для 17 ключевых точек (для каждой точки x, y).
    """
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5):
        self._filters: Dict[int, Dict[str, OneEuroFilter]] = {}
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.start_time = time.time()

    def update(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Обновление и сглаживание ключевых точек.

        Args:
            keypoints (np.ndarray): Массив формы (N, 3), содержащий [x, y, score].

        Returns:
            np.ndarray: Сглаженный массив ключевых точек.
        """
        current_time = time.time() - self.start_time
        smoothed_kps = np.copy(keypoints)
        
        num_joints = keypoints.shape[0]

        for i in range(num_joints):
            x, y, s = keypoints[i]
            
            # Инициализируем фильтр для данного сустава, если его нет
            if i not in self._filters:
                self._filters[i] = {
                    'x': OneEuroFilter(current_time, x, min_cutoff=self.min_cutoff, beta=self.beta),
                    'y': OneEuroFilter(current_time, y, min_cutoff=self.min_cutoff, beta=self.beta)
                }
                continue # Возвращаем исходное значение для первого кадра

            # Применяем фильтрацию
            smooth_x = self._filters[i]['x'](current_time, x)
            smooth_y = self._filters[i]['y'](current_time, y)
            
            # Score (уверенность) не сглаживаем
            smoothed_kps[i] = [smooth_x, smooth_y, s]
            
        return smoothed_kps
