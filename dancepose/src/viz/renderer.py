import cv2
import numpy as np
from collections import deque
from typing import List, Tuple, Optional

class VisualStyles:
    """
    Класс констант для стилей визуализации и цветовых схем.
    """
    # Связи скелета в формате COCO
    SKELETON_LINKS = [
        (5, 7), (7, 9), (6, 8), (8, 10),  # Руки
        (11, 13), (13, 15), (12, 14), (14, 16),  # Ноги
        (5, 6), (5, 11), (6, 12), (11, 12),  # Торс
        (5, 6), (5, 11), (6, 12), (11, 12)   # Корпус
    ]
    
    # Цветовая схема "Киберпанк" (BGR)
    COLOR_NEON_BLUE = (255, 255, 0)
    COLOR_NEON_PINK = (203, 192, 255)
    COLOR_NEON_GREEN = (50, 255, 50)
    COLOR_DARK_BG = (20, 20, 20)

class AdvancedPoseRenderer:
    """
    Продвинутый рендерер поз.
    Поддерживает отслеживание траекторий, спецэффекты и информационную панель.
    """
    def __init__(self, width: int, height: int, history_len: int = 10):
        self.width = width
        self.height = height
        # Буфер истории для отрисовки шлейфа движения
        self.history_buffer = deque(maxlen=history_len)
        
    def draw_cyberpunk_skeleton(self, frame: np.ndarray, keypoints: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        """
        Отрисовка скелета с эффектом свечения (стиль киберпанк).
        """
        # Создаем слой для полупрозрачных эффектов
        overlay = frame.copy()
        
        # 1. Рисуем связи скелета
        for idx_a, idx_b in VisualStyles.SKELETON_LINKS:
            if idx_a >= len(keypoints) or idx_b >= len(keypoints):
                continue
                
            kp_a = keypoints[idx_a]
            kp_b = keypoints[idx_b]
            
            # Проверка уверенности (confidence)
            if kp_a[2] < threshold or kp_b[2] < threshold:
                continue
                
            pt_a = (int(kp_a[0]), int(kp_a[1]))
            pt_b = (int(kp_b[0]), int(kp_b[1]))
            
            # Рисуем толстую линию для эффекта "свечения"
            cv2.line(overlay, pt_a, pt_b, VisualStyles.COLOR_NEON_PINK, 6)
            # Рисуем тонкую центральную линию
            cv2.line(frame, pt_a, pt_b, (255, 255, 255), 2)

        # 2. Рисуем суставы
        for idx, kp in enumerate(keypoints):
            if kp[2] < threshold:
                continue
                
            cx, cy = int(kp[0]), int(kp[1])
            
            # Рисуем внешний ореол
            cv2.circle(overlay, (cx, cy), 8, VisualStyles.COLOR_NEON_BLUE, -1)
            # Рисуем ядро
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

        # Смешиваем слои для создания эффекта свечения
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        return frame

    def draw_motion_trails(self, frame: np.ndarray, current_kps: np.ndarray) -> np.ndarray:
        """
        Отрисовка траекторий движения (шлейф) для рук и ног.
        """
        self.history_buffer.append(current_kps)
        
        # Отслеживаем только запястья (9, 10) и лодыжки (15, 16)
        track_indices = [9, 10, 15, 16] 
        
        for joint_idx in track_indices:
            pts = []
            for hist_kps in self.history_buffer:
                if hist_kps[joint_idx][2] > 0.3: # Проверка confidence
                    pts.append((int(hist_kps[joint_idx][0]), int(hist_kps[joint_idx][1])))
            
            if len(pts) < 2:
                continue
                
            # Рисуем градиентный след
            for i in range(len(pts) - 1):
                # Чем новее точка, тем она толще и ярче
                alpha = (i + 1) / len(pts)
                thickness = int(2 + 3 * alpha)
                color = VisualStyles.COLOR_NEON_GREEN
                
                cv2.line(frame, pts[i], pts[i+1], color, thickness)
                
        return frame

    def draw_dashboard(self, frame: np.ndarray, fps: float, confidence: float, model_name: str):
        """
        Отрисовка информационной панели в техно-стиле.
        """
        h, w = frame.shape[:2]
        panel_h = 80
        
        # Верхняя полупрозрачная черная полоса
        sub_img = frame[0:panel_h, 0:w]
        black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        res = cv2.addWeighted(sub_img, 0.7, black_rect, 0.3, 1.0)
        frame[0:panel_h, 0:w] = res
        
        # Отрисовка полосы FPS
        bar_len = 150
        bar_height = 10
        fps_ratio = min(fps / 60.0, 1.0)
        
        # Метка FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        # Фон полосы FPS
        cv2.rectangle(frame, (20, 40), (20 + bar_len, 40 + bar_height), (50, 50, 50), -1)
        # Заполнение полосы FPS
        cv2.rectangle(frame, (20, 40), (20 + int(bar_len * fps_ratio), 40 + bar_height), VisualStyles.COLOR_NEON_BLUE, -1)
        
        # Информация о модели
        cv2.putText(frame, f"MODEL: {model_name}", (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Средняя уверенность (Confidence)
        conf_color = VisualStyles.COLOR_NEON_GREEN if confidence > 0.7 else VisualStyles.COLOR_NEON_PINK
        cv2.putText(frame, f"CONF: {confidence:.2f}", (w - 200, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)
