"""
Создание видео с метриками на экране
"""

import cv2
import json
from pathlib import Path
import os

# Пути
video_dir = Path("outputs/lv_0_20251028145433")
overlay_video = video_dir / "overlay_lv_0_20251028145433.mp4"
result_file = video_dir / "analysis_result.json"
output_video = video_dir / "VIDEO_S_METRIKAMI.mp4"

print("="*70)
print("SOZDANIE VIDOE S METRIKAMI")
print("="*70)

# Проверка файлов
if not overlay_video.exists():
    print(f"[ERROR] Video ne naiden: {overlay_video}")
    exit(1)

if not result_file.exists():
    print(f"[ERROR] Results ne naideny: {result_file}")
    exit(1)

# Загружаем результаты
print("\n[OK] Zagruzka rezultatov...")
with open(result_file, 'r', encoding='utf-8') as f:
    result = json.load(f)

figure = result['classification']['figure']
confidence = result['classification']['confidence'] * 100
scores = result['scores']

print(f"  Figura: {figure}")
print(f"  Uverennost: {confidence:.1f}%")
print(f"  Technique: {scores['technique']['score']:.1f}")
print(f"  Timing: {scores['timing']['score']:.1f}")
print(f"  Balance: {scores['balance']['score']:.1f}")

# Открываем видео
print(f"\n[OK] Otkrytie video: {overlay_video.name}")
cap = cv2.VideoCapture(str(overlay_video))

if not cap.isOpened():
    print("[ERROR] Ne udalos otkryt video")
    exit(1)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"  Razmer: {width}x{height}")
print(f"  FPS: {fps}")
print(f"  Kadrov: {total_frames}")

# Создаем выходное видео
print(f"\n[OK] Sozdanie vykhodnogo video...")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

if not out.isOpened():
    print("[ERROR] Ne udalos sozdat video writer")
    cap.release()
    exit(1)

# Метрики для отображения
metrics_list = [
    ("Technique", scores['technique']['score']),
    ("Timing", scores['timing']['score']),
    ("Balance", scores['balance']['score']),
    ("Dynamics", scores['dynamics']['score']),
    ("Posture", scores['posture']['score'])
]

print(f"\n[OK] Obrabotka kadrov...")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]
    
    # ===== ВЕРХНЯЯ ПАНЕЛЬ =====
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 260), (0, 0, 0), -1)  # Увеличена высота панели (было 220, стало 260)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Название фигуры (ОПУЩЕНО ЕЩЕ НА 20% - было y=160, стало y=200)
    cv2.putText(frame, f"FIGURE: {figure}", (120, 200),
               cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 255, 255), 4)
    
    # Уверенность (ОПУЩЕНО ЕЩЕ НА 20% - было y=200, стало y=240)
    conf_color = (0, 255, 0) if confidence >= 70 else (0, 165, 255) if confidence >= 50 else (0, 0, 255)
    cv2.putText(frame, f"CONFIDENCE: {confidence:.1f}%", (120, 240),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, conf_color, 3)
    
    # ===== ПРАВАЯ ПАНЕЛЬ - МЕТРИКИ =====
    panel_w = 360  # Увеличена ширина (было 300) чтобы цифры поместились
    panel_h = 450
    panel_x = w - panel_w - 400  # СДВИНУТО СИЛЬНО ВЛЕВО (было 250, стало 400)
    panel_y = 180  # Опущено ниже
    
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0)
    
    # Заголовок
    cv2.putText(frame, "QUALITY METRICS", (panel_x + 15, panel_y + 40),
               cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
    
    # Линия
    cv2.line(frame, (panel_x + 15, panel_y + 55), (panel_x + panel_w - 15, panel_y + 55), (200, 200, 200), 2)
    
    # Метрики
    y_pos = panel_y + 90
    for name, score in metrics_list:
        # Цвет
        if score >= 70:
            color = (0, 255, 0)  # Зеленый
        elif score >= 50:
            color = (0, 165, 255)  # Оранжевый
        else:
            color = (0, 0, 255)  # Красный
        
        # Название
        cv2.putText(frame, name, (panel_x + 15, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (220, 220, 220), 2)
        
        # Прогресс-бар (УМЕНЬШЕН чтобы цифры поместились)
        bar_x = panel_x + 15
        bar_y = y_pos + 15
        bar_w = 180  # УМЕНЬШЕНО (было 200) чтобы цифры поместились с крупным шрифтом
        bar_h = 28
        
        # Фон
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 2)
        
        # Заполнение
        fill_w = int(bar_w * (score / 100))
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x + 2, bar_y + 2), (bar_x + fill_w - 2, bar_y + bar_h - 2), color, -1)
        
        # Значение (справа от бара, ВНУТРИ панели)
        score_text = f"{score:.1f}"
        text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]  # Увеличен с 0.85 до 1.0
        text_x = bar_x + bar_w + 10  # Отступ от бара
        text_y = bar_y + bar_h // 2 + text_size[1] // 2
        # Проверяем, не выходит ли за край панели, если выходит - уменьшаем отступ
        max_x = panel_x + panel_w - 10  # Отступ от правого края
        if text_x + text_size[0] > max_x:
            text_x = max_x - text_size[0] - 5  # Сдвигаем влево если не помещается
        # Всегда рисуем цифры
        cv2.putText(frame, score_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)  # Увеличен с 0.85 до 1.0
        
        y_pos += 70
    
    # ===== НИЖНЯЯ ПАНЕЛЬ - ПРОГРЕСС =====
    overlay3 = frame.copy()
    cv2.rectangle(overlay3, (0, h-60), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay3, 0.7, frame, 0.3, 0)
    
    progress = frame_idx / total_frames
    bar_w = w - 40
    bar_h = 20
    fill_w = int(bar_w * progress)
    
    cv2.rectangle(frame, (20, h-40), (20 + bar_w, h-20), (60, 60, 60), -1)
    cv2.rectangle(frame, (20, h-40), (20 + bar_w, h-20), (100, 100, 100), 2)
    cv2.rectangle(frame, (20, h-40), (20 + fill_w, h-20), (0, 255, 0), -1)
    
    # Время
    time_text = f"{frame_idx}/{total_frames} frames"
    cv2.putText(frame, time_text, (20, h-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    out.write(frame)
    frame_idx += 1
    
    if frame_idx % 30 == 0:
        percent = (frame_idx / total_frames) * 100
        print(f"  Progress: {percent:.1f}% ({frame_idx}/{total_frames})", end='\r')

cap.release()
out.release()

print(f"\n\n[OK] Video sozdano: {output_video}")
print(f"  Razmer: {output_video.stat().st_size / (1024*1024):.2f} MB")
print(f"  Put: {output_video.absolute()}")

# Открываем
print("\n[OK] Otkryvayu video...")
try:
    os.startfile(str(output_video.absolute()))
    print("[OK] Video otkryto v pleere!")
except Exception as e:
    print(f"[WARN] Ne udalos otkryt avtomaticheski: {e}")
    print(f"Otkroite vruchnuyu: {output_video.absolute()}")

print("\n" + "="*70)
print("GOTOVO!")
print("="*70)




