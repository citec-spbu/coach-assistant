"""
Визуализация ошибок на видео
Наложение меток на существующее overlay видео
"""

import cv2
import json
import numpy as np
from pathlib import Path
import sys


def draw_error_markers(frame, errors_at_frame, frame_height, scale=1.0):
    """
    Рисует маркеры ошибок на кадре
    
    Args:
        frame: Кадр видео
        errors_at_frame: Список ошибок в этом кадре
        frame_height: Высота кадра
        scale: Масштаб текста
    """
    if not errors_at_frame:
        return frame
    
    # Цвета по серьезности (более яркие)
    colors = {
        'high': (0, 0, 255),      # Красный
        'medium': (0, 140, 255),  # Оранжевый
        'low': (0, 255, 255)      # Желтый
    }
    
    # Добавляем красную рамку если есть критичные ошибки
    has_high = any(e.get('severity') == 'high' for e in errors_at_frame)
    if has_high:
        thickness = max(10, int(12 * scale))
        cv2.rectangle(frame, (10, 10), 
                     (frame.shape[1]-10, frame.shape[0]-10), 
                     (0, 0, 255), thickness)
    
    # Умеренно увеличенный текст с ошибками
    font_scale = 1.0 * scale  # Оптимальный размер
    thickness = max(3, int(3 * scale))
    y_offset = int(80 * scale)
    
    for error in errors_at_frame[:3]:  # Показываем топ-3
        severity = error.get('severity', 'medium')
        color = colors.get(severity, (0, 255, 255))
        
        # Форматируем текст
        if 'joint' in error:
            text = f"{error['joint']}: {error['current_value']:.0f} (need {error['expected_value']:.0f})"
        else:
            text = error.get('issue', 'Error')[:50]
        
        # Фон для текста (черный с белой обводкой)
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        padding = int(15 * scale)
        
        # Черный фон
        cv2.rectangle(frame, 
                     (padding, y_offset - text_height - padding), 
                     (padding*2 + text_width, y_offset + padding), 
                     (0, 0, 0), -1)
        
        # Белая обводка вокруг фона
        cv2.rectangle(frame, 
                     (padding, y_offset - text_height - padding), 
                     (padding*2 + text_width, y_offset + padding), 
                     (255, 255, 255), max(2, int(3*scale)))
        
        # Текст с обводкой (сначала черная обводка)
        cv2.putText(frame, text, (padding*2, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness+4)
        # Потом цветной текст
        cv2.putText(frame, text, (padding*2, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        y_offset += int((text_height + padding*2 + 20) * scale)
    
    return frame


def add_score_panel(frame, scores, scale=1.0):
    """
    Добавляет панель с оценками
    
    Args:
        frame: Кадр видео
        scores: Словарь с оценками
        scale: Масштаб текста
    """
    # Умеренно увеличенная панель
    panel_height = int(300 * scale)
    panel_width = int(450 * scale)
    panel_x = frame.shape[1] - panel_width - int(30 * scale)
    panel_y = int(30 * scale)
    
    # Полупрозрачный фон
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height), 
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Белая рамка вокруг панели
    cv2.rectangle(frame, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height), 
                 (255, 255, 255), max(3, int(4*scale)))
    
    # Умеренно увеличенный заголовок
    title_font = 1.2 * scale
    title_thickness = max(3, int(3 * scale))
    cv2.putText(frame, "SCORES", (panel_x + int(15*scale), panel_y + int(45*scale)), 
               cv2.FONT_HERSHEY_SIMPLEX, title_font, (255, 255, 255), title_thickness)
    
    # Оценки
    metrics = [
        ('Overall', scores['overall']),
        ('Technique', scores['technique']['score']),
        ('Timing', scores['timing']['score']),
        ('Balance', scores['balance']['score']),
        ('Dynamics', scores['dynamics']['score'])
    ]
    
    y_pos = panel_y + int(90 * scale)
    font_scale = 0.8 * scale
    font_thickness = max(2, int(2 * scale))
    
    for name, score in metrics:
        # Название
        cv2.putText(frame, f"{name}:", (panel_x + int(20*scale), y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220, 220, 220), font_thickness)
        
        # Оценка (СЛЕВА от бара)
        color = (0, 255, 0) if score >= 85 else (0, 165, 255) if score >= 70 else (0, 0, 255)
        score_x = panel_x + int(180 * scale)
        cv2.putText(frame, f"{score:.0f}", (score_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness+1)
        
        # Умеренно увеличенный бар (СПРАВА от цифры)
        bar_x = panel_x + int(250 * scale)
        bar_width = int((panel_width - 270 * scale) * score / 100)
        bar_height = int(15 * scale)
        
        # Фон бара (серый)
        cv2.rectangle(frame, (bar_x, y_pos - int(15*scale)), 
                     (bar_x + int(panel_width - 270*scale), y_pos - int(15*scale) + bar_height), 
                     (50, 50, 50), -1)
        
        # Заполненный бар
        cv2.rectangle(frame, (bar_x, y_pos - int(15*scale)), 
                     (bar_x + bar_width, y_pos - int(15*scale) + bar_height), 
                     color, -1)
        
        y_pos += int(45 * scale)
    
    return frame


def visualize_errors_on_video(video_path, analysis_result_path, output_path):
    """
    Создаёт видео с наложенными метками ошибок
    
    Args:
        video_path: Путь к исходному overlay видео
        analysis_result_path: Путь к JSON с результатами анализа
        output_path: Путь для сохранения результата
    """
    # Загружаем результаты анализа
    with open(analysis_result_path, 'r') as f:
        result = json.load(f)
    
    # Группируем ошибки по кадрам
    errors_by_frame = {}
    for error in result['errors']:
        frame_num = error['frame']
        if frame_num not in errors_by_frame:
            errors_by_frame[frame_num] = []
        errors_by_frame[frame_num].append(error)
    
    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return False
    
    # Параметры видео
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Вычисляем масштаб на основе разрешения (4K видео -> scale ~2.0)
    scale = min(width, height) / 1080.0
    
    print(f"Video info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Errors to visualize: {len(result['errors'])}")
    print(f"  Text scale: {scale:.2f}x")
    
    # Создаём writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Cannot create output video")
        cap.release()
        return False
    
    print(f"\nProcessing video...")
    
    frame_num = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Добавляем маркеры ошибок (с масштабом)
        errors_here = errors_by_frame.get(frame_num, [])
        frame = draw_error_markers(frame, errors_here, height, scale)
        
        # Добавляем панель с оценками (с масштабом)
        frame = add_score_panel(frame, result['scores'], scale)
        
        # Умеренно увеличенная информация о фигуре
        figure_text = f"Figure: {result['classification']['figure']} ({result['classification']['confidence']*100:.0f}%)"
        figure_font = 1.0 * scale
        figure_thickness = max(2, int(3 * scale))
        y_bottom = height - int(50 * scale)
        
        # Черная обводка
        cv2.putText(frame, figure_text, (int(30*scale), y_bottom), 
                   cv2.FONT_HERSHEY_SIMPLEX, figure_font, (0, 0, 0), figure_thickness+2)
        # Белый текст
        cv2.putText(frame, figure_text, (int(30*scale), y_bottom), 
                   cv2.FONT_HERSHEY_SIMPLEX, figure_font, (255, 255, 255), figure_thickness)
        
        # Умеренно увеличенный номер кадра
        frame_text = f"Frame: {frame_num}/{total_frames}"
        frame_font = 0.7 * scale
        frame_thickness = max(2, int(2 * scale))
        y_frame = height - int(90 * scale)
        
        # Черная обводка
        cv2.putText(frame, frame_text, (int(40*scale), y_frame), 
                   cv2.FONT_HERSHEY_SIMPLEX, frame_font, (0, 0, 0), frame_thickness+2)
        # Белый текст
        cv2.putText(frame, frame_text, (int(40*scale), y_frame), 
                   cv2.FONT_HERSHEY_SIMPLEX, frame_font, (200, 200, 200), frame_thickness)
        
        # Записываем
        out.write(frame)
        
        frame_num += 1
        
        if frame_num % 30 == 0:
            print(f"  Progress: {frame_num}/{total_frames} ({frame_num/total_frames*100:.1f}%)")
    
    cap.release()
    out.release()
    
    print(f"\nDone! Output saved to: {output_path}")
    return True


def main():
    """Главная функция"""
    if len(sys.argv) < 2:
        # По умолчанию обрабатываем первое видео
        video_folder = Path("outputs/lv_0_20251028145433")
    else:
        video_folder = Path(sys.argv[1])
    
    # Файлы
    overlay_video = video_folder / f"overlay_{video_folder.name}.mp4"
    analysis_result = video_folder / "analysis_result.json"
    output_video = video_folder / f"analyzed_{video_folder.name}.mp4"
    
    # Проверки
    if not overlay_video.exists():
        print(f"Error: Video not found: {overlay_video}")
        return
    
    if not analysis_result.exists():
        print(f"Error: Analysis result not found: {analysis_result}")
        print(f"Please run: python analyze_dance.py {video_folder}/poses.jsonl")
        return
    
    print(f"Input video: {overlay_video}")
    print(f"Analysis result: {analysis_result}")
    print(f"Output video: {output_video}")
    
    # Обрабатываем
    success = visualize_errors_on_video(
        str(overlay_video),
        str(analysis_result),
        str(output_video)
    )
    
    if success:
        print(f"\nSuccess! You can now watch: {output_video}")
    else:
        print(f"\nFailed to create video")


if __name__ == "__main__":
    main()


