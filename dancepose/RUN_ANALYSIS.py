"""
ГЛАВНЫЙ СКРИПТ - АНАЛИЗ ТАНЦА

Выдаёт:
1. ТЕКСТОВЫЙ ОТЧЁТ (в консоли)
2. ВИДЕО С ОЦЕНКАМИ (автоматически открывается)

Использование: python RUN_ANALYSIS.py <папка_с_видео>
Пример: python RUN_ANALYSIS.py outputs/lv_0_20251028145628
"""

import subprocess
import sys
from pathlib import Path
import json
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python RUN_ANALYSIS.py <video_folder>")
        print("Example: python RUN_ANALYSIS.py outputs/lv_0_20251028145628")
        return
    
    video_folder = Path(sys.argv[1])
    poses_file = video_folder / "poses.jsonl"
    
    if not poses_file.exists():
        print(f"Error: {poses_file} not found")
        return
    
    print("=" * 80)
    print(f"ANALYZING: {video_folder.name}")
    print("=" * 80)
    
    # Шаг 1: Анализ
    print("\n[1/3] Running full analysis...")
    subprocess.run([
        sys.executable, "analyze_dance.py",
        str(poses_file),
        "dance_classifier/best_model_20pct_adapted.pth"
    ])
    
    # Шаг 2: Визуализация
    print("\n[2/3] Creating video with scores...")
    subprocess.run([
        sys.executable, "visualize_errors_on_video.py",
        str(video_folder)
    ])
    
    # Показываем результаты
    analysis_file = video_folder / "analysis_result.json"
    video_file = video_folder / f"analyzed_{video_folder.name}.mp4"
    
    if analysis_file.exists():
        with open(analysis_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("\n" + "=" * 80)
        print("TEXT RESULTS:")
        print("=" * 80)
        print(f"\nFigure: {data['classification']['figure']}")
        print(f"Confidence: {data['classification']['confidence']:.1%}")
        print(f"\nOverall Score: {data['scores']['overall']:.1f}/100")
        print(f"\nDetailed Scores:")
        for key in ['technique', 'timing', 'balance', 'dynamics', 'posture']:
            score = data['scores'][key]['score']
            bar = '=' * int(score/5) + '-' * (20 - int(score/5))
            print(f"  {key.capitalize():<12} [{bar}] {score:.1f}/100")
        print(f"\nErrors Found: {data['total_errors']}")
    
    # Создаём текстовый отчёт
    print("\n[3/3] Creating text report...")
    subprocess.run([
        sys.executable, "save_text_report.py",
        str(video_folder)
    ])
    
    if video_file.exists():
        print("\n" + "=" * 80)
        print("COMPLETE!")
        print("=" * 80)
        print(f"\nCreated:")
        print(f"  1. Text report: REPORT.txt (opened)")
        print(f"  2. Video: {video_file.name}")
        print("\nOpening video...")
        os.startfile(str(video_file.absolute()))
        print("\n" + "=" * 80)
        print("Check your text editor and video player!")
        print("=" * 80)
    else:
        print(f"\nWarning: Video not created: {video_file}")

if __name__ == "__main__":
    main()

