"""
Главный скрипт для полного анализа танцевального видео
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

# Импорт классификатора
sys.path.append(str(Path(__file__).parent.parent / "FOR_GITHUB" / "coach-assistant"))
from use_classifier import classify_video

# Импорт анализаторов метрик
from metrics_analyzer.technique import TechniqueAnalyzer
from metrics_analyzer.timing import TimingAnalyzer
from metrics_analyzer.balance import BalanceAnalyzer
from metrics_analyzer.dynamics import DynamicsAnalyzer
from metrics_analyzer.posture import PostureAnalyzer


class DanceAnalyzer:
    """Полный анализатор танцевальных выступлений"""
    
    def __init__(self, reference_db_path: str = None):
        if reference_db_path is None:
            # Автоматически определяем путь относительно этого файла
            reference_db_path = Path(__file__).parent / "reference_database"
        self.reference_db_path = Path(reference_db_path)
        self.reference_cache = {}
    
    def load_reference(self, figure: str) -> Dict:
        """
        Загружает эталонные значения для фигуры
        
        Args:
            figure: Название фигуры
            
        Returns:
            Словарь с эталонными углами
        """
        if figure in self.reference_cache:
            return self.reference_cache[figure]
        
        reference_file = self.reference_db_path / f"{figure}.json"
        
        if not reference_file.exists():
            print(f"Warning: No reference found for {figure}, using default")
            return {}
        
        with open(reference_file, 'r') as f:
            reference = json.load(f)
        
        self.reference_cache[figure] = reference
        return reference
    
    def load_poses(self, poses_file: str) -> List[Dict]:
        """Загружает позы из poses.jsonl"""
        poses = []
        with open(poses_file, 'r') as f:
            for line in f:
                poses.append(json.loads(line))
        return poses
    
    def analyze_from_poses(
        self, 
        poses_file: str,
        model_path: str = "dance_classifier/best_model_10pct.pth",
        video_path: str = None
    ) -> Dict:
        """
        Полный анализ из poses.jsonl
        
        Args:
            poses_file: Путь к poses.jsonl
            model_path: Путь к модели классификатора
            video_path: Путь к видео (опционально, для анализа музыки)
            
        Returns:
            Полные результаты анализа
        """
        print(f"Analyzing: {poses_file}")
        print("-" * 60)
        
        # 1. Классификация движения
        print("Step 1/6: Classifying movement...")
        classification = classify_video(poses_file, model_path)
        
        if not classification['success']:
            return {
                'success': False,
                'error': classification.get('error', 'Classification failed')
            }
        
        figure = classification['predicted_figure']
        confidence = classification['confidence']
        
        print(f"  Figure: {figure} (confidence: {confidence:.1%})")
        
        # 2. Загрузка поз
        print("Step 2/6: Loading poses...")
        poses = self.load_poses(poses_file)
        print(f"  Loaded {len(poses)} frames")
        
        # 3. Загрузка эталонов
        print("Step 3/6: Loading reference data...")
        reference = self.load_reference(figure)
        
        if not reference:
            print("  Warning: No reference data, skipping comparison")
        
        # 4. Анализ техники
        print("Step 4/6: Analyzing technique...")
        technique_analyzer = TechniqueAnalyzer()
        video_angles = technique_analyzer.extract_angles_from_video(poses)
        
        if reference:
            technique_score = technique_analyzer.calculate_technique_score(video_angles, reference)
            technique_errors = technique_analyzer.compare_with_reference(video_angles, reference)
        else:
            technique_score = {'score': 0, 'status': 'no_reference', 'details': {}}
            technique_errors = []
        
        print(f"  Technique score: {technique_score['score']:.1f}/100")
        
        # 5. Анализ тайминга
        print("Step 5/6: Analyzing timing...")
        timing_analyzer = TimingAnalyzer()
        
        if video_path:
            try:
                from metrics_analyzer.timing import extract_audio_and_analyze
                timing_result = extract_audio_and_analyze(video_path, poses)
            except Exception as e:
                print(f"  Warning: Could not analyze audio: {e}")
                timing_result = timing_analyzer.analyze_without_music(poses)
        else:
            timing_result = timing_analyzer.analyze_without_music(poses)
        
        print(f"  Timing score: {timing_result['score']:.1f}/100")
        
        # 6. Анализ баланса, динамики, осанки
        print("Step 6/6: Analyzing balance, dynamics, posture...")
        
        balance_analyzer = BalanceAnalyzer()
        balance_result = balance_analyzer.analyze_balance(poses)
        balance_errors = balance_analyzer.find_balance_issues(poses)
        
        dynamics_analyzer = DynamicsAnalyzer()
        dynamics_result = dynamics_analyzer.analyze_dynamics(poses)
        
        posture_analyzer = PostureAnalyzer()
        posture_result = posture_analyzer.analyze_posture(poses)
        posture_errors = posture_analyzer.find_posture_issues(poses)
        
        print(f"  Balance score: {balance_result['score']:.1f}/100")
        print(f"  Dynamics score: {dynamics_result['score']:.1f}/100")
        print(f"  Posture score: {posture_result['score']:.1f}/100")
        
        # Общая оценка
        scores = [
            technique_score['score'],
            timing_result['score'],
            balance_result['score'],
            dynamics_result['score'],
            posture_result['score']
        ]
        
        overall_score = sum(scores) / len(scores)
        
        # Собираем все ошибки
        all_errors = []
        
        # Ошибки техники
        for err in technique_errors:
            for joint, details in err['errors'].items():
                all_errors.append({
                    'timestamp': err['timestamp'],
                    'frame': err['frame'],
                    'category': 'technique',
                    'joint': joint,
                    'issue': f"{joint}: {details['current']:.0f}° (expected {details['expected']:.0f}°)",
                    'severity': details['severity'],
                    'current_value': details['current'],
                    'expected_value': details['expected'],
                    'deviation': details['diff']
                })
        
        # Ошибки баланса
        all_errors.extend(balance_errors)
        
        # Ошибки осанки
        all_errors.extend(posture_errors)
        
        # Сортируем по времени
        all_errors.sort(key=lambda x: x['timestamp'])
        
        # Результат
        result = {
            'success': True,
            'video': str(poses_file),
            'classification': {
                'figure': figure,
                'confidence': float(confidence),
                'all_predictions': classification.get('all_predictions', [])
            },
            'scores': {
                'overall': float(overall_score),
                'technique': technique_score,
                'timing': timing_result,
                'balance': balance_result,
                'dynamics': dynamics_result,
                'posture': posture_result
            },
            'errors': all_errors[:20],  # Топ-20 ошибок
            'total_errors': len(all_errors),
            'total_frames': len(poses)
        }
        
        print("\n" + "=" * 60)
        print(f"OVERALL SCORE: {overall_score:.1f}/100")
        print(f"ERRORS FOUND: {len(all_errors)}")
        print("=" * 60)
        
        return result
    
    def print_report(self, result: Dict):
        """Выводит читаемый отчёт"""
        if not result['success']:
            print(f"Error: {result.get('error', 'Unknown error')}")
            return
        
        print("\n" + "=" * 60)
        print("DANCE ANALYSIS REPORT")
        print("=" * 60)
        
        # Классификация
        classification = result['classification']
        print(f"\nFigure: {classification['figure']}")
        print(f"Confidence: {classification['confidence']:.1%}")
        
        # Оценки
        print("\nSCORES:")
        scores = result['scores']
        print(f"  Overall:  {scores['overall']:.1f}/100")
        print(f"  Technique: {scores['technique']['score']:.1f}/100 ({scores['technique']['status']})")
        print(f"  Timing:    {scores['timing']['score']:.1f}/100 ({scores['timing']['status']})")
        print(f"  Balance:   {scores['balance']['score']:.1f}/100 ({scores['balance']['status']})")
        print(f"  Dynamics:  {scores['dynamics']['score']:.1f}/100 ({scores['dynamics']['status']})")
        print(f"  Posture:   {scores['posture']['score']:.1f}/100 ({scores['posture']['status']})")
        
        # Ошибки
        print(f"\nERRORS: {result['total_errors']} found")
        
        if result['errors']:
            print("\nTop 5 errors:")
            for i, error in enumerate(result['errors'][:5], 1):
                print(f"\n  {i}. [{error['timestamp']:.1f}s] {error['category'].upper()}")
                print(f"     {error['issue']}")
                if error.get('severity'):
                    print(f"     Severity: {error['severity']}")
        
        print("\n" + "=" * 60)


def main():
    """Главная функция"""
    if len(sys.argv) < 2:
        print("Usage: python analyze_dance.py <poses.jsonl> [model_path] [video_path]")
        print("\nExample:")
        print("  python analyze_dance.py outputs/video1/poses.jsonl")
        print("  python analyze_dance.py outputs/video1/poses.jsonl dance_classifier/best_model_10pct.pth")
        print("  python analyze_dance.py outputs/video1/poses.jsonl dance_classifier/best_model_10pct.pth Видео движение/Видео движение/video.mp4")
        return
    
    poses_file = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "../FOR_GITHUB/coach-assistant/best_model_10pct.pth"
    video_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    analyzer = DanceAnalyzer()
    result = analyzer.analyze_from_poses(poses_file, model_path, video_path)
    
    # Выводим отчёт
    analyzer.print_report(result)
    
    # Сохраняем JSON
    output_file = Path(poses_file).parent / "analysis_result.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nFull results saved to: {output_file}")


if __name__ == "__main__":
    main()



