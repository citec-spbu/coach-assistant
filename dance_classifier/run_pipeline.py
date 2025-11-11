"""
Полный pipeline для классификации танцевальных фигур
Запускает все этапы: извлечение поз, подготовку данных, обучение и оценку
"""
import argparse
import sys
from pathlib import Path
import subprocess
import json
import yaml


def run_command(cmd, description):
    """Запускает команду с описанием"""
    print(f"\n{'='*70}")
    print(f"ШАГ: {description}")
    print(f"{'='*70}")
    print(f"Команда: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Ошибка на шаге: {description}")
        sys.exit(1)
    else:
        print(f"\n✓ Шаг завершен: {description}")


def main():
    parser = argparse.ArgumentParser(
        description="Полный pipeline для классификации танцевальных фигур"
    )
    
    # Общие параметры
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Директория с видео файлами')
    parser.add_argument('--output_dir', type=str, default='../outputs',
                        help='Корневая директория для выходных файлов')
    parser.add_argument('--model_path', type=str,
                        default='../dancepose/models/yolov8s-pose.pt',
                        help='Путь к модели YOLOv8-Pose')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Устройство (cuda или cpu)')
    
    # Параметры датасета
    parser.add_argument('--sequence_length', type=int, default=30,
                        help='Длина последовательности')
    parser.add_argument('--overlap', type=int, default=15,
                        help='Перекрытие между последовательностями')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Размер тестовой выборки')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Размер валидационной выборки')
    
    # Параметры обучения
    parser.add_argument('--config', type=str, default='training/config.yaml',
                        help='Путь к конфигурации обучения')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Количество epochs (переопределяет config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Размер батча (переопределяет config)')
    
    # Флаги для пропуска шагов
    parser.add_argument('--skip_pose_extraction', action='store_true',
                        help='Пропустить извлечение поз')
    parser.add_argument('--skip_dataset_building', action='store_true',
                        help='Пропустить построение датасета')
    parser.add_argument('--skip_training', action='store_true',
                        help='Пропустить обучение')
    parser.add_argument('--skip_evaluation', action='store_true',
                        help='Пропустить оценку')
    
    args = parser.parse_args()
    
    # Создаем директории
    output_dir = Path(args.output_dir)
    poses_dir = output_dir / 'poses'
    dataset_dir = output_dir / 'dataset'
    models_dir = output_dir / 'models'
    eval_dir = output_dir / 'evaluation'
    
    print("\n" + "="*70)
    print("ПОЛНЫЙ PIPELINE КЛАССИФИКАЦИИ ТАНЦЕВАЛЬНЫХ ФИГУР")
    print("="*70)
    print(f"\nДиректория с видео: {args.video_dir}")
    print(f"Выходная директория: {output_dir}")
    print(f"Устройство: {args.device}")
    
    # ===== ШАГ 1: ИЗВЛЕЧЕНИЕ ПОЗ =====
    if not args.skip_pose_extraction:
        cmd = [
            sys.executable, 'data_preparation/extract_poses.py',
            '--video_dir', args.video_dir,
            '--output_dir', str(poses_dir),
            '--model_path', args.model_path,
            '--device', args.device
        ]
        run_command(cmd, "Извлечение поз из видео")
    else:
        print("\n⏭️  Пропускаем извлечение поз")
    
    # ===== ШАГ 2: ПОСТРОЕНИЕ ДАТАСЕТА =====
    if not args.skip_dataset_building:
        cmd = [
            sys.executable, 'data_preparation/dataset_builder.py',
            '--poses_dir', str(poses_dir),
            '--output_dir', str(dataset_dir),
            '--sequence_length', str(args.sequence_length),
            '--overlap', str(args.overlap),
            '--test_size', str(args.test_size),
            '--val_size', str(args.val_size)
        ]
        run_command(cmd, "Построение датасета")
    else:
        print("\n⏭️  Пропускаем построение датасета")
    
    # Загружаем/обновляем конфигурацию
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Переопределяем параметры если указаны
    if args.num_epochs is not None:
        config['num_epochs'] = args.num_epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    
    # Сохраняем обновленную конфигурацию
    temp_config_path = output_dir / 'temp_config.yaml'
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    
    # ===== ШАГ 3: ОБУЧЕНИЕ =====
    if not args.skip_training:
        cmd = [
            sys.executable, 'training/train.py',
            '--config', str(temp_config_path),
            '--data_dir', str(dataset_dir),
            '--output_dir', str(models_dir),
            '--device', args.device
        ]
        run_command(cmd, "Обучение модели")
    else:
        print("\n⏭️  Пропускаем обучение")
    
    # ===== ШАГ 4: ОЦЕНКА =====
    if not args.skip_evaluation:
        best_model_path = models_dir / 'best_model.pth'
        if best_model_path.exists():
            cmd = [
                sys.executable, 'inference/predict.py',
                '--model_path', str(best_model_path),
                '--data_dir', str(dataset_dir),
                '--output_dir', str(eval_dir),
                '--device', args.device
            ]
            run_command(cmd, "Оценка модели")
        else:
            print(f"\n⚠️  Модель не найдена: {best_model_path}")
            print("Пропускаем оценку")
    else:
        print("\n⏭️  Пропускаем оценку")
    
    # Удаляем временный файл конфигурации
    if temp_config_path.exists():
        temp_config_path.unlink()
    
    # ===== ФИНАЛЬНЫЙ ОТЧЕТ =====
    print("\n" + "="*70)
    print("PIPELINE ЗАВЕРШЕН")
    print("="*70)
    
    print("\nРезультаты сохранены в:")
    print(f"  📁 Позы:     {poses_dir}")
    print(f"  📁 Датасет:  {dataset_dir}")
    print(f"  📁 Модели:   {models_dir}")
    print(f"  📁 Оценка:   {eval_dir}")
    
    # Выводим метрики если доступны
    metrics_path = eval_dir / 'metrics.json'
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        print("\n📊 Метрики на тестовой выборке:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
    
    # Информация о классах
    metadata_path = dataset_dir / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"\n🏷️  Классы: {metadata['label_encoder']['classes']}")
        print(f"📹 Видео обработано: {len(metadata['videos'])}")
    
    print("\n✅ Все готово!")
    print("\nДля анализа результатов откройте:")
    print(f"  jupyter notebook notebooks/analysis.ipynb")
    print()


if __name__ == "__main__":
    main()


Полный pipeline для классификации танцевальных фигур
Запускает все этапы: извлечение поз, подготовку данных, обучение и оценку
"""
import argparse
import sys
from pathlib import Path
import subprocess
import json
import yaml


def run_command(cmd, description):
    """Запускает команду с описанием"""
    print(f"\n{'='*70}")
    print(f"ШАГ: {description}")
    print(f"{'='*70}")
    print(f"Команда: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Ошибка на шаге: {description}")
        sys.exit(1)
    else:
        print(f"\n✓ Шаг завершен: {description}")


def main():
    parser = argparse.ArgumentParser(
        description="Полный pipeline для классификации танцевальных фигур"
    )
    
    # Общие параметры
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Директория с видео файлами')
    parser.add_argument('--output_dir', type=str, default='../outputs',
                        help='Корневая директория для выходных файлов')
    parser.add_argument('--model_path', type=str,
                        default='../dancepose/models/yolov8s-pose.pt',
                        help='Путь к модели YOLOv8-Pose')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Устройство (cuda или cpu)')
    
    # Параметры датасета
    parser.add_argument('--sequence_length', type=int, default=30,
                        help='Длина последовательности')
    parser.add_argument('--overlap', type=int, default=15,
                        help='Перекрытие между последовательностями')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Размер тестовой выборки')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Размер валидационной выборки')
    
    # Параметры обучения
    parser.add_argument('--config', type=str, default='training/config.yaml',
                        help='Путь к конфигурации обучения')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Количество epochs (переопределяет config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Размер батча (переопределяет config)')
    
    # Флаги для пропуска шагов
    parser.add_argument('--skip_pose_extraction', action='store_true',
                        help='Пропустить извлечение поз')
    parser.add_argument('--skip_dataset_building', action='store_true',
                        help='Пропустить построение датасета')
    parser.add_argument('--skip_training', action='store_true',
                        help='Пропустить обучение')
    parser.add_argument('--skip_evaluation', action='store_true',
                        help='Пропустить оценку')
    
    args = parser.parse_args()
    
    # Создаем директории
    output_dir = Path(args.output_dir)
    poses_dir = output_dir / 'poses'
    dataset_dir = output_dir / 'dataset'
    models_dir = output_dir / 'models'
    eval_dir = output_dir / 'evaluation'
    
    print("\n" + "="*70)
    print("ПОЛНЫЙ PIPELINE КЛАССИФИКАЦИИ ТАНЦЕВАЛЬНЫХ ФИГУР")
    print("="*70)
    print(f"\nДиректория с видео: {args.video_dir}")
    print(f"Выходная директория: {output_dir}")
    print(f"Устройство: {args.device}")
    
    # ===== ШАГ 1: ИЗВЛЕЧЕНИЕ ПОЗ =====
    if not args.skip_pose_extraction:
        cmd = [
            sys.executable, 'data_preparation/extract_poses.py',
            '--video_dir', args.video_dir,
            '--output_dir', str(poses_dir),
            '--model_path', args.model_path,
            '--device', args.device
        ]
        run_command(cmd, "Извлечение поз из видео")
    else:
        print("\n⏭️  Пропускаем извлечение поз")
    
    # ===== ШАГ 2: ПОСТРОЕНИЕ ДАТАСЕТА =====
    if not args.skip_dataset_building:
        cmd = [
            sys.executable, 'data_preparation/dataset_builder.py',
            '--poses_dir', str(poses_dir),
            '--output_dir', str(dataset_dir),
            '--sequence_length', str(args.sequence_length),
            '--overlap', str(args.overlap),
            '--test_size', str(args.test_size),
            '--val_size', str(args.val_size)
        ]
        run_command(cmd, "Построение датасета")
    else:
        print("\n⏭️  Пропускаем построение датасета")
    
    # Загружаем/обновляем конфигурацию
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Переопределяем параметры если указаны
    if args.num_epochs is not None:
        config['num_epochs'] = args.num_epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    
    # Сохраняем обновленную конфигурацию
    temp_config_path = output_dir / 'temp_config.yaml'
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    
    # ===== ШАГ 3: ОБУЧЕНИЕ =====
    if not args.skip_training:
        cmd = [
            sys.executable, 'training/train.py',
            '--config', str(temp_config_path),
            '--data_dir', str(dataset_dir),
            '--output_dir', str(models_dir),
            '--device', args.device
        ]
        run_command(cmd, "Обучение модели")
    else:
        print("\n⏭️  Пропускаем обучение")
    
    # ===== ШАГ 4: ОЦЕНКА =====
    if not args.skip_evaluation:
        best_model_path = models_dir / 'best_model.pth'
        if best_model_path.exists():
            cmd = [
                sys.executable, 'inference/predict.py',
                '--model_path', str(best_model_path),
                '--data_dir', str(dataset_dir),
                '--output_dir', str(eval_dir),
                '--device', args.device
            ]
            run_command(cmd, "Оценка модели")
        else:
            print(f"\n⚠️  Модель не найдена: {best_model_path}")
            print("Пропускаем оценку")
    else:
        print("\n⏭️  Пропускаем оценку")
    
    # Удаляем временный файл конфигурации
    if temp_config_path.exists():
        temp_config_path.unlink()
    
    # ===== ФИНАЛЬНЫЙ ОТЧЕТ =====
    print("\n" + "="*70)
    print("PIPELINE ЗАВЕРШЕН")
    print("="*70)
    
    print("\nРезультаты сохранены в:")
    print(f"  📁 Позы:     {poses_dir}")
    print(f"  📁 Датасет:  {dataset_dir}")
    print(f"  📁 Модели:   {models_dir}")
    print(f"  📁 Оценка:   {eval_dir}")
    
    # Выводим метрики если доступны
    metrics_path = eval_dir / 'metrics.json'
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        print("\n📊 Метрики на тестовой выборке:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
    
    # Информация о классах
    metadata_path = dataset_dir / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"\n🏷️  Классы: {metadata['label_encoder']['classes']}")
        print(f"📹 Видео обработано: {len(metadata['videos'])}")
    
    print("\n✅ Все готово!")
    print("\nДля анализа результатов откройте:")
    print(f"  jupyter notebook notebooks/analysis.ipynb")
    print()


if __name__ == "__main__":
    main()


