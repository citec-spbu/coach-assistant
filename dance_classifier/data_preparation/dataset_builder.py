"""
Модуль подготовки датасета для обучения классификатора
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from tqdm import tqdm
import argparse

from .feature_extraction import FeatureExtractor, get_simple_features


class DanceDatasetBuilder:
    """Класс для построения датасета для классификации танцевальных фигур"""
    
    def __init__(self, poses_dir, labels_file=None, feature_names=None, 
                 sequence_length=30, overlap=15):
        """
        Args:
            poses_dir: директория с JSON файлами поз
            labels_file: файл с разметкой (опционально)
            feature_names: список признаков для использования
            sequence_length: длина последовательности (количество кадров)
            overlap: перекрытие между последовательностями
        """
        self.poses_dir = Path(poses_dir)
        self.labels_file = labels_file
        self.feature_names = feature_names or get_simple_features()
        self.sequence_length = sequence_length
        self.overlap = overlap
        
        self.feature_extractor = FeatureExtractor(min_confidence=0.3)
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        self.labels_data = None
        if labels_file and Path(labels_file).exists():
            self._load_labels()
    
    def _load_labels(self):
        """Загружает разметку из файла"""
        labels_path = Path(self.labels_file)
        
        if labels_path.suffix == '.json':
            with open(labels_path, 'r', encoding='utf-8') as f:
                self.labels_data = json.load(f)
        elif labels_path.suffix == '.csv':
            self.labels_data = pd.read_csv(labels_path).to_dict('records')
        else:
            print(f"Неподдерживаемый формат файла разметки: {labels_path.suffix}")
    
    def get_label_for_video(self, video_name):
        """
        Получает метку класса для видео по его имени
        
        Args:
            video_name: имя видео файла
        
        Returns:
            str: название класса танцевальной фигуры
        """
        if self.labels_data is None:
            # Если разметки нет, пытаемся извлечь из имени файла
            # Предполагается формат: "фигура_номер.mp4" или подобный
            # Например: "basic_step_001.mp4" -> "basic_step"
            parts = video_name.split('_')
            if len(parts) >= 2:
                # Убираем числовую часть и расширение
                label = '_'.join([p for p in parts if not p.isdigit()])
                return label.lower()
            return 'unknown'
        
        # Ищем метку в загруженных данных
        for item in self.labels_data:
            if item.get('video_name') == video_name or item.get('video_id') == video_name:
                return item.get('label', 'unknown')
        
        return 'unknown'
    
    def create_sequences(self, features, label, stride=None):
        """
        Создает последовательности из признаков
        
        Args:
            features: массив признаков (num_frames, num_features)
            label: метка класса
            stride: шаг скользящего окна (если None, используется overlap)
        
        Returns:
            list: список кортежей (sequence, label)
        """
        if stride is None:
            stride = self.sequence_length - self.overlap
        
        sequences = []
        num_frames = len(features)
        
        for start_idx in range(0, num_frames - self.sequence_length + 1, stride):
            end_idx = start_idx + self.sequence_length
            sequence = features[start_idx:end_idx]
            
            # Проверяем, что в последовательности нет слишком много пропусков
            valid_ratio = np.sum(~np.isnan(sequence[:, 0])) / len(sequence)
            if valid_ratio >= 0.7:  # Минимум 70% валидных кадров
                # Заполняем пропуски интерполяцией
                sequence = self._interpolate_missing(sequence)
                sequences.append((sequence, label))
        
        return sequences
    
    def _interpolate_missing(self, sequence):
        """
        Заполняет пропущенные значения в последовательности
        
        Args:
            sequence: массив (seq_len, num_features)
        
        Returns:
            массив с заполненными значениями
        """
        sequence = sequence.copy()
        
        for feature_idx in range(sequence.shape[1]):
            feature_values = sequence[:, feature_idx]
            
            # Находим валидные индексы
            valid_mask = ~np.isnan(feature_values)
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                # Если нет валидных значений, заполняем нулями
                sequence[:, feature_idx] = 0.0
                continue
            
            if len(valid_indices) < len(feature_values):
                # Интерполяция
                invalid_indices = np.where(~valid_mask)[0]
                sequence[invalid_indices, feature_idx] = np.interp(
                    invalid_indices,
                    valid_indices,
                    feature_values[valid_indices]
                )
        
        return sequence
    
    def process_video(self, poses_json_path):
        """
        Обрабатывает один видео файл
        
        Args:
            poses_json_path: путь к JSON файлу с позами
        
        Returns:
            tuple: (sequences, labels, video_name)
        """
        # Загружаем данные поз
        with open(poses_json_path, 'r', encoding='utf-8') as f:
            poses_data = json.load(f)
        
        video_name = poses_data['video_name']
        
        # Извлекаем признаки
        feature_list, valid_mask = self.feature_extractor.extract_sequence_features(poses_data)
        
        # Преобразуем в массив
        feature_array, actual_feature_names, valid_indices = \
            self.feature_extractor.features_to_array(feature_list, self.feature_names)
        
        if len(valid_indices) < self.sequence_length:
            print(f"Пропускаем {video_name}: недостаточно валидных кадров ({len(valid_indices)} < {self.sequence_length})")
            return [], [], video_name
        
        # Получаем метку
        label = self.get_label_for_video(video_name)
        
        # Создаем последовательности
        sequences = self.create_sequences(feature_array, label)
        
        return sequences, label, video_name
    
    def build_dataset(self, output_dir=None):
        """
        Строит полный датасет из всех видео
        
        Args:
            output_dir: директория для сохранения датасета
        
        Returns:
            tuple: (X, y, metadata)
        """
        # Находим все JSON файлы с позами
        pose_files = list(self.poses_dir.glob('*_poses.json'))
        print(f"Найдено {len(pose_files)} файлов с позами")
        
        all_sequences = []
        all_labels = []
        metadata = {
            'videos': [],
            'feature_names': self.feature_names,
            'sequence_length': self.sequence_length,
            'overlap': self.overlap
        }
        
        # Обрабатываем каждое видео
        for pose_file in tqdm(pose_files, desc="Обработка видео"):
            sequences, label, video_name = self.process_video(pose_file)
            
            if len(sequences) > 0:
                for seq, lbl in sequences:
                    all_sequences.append(seq)
                    all_labels.append(lbl)
                
                metadata['videos'].append({
                    'video_name': video_name,
                    'label': label,
                    'num_sequences': len(sequences)
                })
        
        print(f"\nСоздано {len(all_sequences)} последовательностей из {len(metadata['videos'])} видео")
        
        # Преобразуем в numpy массивы
        X = np.array(all_sequences)
        y = np.array(all_labels)
        
        # Кодируем метки
        y_encoded = self.label_encoder.fit_transform(y)
        metadata['label_encoder'] = {
            'classes': self.label_encoder.classes_.tolist(),
            'class_to_idx': {cls: idx for idx, cls in enumerate(self.label_encoder.classes_)}
        }
        
        # Нормализуем признаки
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_normalized = self.scaler.fit_transform(X_reshaped)
        X_normalized = X_normalized.reshape(X.shape)
        
        # Сохраняем датасет
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Сохраняем данные
            np.save(output_dir / 'X.npy', X_normalized)
            np.save(output_dir / 'y.npy', y_encoded)
            
            # Сохраняем метаданные
            with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # Сохраняем scaler и label_encoder
            with open(output_dir / 'scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            with open(output_dir / 'label_encoder.pkl', 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            print(f"\nДатасет сохранен в: {output_dir}")
        
        return X_normalized, y_encoded, metadata
    
    def split_dataset(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """
        Разбивает датасет на train/val/test
        
        Args:
            X: массив признаков
            y: массив меток
            test_size: размер тестовой выборки
            val_size: размер валидационной выборки
            random_state: seed для воспроизводимости
        
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Сначала отделяем test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Затем разбиваем оставшееся на train и val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    parser = argparse.ArgumentParser(description="Построение датасета для классификации")
    parser.add_argument('--poses_dir', type=str, required=True,
                        help='Директория с JSON файлами поз')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Директория для сохранения датасета')
    parser.add_argument('--labels_file', type=str, default=None,
                        help='Файл с разметкой (опционально)')
    parser.add_argument('--sequence_length', type=int, default=30,
                        help='Длина последовательности (кадров)')
    parser.add_argument('--overlap', type=int, default=15,
                        help='Перекрытие между последовательностями')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Размер тестовой выборки')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Размер валидационной выборки')
    
    args = parser.parse_args()
    
    # Создаем builder
    builder = DanceDatasetBuilder(
        poses_dir=args.poses_dir,
        labels_file=args.labels_file,
        sequence_length=args.sequence_length,
        overlap=args.overlap
    )
    
    # Строим датасет
    X, y, metadata = builder.build_dataset(output_dir=args.output_dir)
    
    # Разбиваем на train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test = builder.split_dataset(
        X, y, 
        test_size=args.test_size, 
        val_size=args.val_size
    )
    
    # Сохраняем разбиение
    output_dir = Path(args.output_dir)
    np.save(output_dir / 'X_train.npy', X_train)
    np.save(output_dir / 'X_val.npy', X_val)
    np.save(output_dir / 'X_test.npy', X_test)
    np.save(output_dir / 'y_train.npy', y_train)
    np.save(output_dir / 'y_val.npy', y_val)
    np.save(output_dir / 'y_test.npy', y_test)
    
    print(f"\nРазмеры выборок:")
    print(f"Train: {X_train.shape}")
    print(f"Val: {X_val.shape}")
    print(f"Test: {X_test.shape}")
    print(f"\nРаспределение классов в train:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls_idx, count in zip(unique, counts):
        cls_name = metadata['label_encoder']['classes'][cls_idx]
        print(f"  {cls_name}: {count}")
    
    print("\nГотово!")


if __name__ == "__main__":
    main()



