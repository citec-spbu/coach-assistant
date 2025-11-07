"""
Скрипт инференса и оценки качества классификатора
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Добавляем пути к модулям
sys.path.append(str(Path(__file__).parent.parent))

from models.gru_classifier import create_gru_classifier
from models.tcn_classifier import create_tcn_classifier
from data_preparation.feature_extraction import FeatureExtractor


class DanceClassifierPredictor:
    """Класс для inference классификатора танцевальных фигур"""
    
    def __init__(self, model_path, metadata_path, scaler_path, 
                 label_encoder_path, device='cuda'):
        """
        Args:
            model_path: путь к сохраненной модели
            metadata_path: путь к метаданным датасета
            scaler_path: путь к scaler'у
            label_encoder_path: путь к label encoder'у
            device: устройство ('cuda' или 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Загружаем метаданные
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Загружаем scaler и label encoder
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Загружаем модель
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        
        # Создаем модель
        model_type = self.config.get('model_type', 'gru')
        model_config = self.config.get('model', {})
        
        if model_type == 'gru' or 'gru' in model_type.lower():
            self.model = create_gru_classifier(model_config)
        elif model_type == 'tcn' or 'tcn' in model_type.lower():
            self.model = create_tcn_classifier(model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.feature_extractor = FeatureExtractor(min_confidence=0.3)
        
        print(f"Модель загружена с устройства: {self.device}")
        print(f"Классы: {self.label_encoder.classes_}")
    
    def predict_sequence(self, sequence):
        """
        Предсказывает класс для одной последовательности
        
        Args:
            sequence: numpy array (seq_len, num_features)
        
        Returns:
            tuple: (predicted_class, probabilities)
        """
        # Нормализуем
        sequence_norm = self.scaler.transform(sequence)
        
        # Преобразуем в тензор
        x = torch.FloatTensor(sequence_norm).unsqueeze(0).to(self.device)
        
        # Инференс
        with torch.no_grad():
            outputs = self.model(x)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_idx = np.argmax(probabilities)
            predicted_class = self.label_encoder.classes_[predicted_idx]
        
        return predicted_class, probabilities
    
    def predict_batch(self, sequences):
        """
        Предсказывает классы для батча последовательностей
        
        Args:
            sequences: numpy array (batch_size, seq_len, num_features)
        
        Returns:
            tuple: (predicted_classes, all_probabilities)
        """
        # Нормализуем
        batch_size, seq_len, num_features = sequences.shape
        sequences_reshaped = sequences.reshape(-1, num_features)
        sequences_norm = self.scaler.transform(sequences_reshaped)
        sequences_norm = sequences_norm.reshape(batch_size, seq_len, num_features)
        
        # Преобразуем в тензор
        x = torch.FloatTensor(sequences_norm).to(self.device)
        
        # Инференс
        with torch.no_grad():
            outputs = self.model(x)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predicted_indices = np.argmax(probabilities, axis=1)
            predicted_classes = self.label_encoder.classes_[predicted_indices]
        
        return predicted_classes, probabilities
    
    def predict_from_poses(self, poses_json_path):
        """
        Предсказывает класс по JSON файлу с позами
        
        Args:
            poses_json_path: путь к JSON файлу с позами
        
        Returns:
            dict: результаты предсказания
        """
        # Загружаем позы
        with open(poses_json_path, 'r', encoding='utf-8') as f:
            poses_data = json.load(f)
        
        # Извлекаем признаки
        feature_list, valid_mask = self.feature_extractor.extract_sequence_features(poses_data)
        feature_array, feature_names, valid_indices = \
            self.feature_extractor.features_to_array(
                feature_list, self.metadata['feature_names']
            )
        
        # Проверяем длину последовательности
        sequence_length = self.metadata['sequence_length']
        if len(valid_indices) < sequence_length:
            return {
                'success': False,
                'error': f'Недостаточно валидных кадров: {len(valid_indices)} < {sequence_length}'
            }
        
        # Берем последнюю последовательность
        if len(feature_array) >= sequence_length:
            sequence = feature_array[-sequence_length:]
        else:
            return {
                'success': False,
                'error': f'Недостаточно кадров в видео'
            }
        
        # Заполняем пропуски
        sequence = self._interpolate_missing(sequence)
        
        # Предсказание
        predicted_class, probabilities = self.predict_sequence(sequence)
        
        # Формируем результат
        result = {
            'success': True,
            'video_name': poses_data.get('video_name', 'unknown'),
            'predicted_class': predicted_class,
            'confidence': float(probabilities[np.argmax(probabilities)]),
            'probabilities': {
                self.label_encoder.classes_[i]: float(prob)
                for i, prob in enumerate(probabilities)
            }
        }
        
        return result
    
    def _interpolate_missing(self, sequence):
        """Заполняет пропущенные значения"""
        sequence = sequence.copy()
        
        for feature_idx in range(sequence.shape[1]):
            feature_values = sequence[:, feature_idx]
            valid_mask = ~np.isnan(feature_values)
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                sequence[:, feature_idx] = 0.0
                continue
            
            if len(valid_indices) < len(feature_values):
                invalid_indices = np.where(~valid_mask)[0]
                sequence[invalid_indices, feature_idx] = np.interp(
                    invalid_indices, valid_indices, feature_values[valid_indices]
                )
        
        return sequence
    
    def evaluate(self, X_test, y_test, output_dir=None):
        """
        Оценивает качество модели на тестовой выборке
        
        Args:
            X_test: numpy array (num_samples, seq_len, num_features)
            y_test: numpy array (num_samples,)
            output_dir: директория для сохранения результатов
        
        Returns:
            dict: метрики качества
        """
        print("Оценка модели на тестовой выборке...")
        
        # Предсказания
        y_pred, y_prob = self.predict_batch(X_test)
        
        # Преобразуем обратно в индексы для метрик
        y_test_labels = self.label_encoder.classes_[y_test]
        
        # Метрики
        accuracy = accuracy_score(y_test_labels, y_pred)
        f1_macro = f1_score(y_test_labels, y_pred, average='macro')
        f1_weighted = f1_score(y_test_labels, y_pred, average='weighted')
        precision = precision_score(y_test_labels, y_pred, average='weighted')
        recall = recall_score(y_test_labels, y_pred, average='weighted')
        
        metrics = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'precision': float(precision),
            'recall': float(recall)
        }
        
        print(f"\nМетрики:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 (macro): {f1_macro:.4f}")
        print(f"F1 (weighted): {f1_weighted:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        # Classification report
        report = classification_report(
            y_test_labels, y_pred, 
            target_names=self.label_encoder.classes_,
            digits=4
        )
        print(f"\nClassification Report:\n{report}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test_labels, y_pred, labels=self.label_encoder.classes_)
        
        # Сохраняем результаты
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Сохраняем метрики
            with open(output_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Сохраняем classification report
            with open(output_dir / 'classification_report.txt', 'w') as f:
                f.write(report)
            
            # Визуализация confusion matrix
            self.plot_confusion_matrix(
                cm, self.label_encoder.classes_,
                save_path=output_dir / 'confusion_matrix.png'
            )
            
            # Сохраняем предсказания
            predictions_df = {
                'true_label': y_test_labels.tolist(),
                'predicted_label': y_pred.tolist(),
                'correct': (y_test_labels == y_pred).tolist()
            }
            with open(output_dir / 'predictions.json', 'w') as f:
                json.dump(predictions_df, f, indent=2)
            
            print(f"\nРезультаты сохранены в: {output_dir}")
        
        return metrics, cm
    
    def plot_confusion_matrix(self, cm, class_names, save_path=None):
        """Визуализирует confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        # Нормализуем для процентов
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Процент'}
        )
        
        plt.title('Confusion Matrix (нормализованная)')
        plt.ylabel('Истинный класс')
        plt.xlabel('Предсказанный класс')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix сохранена: {save_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Инференс классификатора танцевальных фигур")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Путь к сохраненной модели')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Директория с датасетом')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Директория для сохранения результатов')
    parser.add_argument('--video', type=str, default=None,
                        help='Путь к JSON файлу с позами для предсказания')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Устройство (cuda или cpu)')
    
    args = parser.parse_args()
    
    # Пути к файлам
    data_dir = Path(args.data_dir)
    metadata_path = data_dir / 'metadata.json'
    scaler_path = data_dir / 'scaler.pkl'
    label_encoder_path = data_dir / 'label_encoder.pkl'
    
    # Создаем predictor
    predictor = DanceClassifierPredictor(
        model_path=args.model_path,
        metadata_path=metadata_path,
        scaler_path=scaler_path,
        label_encoder_path=label_encoder_path,
        device=args.device
    )
    
    # Если указано видео, предсказываем для него
    if args.video:
        result = predictor.predict_from_poses(args.video)
        
        if result['success']:
            print(f"\nВидео: {result['video_name']}")
            print(f"Предсказанный класс: {result['predicted_class']}")
            print(f"Уверенность: {result['confidence']:.4f}")
            print(f"\nВероятности для всех классов:")
            for cls, prob in result['probabilities'].items():
                print(f"  {cls}: {prob:.4f}")
        else:
            print(f"Ошибка: {result['error']}")
    
    # Оценка на тестовой выборке
    else:
        X_test = np.load(data_dir / 'X_test.npy')
        y_test = np.load(data_dir / 'y_test.npy')
        
        print(f"Тестовая выборка: {X_test.shape}")
        
        metrics, cm = predictor.evaluate(
            X_test, y_test, 
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()


Скрипт инференса и оценки качества классификатора
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Добавляем пути к модулям
sys.path.append(str(Path(__file__).parent.parent))

from models.gru_classifier import create_gru_classifier
from models.tcn_classifier import create_tcn_classifier
from data_preparation.feature_extraction import FeatureExtractor


class DanceClassifierPredictor:
    """Класс для inference классификатора танцевальных фигур"""
    
    def __init__(self, model_path, metadata_path, scaler_path, 
                 label_encoder_path, device='cuda'):
        """
        Args:
            model_path: путь к сохраненной модели
            metadata_path: путь к метаданным датасета
            scaler_path: путь к scaler'у
            label_encoder_path: путь к label encoder'у
            device: устройство ('cuda' или 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Загружаем метаданные
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Загружаем scaler и label encoder
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Загружаем модель
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        
        # Создаем модель
        model_type = self.config.get('model_type', 'gru')
        model_config = self.config.get('model', {})
        
        if model_type == 'gru' or 'gru' in model_type.lower():
            self.model = create_gru_classifier(model_config)
        elif model_type == 'tcn' or 'tcn' in model_type.lower():
            self.model = create_tcn_classifier(model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.feature_extractor = FeatureExtractor(min_confidence=0.3)
        
        print(f"Модель загружена с устройства: {self.device}")
        print(f"Классы: {self.label_encoder.classes_}")
    
    def predict_sequence(self, sequence):
        """
        Предсказывает класс для одной последовательности
        
        Args:
            sequence: numpy array (seq_len, num_features)
        
        Returns:
            tuple: (predicted_class, probabilities)
        """
        # Нормализуем
        sequence_norm = self.scaler.transform(sequence)
        
        # Преобразуем в тензор
        x = torch.FloatTensor(sequence_norm).unsqueeze(0).to(self.device)
        
        # Инференс
        with torch.no_grad():
            outputs = self.model(x)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_idx = np.argmax(probabilities)
            predicted_class = self.label_encoder.classes_[predicted_idx]
        
        return predicted_class, probabilities
    
    def predict_batch(self, sequences):
        """
        Предсказывает классы для батча последовательностей
        
        Args:
            sequences: numpy array (batch_size, seq_len, num_features)
        
        Returns:
            tuple: (predicted_classes, all_probabilities)
        """
        # Нормализуем
        batch_size, seq_len, num_features = sequences.shape
        sequences_reshaped = sequences.reshape(-1, num_features)
        sequences_norm = self.scaler.transform(sequences_reshaped)
        sequences_norm = sequences_norm.reshape(batch_size, seq_len, num_features)
        
        # Преобразуем в тензор
        x = torch.FloatTensor(sequences_norm).to(self.device)
        
        # Инференс
        with torch.no_grad():
            outputs = self.model(x)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predicted_indices = np.argmax(probabilities, axis=1)
            predicted_classes = self.label_encoder.classes_[predicted_indices]
        
        return predicted_classes, probabilities
    
    def predict_from_poses(self, poses_json_path):
        """
        Предсказывает класс по JSON файлу с позами
        
        Args:
            poses_json_path: путь к JSON файлу с позами
        
        Returns:
            dict: результаты предсказания
        """
        # Загружаем позы
        with open(poses_json_path, 'r', encoding='utf-8') as f:
            poses_data = json.load(f)
        
        # Извлекаем признаки
        feature_list, valid_mask = self.feature_extractor.extract_sequence_features(poses_data)
        feature_array, feature_names, valid_indices = \
            self.feature_extractor.features_to_array(
                feature_list, self.metadata['feature_names']
            )
        
        # Проверяем длину последовательности
        sequence_length = self.metadata['sequence_length']
        if len(valid_indices) < sequence_length:
            return {
                'success': False,
                'error': f'Недостаточно валидных кадров: {len(valid_indices)} < {sequence_length}'
            }
        
        # Берем последнюю последовательность
        if len(feature_array) >= sequence_length:
            sequence = feature_array[-sequence_length:]
        else:
            return {
                'success': False,
                'error': f'Недостаточно кадров в видео'
            }
        
        # Заполняем пропуски
        sequence = self._interpolate_missing(sequence)
        
        # Предсказание
        predicted_class, probabilities = self.predict_sequence(sequence)
        
        # Формируем результат
        result = {
            'success': True,
            'video_name': poses_data.get('video_name', 'unknown'),
            'predicted_class': predicted_class,
            'confidence': float(probabilities[np.argmax(probabilities)]),
            'probabilities': {
                self.label_encoder.classes_[i]: float(prob)
                for i, prob in enumerate(probabilities)
            }
        }
        
        return result
    
    def _interpolate_missing(self, sequence):
        """Заполняет пропущенные значения"""
        sequence = sequence.copy()
        
        for feature_idx in range(sequence.shape[1]):
            feature_values = sequence[:, feature_idx]
            valid_mask = ~np.isnan(feature_values)
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                sequence[:, feature_idx] = 0.0
                continue
            
            if len(valid_indices) < len(feature_values):
                invalid_indices = np.where(~valid_mask)[0]
                sequence[invalid_indices, feature_idx] = np.interp(
                    invalid_indices, valid_indices, feature_values[valid_indices]
                )
        
        return sequence
    
    def evaluate(self, X_test, y_test, output_dir=None):
        """
        Оценивает качество модели на тестовой выборке
        
        Args:
            X_test: numpy array (num_samples, seq_len, num_features)
            y_test: numpy array (num_samples,)
            output_dir: директория для сохранения результатов
        
        Returns:
            dict: метрики качества
        """
        print("Оценка модели на тестовой выборке...")
        
        # Предсказания
        y_pred, y_prob = self.predict_batch(X_test)
        
        # Преобразуем обратно в индексы для метрик
        y_test_labels = self.label_encoder.classes_[y_test]
        
        # Метрики
        accuracy = accuracy_score(y_test_labels, y_pred)
        f1_macro = f1_score(y_test_labels, y_pred, average='macro')
        f1_weighted = f1_score(y_test_labels, y_pred, average='weighted')
        precision = precision_score(y_test_labels, y_pred, average='weighted')
        recall = recall_score(y_test_labels, y_pred, average='weighted')
        
        metrics = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'precision': float(precision),
            'recall': float(recall)
        }
        
        print(f"\nМетрики:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 (macro): {f1_macro:.4f}")
        print(f"F1 (weighted): {f1_weighted:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        # Classification report
        report = classification_report(
            y_test_labels, y_pred, 
            target_names=self.label_encoder.classes_,
            digits=4
        )
        print(f"\nClassification Report:\n{report}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test_labels, y_pred, labels=self.label_encoder.classes_)
        
        # Сохраняем результаты
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Сохраняем метрики
            with open(output_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Сохраняем classification report
            with open(output_dir / 'classification_report.txt', 'w') as f:
                f.write(report)
            
            # Визуализация confusion matrix
            self.plot_confusion_matrix(
                cm, self.label_encoder.classes_,
                save_path=output_dir / 'confusion_matrix.png'
            )
            
            # Сохраняем предсказания
            predictions_df = {
                'true_label': y_test_labels.tolist(),
                'predicted_label': y_pred.tolist(),
                'correct': (y_test_labels == y_pred).tolist()
            }
            with open(output_dir / 'predictions.json', 'w') as f:
                json.dump(predictions_df, f, indent=2)
            
            print(f"\nРезультаты сохранены в: {output_dir}")
        
        return metrics, cm
    
    def plot_confusion_matrix(self, cm, class_names, save_path=None):
        """Визуализирует confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        # Нормализуем для процентов
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Процент'}
        )
        
        plt.title('Confusion Matrix (нормализованная)')
        plt.ylabel('Истинный класс')
        plt.xlabel('Предсказанный класс')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix сохранена: {save_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Инференс классификатора танцевальных фигур")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Путь к сохраненной модели')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Директория с датасетом')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Директория для сохранения результатов')
    parser.add_argument('--video', type=str, default=None,
                        help='Путь к JSON файлу с позами для предсказания')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Устройство (cuda или cpu)')
    
    args = parser.parse_args()
    
    # Пути к файлам
    data_dir = Path(args.data_dir)
    metadata_path = data_dir / 'metadata.json'
    scaler_path = data_dir / 'scaler.pkl'
    label_encoder_path = data_dir / 'label_encoder.pkl'
    
    # Создаем predictor
    predictor = DanceClassifierPredictor(
        model_path=args.model_path,
        metadata_path=metadata_path,
        scaler_path=scaler_path,
        label_encoder_path=label_encoder_path,
        device=args.device
    )
    
    # Если указано видео, предсказываем для него
    if args.video:
        result = predictor.predict_from_poses(args.video)
        
        if result['success']:
            print(f"\nВидео: {result['video_name']}")
            print(f"Предсказанный класс: {result['predicted_class']}")
            print(f"Уверенность: {result['confidence']:.4f}")
            print(f"\nВероятности для всех классов:")
            for cls, prob in result['probabilities'].items():
                print(f"  {cls}: {prob:.4f}")
        else:
            print(f"Ошибка: {result['error']}")
    
    # Оценка на тестовой выборке
    else:
        X_test = np.load(data_dir / 'X_test.npy')
        y_test = np.load(data_dir / 'y_test.npy')
        
        print(f"Тестовая выборка: {X_test.shape}")
        
        metrics, cm = predictor.evaluate(
            X_test, y_test, 
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()


