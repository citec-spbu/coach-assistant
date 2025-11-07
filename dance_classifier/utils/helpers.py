"""
Вспомогательные функции для классификатора танцевальных фигур
"""
import numpy as np
import torch
import random
import json
from pathlib import Path


def set_seed(seed=42):
    """
    Устанавливает seed для воспроизводимости результатов
    
    Args:
        seed: значение seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    Подсчитывает количество обучаемых параметров модели
    
    Args:
        model: PyTorch модель
    
    Returns:
        int: количество параметров
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config(config, save_path):
    """
    Сохраняет конфигурацию в JSON файл
    
    Args:
        config: словарь с конфигурацией
        save_path: путь для сохранения
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_config(config_path):
    """
    Загружает конфигурацию из JSON файла
    
    Args:
        config_path: путь к файлу конфигурации
    
    Returns:
        dict: конфигурация
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_time(seconds):
    """
    Форматирует время в читаемый вид
    
    Args:
        seconds: время в секундах
    
    Returns:
        str: отформатированное время
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class AverageMeter:
    """Вычисляет и хранит среднее и текущее значение"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_label_mapping(video_names):
    """
    Создает маппинг меток из имен видео файлов
    
    Предполагается, что имена файлов содержат метки класса
    Например: "basic_step_001.mp4" -> "basic_step"
    
    Args:
        video_names: список имен видео файлов
    
    Returns:
        dict: словарь {video_name: label}
    """
    label_mapping = {}
    
    for video_name in video_names:
        # Убираем расширение
        name_without_ext = Path(video_name).stem
        
        # Извлекаем метку (всё кроме числовой части в конце)
        parts = name_without_ext.split('_')
        
        # Убираем числовые части
        label_parts = [p for p in parts if not p.isdigit()]
        
        if len(label_parts) > 0:
            label = '_'.join(label_parts).lower()
        else:
            label = 'unknown'
        
        label_mapping[video_name] = label
    
    return label_mapping


def interpolate_sequence(sequence, method='linear'):
    """
    Интерполирует пропущенные значения в последовательности
    
    Args:
        sequence: numpy array (seq_len, num_features) с возможными NaN
        method: метод интерполяции ('linear', 'nearest', 'zero')
    
    Returns:
        numpy array: последовательность с заполненными значениями
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
            invalid_indices = np.where(~valid_mask)[0]
            
            if method == 'linear':
                # Линейная интерполяция
                sequence[invalid_indices, feature_idx] = np.interp(
                    invalid_indices,
                    valid_indices,
                    feature_values[valid_indices]
                )
            elif method == 'nearest':
                # Ближайший сосед
                for idx in invalid_indices:
                    distances = np.abs(valid_indices - idx)
                    nearest_idx = valid_indices[np.argmin(distances)]
                    sequence[idx, feature_idx] = feature_values[nearest_idx]
            elif method == 'zero':
                # Заполнение нулями
                sequence[invalid_indices, feature_idx] = 0.0
    
    return sequence


def validate_sequence(sequence, max_nan_ratio=0.3):
    """
    Проверяет качество последовательности
    
    Args:
        sequence: numpy array (seq_len, num_features)
        max_nan_ratio: максимальная доля NaN значений
    
    Returns:
        bool: True если последовательность валидна
    """
    if sequence is None or len(sequence) == 0:
        return False
    
    # Проверяем долю NaN
    nan_ratio = np.isnan(sequence).sum() / sequence.size
    
    return nan_ratio <= max_nan_ratio


def print_model_summary(model, input_shape):
    """
    Выводит сводку о модели
    
    Args:
        model: PyTorch модель
        input_shape: форма входного тензора (без batch dimension)
    """
    print("=" * 70)
    print(f"Модель: {model.__class__.__name__}")
    print("=" * 70)
    
    # Количество параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Всего параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")
    print(f"Необучаемых параметров: {total_params - trainable_params:,}")
    
    # Размер модели
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    
    print(f"Размер модели: {size_mb:.2f} MB")
    
    # Форма входа/выхода
    print(f"Форма входа: {input_shape}")
    
    # Пробный forward pass
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Форма выхода: {tuple(output.shape[1:])}")
    except Exception as e:
        print(f"Не удалось вычислить форму выхода: {e}")
    
    print("=" * 70)


if __name__ == "__main__":
    # Примеры использования
    print("Вспомогательные функции для классификатора танцевальных фигур")
    
    # Тест форматирования времени
    print(f"\nТест format_time:")
    print(f"100 сек: {format_time(100)}")
    print(f"3661 сек: {format_time(3661)}")
    
    # Тест создания меток
    print(f"\nТест create_label_mapping:")
    video_names = ["basic_step_001.mp4", "spin_turn_002.mp4", "natural_turn_003.mp4"]
    labels = create_label_mapping(video_names)
    for name, label in labels.items():
        print(f"{name} -> {label}")


Вспомогательные функции для классификатора танцевальных фигур
"""
import numpy as np
import torch
import random
import json
from pathlib import Path


def set_seed(seed=42):
    """
    Устанавливает seed для воспроизводимости результатов
    
    Args:
        seed: значение seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    Подсчитывает количество обучаемых параметров модели
    
    Args:
        model: PyTorch модель
    
    Returns:
        int: количество параметров
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config(config, save_path):
    """
    Сохраняет конфигурацию в JSON файл
    
    Args:
        config: словарь с конфигурацией
        save_path: путь для сохранения
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_config(config_path):
    """
    Загружает конфигурацию из JSON файла
    
    Args:
        config_path: путь к файлу конфигурации
    
    Returns:
        dict: конфигурация
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_time(seconds):
    """
    Форматирует время в читаемый вид
    
    Args:
        seconds: время в секундах
    
    Returns:
        str: отформатированное время
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class AverageMeter:
    """Вычисляет и хранит среднее и текущее значение"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_label_mapping(video_names):
    """
    Создает маппинг меток из имен видео файлов
    
    Предполагается, что имена файлов содержат метки класса
    Например: "basic_step_001.mp4" -> "basic_step"
    
    Args:
        video_names: список имен видео файлов
    
    Returns:
        dict: словарь {video_name: label}
    """
    label_mapping = {}
    
    for video_name in video_names:
        # Убираем расширение
        name_without_ext = Path(video_name).stem
        
        # Извлекаем метку (всё кроме числовой части в конце)
        parts = name_without_ext.split('_')
        
        # Убираем числовые части
        label_parts = [p for p in parts if not p.isdigit()]
        
        if len(label_parts) > 0:
            label = '_'.join(label_parts).lower()
        else:
            label = 'unknown'
        
        label_mapping[video_name] = label
    
    return label_mapping


def interpolate_sequence(sequence, method='linear'):
    """
    Интерполирует пропущенные значения в последовательности
    
    Args:
        sequence: numpy array (seq_len, num_features) с возможными NaN
        method: метод интерполяции ('linear', 'nearest', 'zero')
    
    Returns:
        numpy array: последовательность с заполненными значениями
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
            invalid_indices = np.where(~valid_mask)[0]
            
            if method == 'linear':
                # Линейная интерполяция
                sequence[invalid_indices, feature_idx] = np.interp(
                    invalid_indices,
                    valid_indices,
                    feature_values[valid_indices]
                )
            elif method == 'nearest':
                # Ближайший сосед
                for idx in invalid_indices:
                    distances = np.abs(valid_indices - idx)
                    nearest_idx = valid_indices[np.argmin(distances)]
                    sequence[idx, feature_idx] = feature_values[nearest_idx]
            elif method == 'zero':
                # Заполнение нулями
                sequence[invalid_indices, feature_idx] = 0.0
    
    return sequence


def validate_sequence(sequence, max_nan_ratio=0.3):
    """
    Проверяет качество последовательности
    
    Args:
        sequence: numpy array (seq_len, num_features)
        max_nan_ratio: максимальная доля NaN значений
    
    Returns:
        bool: True если последовательность валидна
    """
    if sequence is None or len(sequence) == 0:
        return False
    
    # Проверяем долю NaN
    nan_ratio = np.isnan(sequence).sum() / sequence.size
    
    return nan_ratio <= max_nan_ratio


def print_model_summary(model, input_shape):
    """
    Выводит сводку о модели
    
    Args:
        model: PyTorch модель
        input_shape: форма входного тензора (без batch dimension)
    """
    print("=" * 70)
    print(f"Модель: {model.__class__.__name__}")
    print("=" * 70)
    
    # Количество параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Всего параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")
    print(f"Необучаемых параметров: {total_params - trainable_params:,}")
    
    # Размер модели
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    
    print(f"Размер модели: {size_mb:.2f} MB")
    
    # Форма входа/выхода
    print(f"Форма входа: {input_shape}")
    
    # Пробный forward pass
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Форма выхода: {tuple(output.shape[1:])}")
    except Exception as e:
        print(f"Не удалось вычислить форму выхода: {e}")
    
    print("=" * 70)


if __name__ == "__main__":
    # Примеры использования
    print("Вспомогательные функции для классификатора танцевальных фигур")
    
    # Тест форматирования времени
    print(f"\nТест format_time:")
    print(f"100 сек: {format_time(100)}")
    print(f"3661 сек: {format_time(3661)}")
    
    # Тест создания меток
    print(f"\nТест create_label_mapping:")
    video_names = ["basic_step_001.mp4", "spin_turn_002.mp4", "natural_turn_003.mp4"]
    labels = create_label_mapping(video_names)
    for name, label in labels.items():
        print(f"{name} -> {label}")


