"""
Скрипт обучения классификатора танцевальных фигур
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import yaml
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
import json
import sys

# Добавляем путь к корню проекта
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dance_classifier.inference.predict import GRUModel, HybridModel
from dance_classifier.data_preparation.dataset_builder import DanceDatasetBuilder


def load_config(config_path):
    """Загружает конфигурацию из YAML файла"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_model(model_type, input_size, num_classes, config):
    """Создает модель по типу"""
    if model_type == 'gru':
        model = GRUModel(input_size, num_classes)
    elif model_type == 'hybrid':
        model = HybridModel(input_size, num_classes)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
    return model


def load_pretrained_checkpoint(model, checkpoint_path, device):
    """Загружает pretrained checkpoint для fine-tuning"""
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"[INFO] Загружаем pretrained checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Если checkpoint содержит 'model_state_dict', используем его
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            # Пытаемся загрузить напрямую
            model.load_state_dict(checkpoint, strict=False)
        
        print("[OK] Pretrained checkpoint загружен")
        return True
    return False


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Одна эпоха обучения"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for sequences, labels in tqdm(dataloader, desc="Training"):
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Валидация"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc="Validation"):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def main():
    # Пути
    config_path = Path(__file__).parent / "config.yaml"
    # Путь к CSV датасету (абсолютный путь)
    base_dir = Path(__file__).parent.parent.parent.parent.parent  # Поднимаемся до !PYTHON_DZ
    csv_dataset_path = base_dir / "coach-assistant" / "датасет" / "csv_ver02" / "csv" / "KeypointData_woScl_noCut.pkl"
    csv_dataset_path_csv = base_dir / "coach-assistant" / "датасет" / "csv_ver02" / "csv" / "KeypointData_woScl_noCut.csv"
    output_dir = Path(__file__).parent.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Для дообучения используем тот же random_state=42, что и при обучении best_model_20pct
    dataset_sample_frac = 0.2  # 20% данных
    dataset_random_state = 42
    
    # Загружаем конфигурацию
    config = load_config(config_path)
    print(f"[INFO] Конфигурация загружена: {config_path}")
    
    # Параметры
    model_type = config.get('model_type', 'gru')
    batch_size = config.get('batch_size', 16)
    num_epochs = config.get('num_epochs', 50)
    learning_rate = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 0.0001)
    pretrained_checkpoint = config.get('pretrained_checkpoint', None)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Устройство: {device}")
    
    # Загружаем данные из CSV/PKL
    import pandas as pd
    
    if csv_dataset_path.exists():
        print(f"[INFO] Загружаем датасет из: {csv_dataset_path}")
        df = pd.read_pickle(csv_dataset_path)
    elif csv_dataset_path_csv.exists():
        print(f"[INFO] Загружаем датасет из: {csv_dataset_path_csv}")
        df = pd.read_csv(csv_dataset_path_csv)
    else:
        print(f"[ERROR] Датасет не найден: {csv_dataset_path} или {csv_dataset_path_csv}")
        return
    
    print(f"[INFO] Загружено записей: {len(df)}")
    
    # Выбираем 20% для дообучения (тот же random_state=42, что и при обучении best_model_20pct)
    df_sample = df.sample(frac=dataset_sample_frac, random_state=dataset_random_state)
    print(f"[INFO] Используем {dataset_sample_frac*100}% данных (random_state={dataset_random_state}): {len(df_sample)} записей")
    
    # Подготовка данных
    feature_cols = [col for col in df_sample.columns if col not in ['figure', 'video_name', 'frame_num']]
    X = df_sample[feature_cols].values
    y = df_sample['figure'].values
    
    print(f"[INFO] Признаков: {X.shape[1]}")
    print(f"[INFO] Классов: {len(np.unique(y))}")
    
    # Кодирование меток
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"[INFO] Классы: {list(label_encoder.classes_)}")
    
    # Показываем распределение классов
    from collections import Counter
    class_counts = Counter(y)
    print(f"\n[INFO] Распределение классов в датасете:")
    for cls_name in label_encoder.classes_:
        count = class_counts.get(cls_name, 0)
        pct = 100 * count / len(y) if len(y) > 0 else 0
        print(f"  {cls_name:20s}: {count:6d} ({pct:5.1f}%)")
    
    # Разделение на train/val
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    
    print(f"[INFO] Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Вычисляем веса классов для балансировки
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(zip(np.unique(y_train), class_weights))
    print(f"\n[INFO] Веса классов для балансировки:")
    for cls_idx, weight in class_weights_dict.items():
        cls_name = label_encoder.classes_[cls_idx]
        print(f"  {cls_name:20s}: {weight:.3f}")
    
    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Создание последовательностей
    sequence_length = 30
    def create_sequences(X, y, seq_len):
        X_seq = []
        y_seq = []
        for i in range(len(X) - seq_len + 1):
            X_seq.append(X[i:i+seq_len])
            y_seq.append(y[i+seq_len-1])
        return np.array(X_seq), np.array(y_seq)
    
    print(f"[INFO] Создаём последовательности (length={sequence_length})...")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, sequence_length)
    
    print(f"[INFO] Обучающих последовательностей: {len(X_train_seq)}")
    print(f"[INFO] Валидационных последовательностей: {len(X_val_seq)}")
    
    input_size = X_train_seq.shape[-1]
    num_classes = len(label_encoder.classes_)
    
    # Преобразуем в тензоры
    sequences_train = torch.FloatTensor(X_train_seq)
    labels_train = torch.LongTensor(y_train_seq)
    sequences_val = torch.FloatTensor(X_val_seq)
    labels_val = torch.LongTensor(y_val_seq)
    
    train_dataset = TensorDataset(sequences_train, labels_train)
    val_dataset = TensorDataset(sequences_val, labels_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"[INFO] Обучающих примеров: {len(train_dataset)}")
    print(f"[INFO] Валидационных примеров: {len(val_dataset)}")
    
    # Создаем модель
    model = create_model(model_type, input_size, num_classes, config)
    model = model.to(device)
    
    # Loss с взвешиванием классов для балансировки
    # Создаем тензор весов для loss function
    weights_tensor = torch.FloatTensor([class_weights_dict[i] for i in range(len(label_encoder.classes_))]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    print(f"[INFO] Используется взвешенный CrossEntropyLoss для балансировки классов")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Scheduler
    scheduler = None
    if config.get('scheduler') == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    elif config.get('scheduler') == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config.get('step_size', 10), 
            gamma=config.get('gamma', 0.5)
        )
    
    # Проверяем, есть ли промежуточные checkpoint'ы для продолжения обучения
    start_epoch = 1
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = config.get('early_stopping_patience', 15)
    
    # Ищем последний checkpoint_epoch_*.pth
    checkpoint_files = list(output_dir.glob("checkpoint_epoch_*.pth"))
    if checkpoint_files:
        # Находим последний checkpoint по номеру эпохи
        def get_epoch_num(path):
            try:
                return int(path.stem.split('_')[-1])
            except:
                return 0
        
        last_checkpoint = max(checkpoint_files, key=get_epoch_num)
        last_epoch = get_epoch_num(last_checkpoint)
        
        print(f"[INFO] Найден checkpoint: {last_checkpoint} (эпоха {last_epoch})")
        print(f"[INFO] Продолжаем обучение с эпохи {last_epoch + 1}")
        
        # Загружаем checkpoint
        checkpoint = torch.load(last_checkpoint, map_location=device)
        
        # Восстанавливаем состояние модели
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        
        # Восстанавливаем состояние optimizer
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Восстанавливаем scheduler state (если есть)
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Восстанавливаем метрики
        start_epoch = last_epoch + 1
        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']
        if 'patience_counter' in checkpoint:
            patience_counter = checkpoint['patience_counter']
        
        print(f"[INFO] Восстановлено: best_val_loss={best_val_loss:.4f}, patience_counter={patience_counter}")
    else:
        # Нет промежуточных checkpoint'ов - начинаем с pretrained модели
        if pretrained_checkpoint:
            # Пробуем найти checkpoint в разных местах
            checkpoint_paths = [
                Path(__file__).parent.parent / pretrained_checkpoint,
                Path(__file__).parent.parent.parent / "dance_classifier" / pretrained_checkpoint,
                Path(__file__).parent.parent.parent.parent / "coach-assistant" / "dance_classifier" / pretrained_checkpoint
            ]
            
            checkpoint_path = None
            for cp in checkpoint_paths:
                if cp.exists():
                    checkpoint_path = cp
                    break
            
            if checkpoint_path and checkpoint_path.exists():
                load_pretrained_checkpoint(model, str(checkpoint_path), device)
                # Для fine-tuning уменьшаем learning rate
                learning_rate = learning_rate / 10.0
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                print(f"[INFO] Fine-tuning mode: learning_rate = {learning_rate}")
            else:
                print(f"[WARN] Pretrained checkpoint не найден: {pretrained_checkpoint}")
                print("[INFO] Обучение будет с нуля")
    
    # Обучение
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n[EPOCH {epoch}/{num_epochs}]")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Scheduler step
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'scaler': scaler,
                'label_encoder': label_encoder,
                'input_size': input_size,
                'num_classes': num_classes,
                'model_type': model_type
            }
            
            best_model_path = output_dir / "best_model_20pct_finetuned.pth"
            torch.save(checkpoint, best_model_path)
            print(f"[OK] Сохранена лучшая модель: {best_model_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"[INFO] Early stopping после {epoch} эпох")
            break
        
        # Периодическое сохранение
        if config.get('save_every') and epoch % config.get('save_every') == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
            # Сохраняем полный checkpoint для продолжения обучения
            full_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'scaler': scaler,
                'label_encoder': label_encoder,
                'input_size': input_size,
                'num_classes': num_classes,
                'model_type': model_type
            }
            if scheduler:
                full_checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(full_checkpoint, checkpoint_path)
            print(f"[OK] Сохранен checkpoint: {checkpoint_path}")
    
    print(f"\n[OK] Обучение завершено. Лучшая валидационная loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

