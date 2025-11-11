"""
ПРОСТОЙ СПОСОБ ИСПОЛЬЗОВАТЬ КЛАССИФИКАТОР

Просто запустите:
python use_classifier.py путь/к/poses.jsonl

Или импортируйте в своём коде:
from use_classifier import classify_video
result = classify_video("outputs/video1/poses.jsonl")
"""

import torch
import numpy as np
import json
import sys
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Разрешить загрузку sklearn классов (для PyTorch 2.6+)
torch.serialization.add_safe_globals([LabelEncoder, StandardScaler])


class GRUModel(torch.nn.Module):
    """Простая GRU модель"""
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.gru = torch.nn.GRU(input_size, 64, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = torch.nn.Linear(64, num_classes)
    
    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])


def classify_video(poses_file, model_path="best_model_10pct.pth"):
    """
    Классифицировать танцевальное движение из poses.jsonl
    
    Args:
        poses_file: путь к poses.jsonl
        model_path: путь к модели (по умолчанию best_model_10pct.pth)
    
    Returns:
        dict с результатами:
        {
            'success': True/False,
            'predicted_figure': 'Fan',
            'confidence': 0.63,
            'error': 'описание ошибки' (если success=False)
        }
    """
    try:
        # 1. Загрузка модели
        checkpoint = torch.load(model_path, weights_only=False)
        
        scaler = checkpoint['scaler']
        label_encoder = checkpoint['label_encoder']
        sequence_length = checkpoint['sequence_length']
        
        model = GRUModel(
            input_size=checkpoint['input_size'],
            num_classes=checkpoint['num_classes']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 2. Чтение поз
        poses = []
        with open(poses_file, 'r') as f:
            for line in f:
                pose = json.loads(line)
                if pose['valid']:
                    coords = []
                    for kp in pose['keypoints']:
                        coords.append(kp[0])
                        coords.append(kp[1])
                    poses.append(coords)
        
        if len(poses) < sequence_length:
            return {
                'success': False,
                'error': f'Not enough frames: {len(poses)} < {sequence_length}'
            }
        
        poses = np.array(poses)
        
        # 3. Создание последовательностей
        sequences = []
        for i in range(0, len(poses) - sequence_length + 1, sequence_length // 2):
            seq = poses[i:i + sequence_length]
            if len(seq) == sequence_length:
                sequences.append(seq)
        
        X = np.array(sequences)
        
        # 4. Дополнение до 50 координат (если нужно)
        if X.shape[2] != 50:
            padding = np.zeros((X.shape[0], X.shape[1], 50 - X.shape[2]))
            X = np.concatenate([X, padding], axis=2)
        
        # 5. Нормализация
        X_norm = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # 6. Предсказание
        X_tensor = torch.FloatTensor(X_norm)
        with torch.no_grad():
            outputs = model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
        
        predictions = label_encoder.inverse_transform(predicted.numpy())
        confidences = probs.numpy()
        
        # 7. Наиболее частое предсказание
        most_common = Counter(predictions).most_common(1)[0]
        avg_confidence = np.mean([confidences[i, predicted[i]] for i in range(len(predictions))])
        
        return {
            'success': True,
            'predicted_figure': most_common[0],
            'confidence': float(avg_confidence),
            'all_predictions': predictions.tolist()
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == '__main__':
    # Использование из командной строки
    if len(sys.argv) < 2:
        print("Использование: python use_classifier.py путь/к/poses.jsonl")
        print("\nПример:")
        print("  python use_classifier.py outputs/video1/poses.jsonl")
        sys.exit(1)
    
    poses_file = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "best_model_10pct.pth"
    
    print(f"Классификация: {poses_file}")
    print(f"Модель: {model_path}")
    print("-" * 60)
    
    result = classify_video(poses_file, model_path)
    
    if result['success']:
        print(f"\n✅ Движение: {result['predicted_figure']}")
        print(f"   Уверенность: {result['confidence']*100:.1f}%")
    else:
        print(f"\n❌ Ошибка: {result['error']}")

