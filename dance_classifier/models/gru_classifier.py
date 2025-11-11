"""
GRU-based классификатор танцевальных фигур
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUClassifier(nn.Module):
    """
    Классификатор на основе GRU (Gated Recurrent Unit)
    
    Архитектура:
    - Входной слой
    - GRU слои (можно несколько)
    - Dropout для регуляризации
    - Полносвязные слои
    - Выходной слой с softmax
    """
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, 
                 dropout=0.3, bidirectional=False):
        """
        Args:
            input_size: размерность входных признаков
            hidden_size: размерность скрытого состояния GRU
            num_layers: количество слоев GRU
            num_classes: количество классов для классификации
            dropout: вероятность dropout (для регуляризации)
            bidirectional: использовать ли двунаправленный GRU
        """
        super(GRUClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # GRU слои
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout для регуляризации
        self.dropout = nn.Dropout(dropout)
        
        # Размерность после GRU
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Полносвязные слои
        self.fc1 = nn.Linear(gru_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: тензор размера (batch_size, sequence_length, input_size)
        
        Returns:
            тензор размера (batch_size, num_classes)
        """
        # GRU
        # output: (batch_size, seq_len, hidden_size * num_directions)
        # h_n: (num_layers * num_directions, batch_size, hidden_size)
        gru_out, h_n = self.gru(x)
        
        # Берем последний выход
        if self.bidirectional:
            # Конкатенируем forward и backward
            h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h_n = h_n[-1]
        
        # Полносвязные слои
        out = self.dropout(h_n)
        out = self.fc1(out)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
    def get_embeddings(self, x):
        """
        Получить embeddings (представления) перед классификацией
        
        Args:
            x: тензор размера (batch_size, sequence_length, input_size)
        
        Returns:
            тензор размера (batch_size, hidden_size)
        """
        with torch.no_grad():
            gru_out, h_n = self.gru(x)
            
            if self.bidirectional:
                h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)
            else:
                h_n = h_n[-1]
            
            out = self.fc1(h_n)
            out = F.relu(out)
        
        return out


class AttentionGRUClassifier(nn.Module):
    """
    GRU классификатор с механизмом внимания (attention)
    
    Attention позволяет модели фокусироваться на важных частях последовательности
    """
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, 
                 dropout=0.3, bidirectional=False):
        """
        Args:
            input_size: размерность входных признаков
            hidden_size: размерность скрытого состояния GRU
            num_layers: количество слоев GRU
            num_classes: количество классов для классификации
            dropout: вероятность dropout
            bidirectional: использовать ли двунаправленный GRU
        """
        super(AttentionGRUClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        # GRU слои
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention mechanism
        self.attention = nn.Linear(gru_output_size, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Классификационные слои
        self.fc1 = nn.Linear(gru_output_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def attention_net(self, gru_output):
        """
        Применяет механизм внимания к выходу GRU
        
        Args:
            gru_output: (batch_size, seq_len, hidden_size * num_directions)
        
        Returns:
            context_vector: (batch_size, hidden_size * num_directions)
            attention_weights: (batch_size, seq_len)
        """
        # Вычисляем веса внимания
        # attention_scores: (batch_size, seq_len, 1)
        attention_scores = self.attention(gru_output)
        
        # Применяем softmax для получения весов
        # attention_weights: (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Взвешенная сумма
        # context_vector: (batch_size, hidden_size * num_directions)
        context_vector = torch.sum(attention_weights * gru_output, dim=1)
        
        return context_vector, attention_weights.squeeze(-1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: тензор размера (batch_size, sequence_length, input_size)
        
        Returns:
            тензор размера (batch_size, num_classes)
        """
        # GRU
        gru_out, _ = self.gru(x)
        
        # Attention
        context_vector, attention_weights = self.attention_net(gru_out)
        
        # Классификация
        out = self.dropout(context_vector)
        out = self.fc1(out)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
    def forward_with_attention(self, x):
        """
        Forward pass с возвратом весов внимания (для визуализации)
        
        Args:
            x: тензор размера (batch_size, sequence_length, input_size)
        
        Returns:
            tuple: (logits, attention_weights)
        """
        gru_out, _ = self.gru(x)
        context_vector, attention_weights = self.attention_net(gru_out)
        
        out = self.dropout(context_vector)
        out = self.fc1(out)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out, attention_weights


def create_gru_classifier(config):
    """
    Создает GRU классификатор по конфигурации
    
    Args:
        config: словарь с параметрами модели
    
    Returns:
        модель классификатора
    """
    use_attention = config.get('use_attention', False)
    
    if use_attention:
        model = AttentionGRUClassifier(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            dropout=config.get('dropout', 0.3),
            bidirectional=config.get('bidirectional', False)
        )
    else:
        model = GRUClassifier(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            dropout=config.get('dropout', 0.3),
            bidirectional=config.get('bidirectional', False)
        )
    
    return model


if __name__ == "__main__":
    # Тестирование модели
    config = {
        'input_size': 8,
        'hidden_size': 64,
        'num_layers': 2,
        'num_classes': 5,
        'dropout': 0.3,
        'bidirectional': True,
        'use_attention': False
    }
    
    model = create_gru_classifier(config)
    
    # Тестовый вход
    batch_size = 4
    seq_length = 30
    x = torch.randn(batch_size, seq_length, config['input_size'])
    
    # Forward pass
    output = model(x)
    
    print("Модель:", model.__class__.__name__)
    print("Вход:", x.shape)
    print("Выход:", output.shape)
    print("Количество параметров:", sum(p.numel() for p in model.parameters()))
    
    # Тест с attention
    config['use_attention'] = True
    model_att = create_gru_classifier(config)
    output_att = model_att(x)
    
    print("\nМодель с attention:", model_att.__class__.__name__)
    print("Выход:", output_att.shape)
    print("Количество параметров:", sum(p.numel() for p in model_att.parameters()))


GRU-based классификатор танцевальных фигур
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUClassifier(nn.Module):
    """
    Классификатор на основе GRU (Gated Recurrent Unit)
    
    Архитектура:
    - Входной слой
    - GRU слои (можно несколько)
    - Dropout для регуляризации
    - Полносвязные слои
    - Выходной слой с softmax
    """
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, 
                 dropout=0.3, bidirectional=False):
        """
        Args:
            input_size: размерность входных признаков
            hidden_size: размерность скрытого состояния GRU
            num_layers: количество слоев GRU
            num_classes: количество классов для классификации
            dropout: вероятность dropout (для регуляризации)
            bidirectional: использовать ли двунаправленный GRU
        """
        super(GRUClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # GRU слои
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout для регуляризации
        self.dropout = nn.Dropout(dropout)
        
        # Размерность после GRU
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Полносвязные слои
        self.fc1 = nn.Linear(gru_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: тензор размера (batch_size, sequence_length, input_size)
        
        Returns:
            тензор размера (batch_size, num_classes)
        """
        # GRU
        # output: (batch_size, seq_len, hidden_size * num_directions)
        # h_n: (num_layers * num_directions, batch_size, hidden_size)
        gru_out, h_n = self.gru(x)
        
        # Берем последний выход
        if self.bidirectional:
            # Конкатенируем forward и backward
            h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h_n = h_n[-1]
        
        # Полносвязные слои
        out = self.dropout(h_n)
        out = self.fc1(out)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
    def get_embeddings(self, x):
        """
        Получить embeddings (представления) перед классификацией
        
        Args:
            x: тензор размера (batch_size, sequence_length, input_size)
        
        Returns:
            тензор размера (batch_size, hidden_size)
        """
        with torch.no_grad():
            gru_out, h_n = self.gru(x)
            
            if self.bidirectional:
                h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)
            else:
                h_n = h_n[-1]
            
            out = self.fc1(h_n)
            out = F.relu(out)
        
        return out


class AttentionGRUClassifier(nn.Module):
    """
    GRU классификатор с механизмом внимания (attention)
    
    Attention позволяет модели фокусироваться на важных частях последовательности
    """
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, 
                 dropout=0.3, bidirectional=False):
        """
        Args:
            input_size: размерность входных признаков
            hidden_size: размерность скрытого состояния GRU
            num_layers: количество слоев GRU
            num_classes: количество классов для классификации
            dropout: вероятность dropout
            bidirectional: использовать ли двунаправленный GRU
        """
        super(AttentionGRUClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        # GRU слои
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention mechanism
        self.attention = nn.Linear(gru_output_size, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Классификационные слои
        self.fc1 = nn.Linear(gru_output_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def attention_net(self, gru_output):
        """
        Применяет механизм внимания к выходу GRU
        
        Args:
            gru_output: (batch_size, seq_len, hidden_size * num_directions)
        
        Returns:
            context_vector: (batch_size, hidden_size * num_directions)
            attention_weights: (batch_size, seq_len)
        """
        # Вычисляем веса внимания
        # attention_scores: (batch_size, seq_len, 1)
        attention_scores = self.attention(gru_output)
        
        # Применяем softmax для получения весов
        # attention_weights: (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Взвешенная сумма
        # context_vector: (batch_size, hidden_size * num_directions)
        context_vector = torch.sum(attention_weights * gru_output, dim=1)
        
        return context_vector, attention_weights.squeeze(-1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: тензор размера (batch_size, sequence_length, input_size)
        
        Returns:
            тензор размера (batch_size, num_classes)
        """
        # GRU
        gru_out, _ = self.gru(x)
        
        # Attention
        context_vector, attention_weights = self.attention_net(gru_out)
        
        # Классификация
        out = self.dropout(context_vector)
        out = self.fc1(out)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
    def forward_with_attention(self, x):
        """
        Forward pass с возвратом весов внимания (для визуализации)
        
        Args:
            x: тензор размера (batch_size, sequence_length, input_size)
        
        Returns:
            tuple: (logits, attention_weights)
        """
        gru_out, _ = self.gru(x)
        context_vector, attention_weights = self.attention_net(gru_out)
        
        out = self.dropout(context_vector)
        out = self.fc1(out)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out, attention_weights


def create_gru_classifier(config):
    """
    Создает GRU классификатор по конфигурации
    
    Args:
        config: словарь с параметрами модели
    
    Returns:
        модель классификатора
    """
    use_attention = config.get('use_attention', False)
    
    if use_attention:
        model = AttentionGRUClassifier(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            dropout=config.get('dropout', 0.3),
            bidirectional=config.get('bidirectional', False)
        )
    else:
        model = GRUClassifier(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            dropout=config.get('dropout', 0.3),
            bidirectional=config.get('bidirectional', False)
        )
    
    return model


if __name__ == "__main__":
    # Тестирование модели
    config = {
        'input_size': 8,
        'hidden_size': 64,
        'num_layers': 2,
        'num_classes': 5,
        'dropout': 0.3,
        'bidirectional': True,
        'use_attention': False
    }
    
    model = create_gru_classifier(config)
    
    # Тестовый вход
    batch_size = 4
    seq_length = 30
    x = torch.randn(batch_size, seq_length, config['input_size'])
    
    # Forward pass
    output = model(x)
    
    print("Модель:", model.__class__.__name__)
    print("Вход:", x.shape)
    print("Выход:", output.shape)
    print("Количество параметров:", sum(p.numel() for p in model.parameters()))
    
    # Тест с attention
    config['use_attention'] = True
    model_att = create_gru_classifier(config)
    output_att = model_att(x)
    
    print("\nМодель с attention:", model_att.__class__.__name__)
    print("Выход:", output_att.shape)
    print("Количество параметров:", sum(p.numel() for p in model_att.parameters()))


