"""
TCN (Temporal Convolutional Network) классификатор танцевальных фигур
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """Удаляет лишние элементы из конца для causal convolution"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Базовый блок TCN
    
    Состоит из:
    - Dilated causal convolution
    - Weight normalization
    - ReLU activation
    - Dropout
    - Residual connection
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, 
                 padding, dropout=0.2):
        """
        Args:
            n_inputs: количество входных каналов
            n_outputs: количество выходных каналов
            kernel_size: размер ядра свертки
            stride: шаг свертки
            dilation: dilation factor
            padding: padding
            dropout: вероятность dropout
        """
        super(TemporalBlock, self).__init__()
        
        # Первая свертка
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Вторая свертка
        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        self.init_weights()

    def init_weights(self):
        """Инициализация весов"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        Args:
            x: (batch_size, n_inputs, seq_len)
        
        Returns:
            (batch_size, n_outputs, seq_len)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    TCN (Temporal Convolutional Network)
    
    Стек TemporalBlock'ов с увеличивающимся dilation
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        Args:
            num_inputs: размерность входных признаков
            num_channels: список с количеством каналов для каждого слоя
            kernel_size: размер ядра свертки
            dropout: вероятность dropout
        """
        super(TemporalConvNet, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size, 
                stride=1, dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size, 
                dropout=dropout
            )]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_inputs, seq_len)
        
        Returns:
            (batch_size, num_channels[-1], seq_len)
        """
        return self.network(x)


class TCNClassifier(nn.Module):
    """
    Классификатор на основе TCN
    
    Архитектура:
    - TCN для извлечения темпоральных признаков
    - Global average pooling
    - Полносвязные слои для классификации
    """
    def __init__(self, input_size, num_channels, num_classes, 
                 kernel_size=3, dropout=0.2):
        """
        Args:
            input_size: размерность входных признаков
            num_channels: список с количеством каналов для каждого TCN слоя
            num_classes: количество классов
            kernel_size: размер ядра свертки
            dropout: вероятность dropout
        """
        super(TCNClassifier, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # TCN
        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Классификационные слои
        self.fc1 = nn.Linear(num_channels[-1], num_channels[-1] // 2)
        self.batch_norm = nn.BatchNorm1d(num_channels[-1] // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_channels[-1] // 2, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        
        Returns:
            (batch_size, num_classes)
        """
        # TCN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # TCN
        tcn_out = self.tcn(x)  # (batch, num_channels[-1], seq_len)
        
        # Global pooling
        pooled = self.global_pool(tcn_out)  # (batch, num_channels[-1], 1)
        pooled = pooled.squeeze(-1)  # (batch, num_channels[-1])
        
        # Classification
        out = self.fc1(pooled)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
    def get_embeddings(self, x):
        """
        Получить embeddings перед классификацией
        
        Args:
            x: (batch_size, seq_len, input_size)
        
        Returns:
            (batch_size, num_channels[-1] // 2)
        """
        with torch.no_grad():
            x = x.transpose(1, 2)
            tcn_out = self.tcn(x)
            pooled = self.global_pool(tcn_out).squeeze(-1)
            out = self.fc1(pooled)
            out = F.relu(out)
        return out


class HybridTCNClassifier(nn.Module):
    """
    Гибридный классификатор: TCN + GRU
    
    Комбинирует преимущества обоих подходов:
    - TCN для локальных темпоральных паттернов
    - GRU для долгосрочных зависимостей
    """
    def __init__(self, input_size, tcn_channels, gru_hidden_size, 
                 num_classes, kernel_size=3, dropout=0.2):
        """
        Args:
            input_size: размерность входных признаков
            tcn_channels: список каналов для TCN
            gru_hidden_size: размер скрытого состояния GRU
            num_classes: количество классов
            kernel_size: размер ядра для TCN
            dropout: вероятность dropout
        """
        super(HybridTCNClassifier, self).__init__()
        
        # TCN для извлечения локальных признаков
        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # GRU для моделирования последовательности
        self.gru = nn.GRU(
            input_size=tcn_channels[-1],
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # Классификация
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(gru_hidden_size, gru_hidden_size // 2)
        self.batch_norm = nn.BatchNorm1d(gru_hidden_size // 2)
        self.fc2 = nn.Linear(gru_hidden_size // 2, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        
        Returns:
            (batch_size, num_classes)
        """
        # TCN
        x_tcn = x.transpose(1, 2)  # (batch, input_size, seq_len)
        tcn_out = self.tcn(x_tcn)  # (batch, tcn_channels[-1], seq_len)
        tcn_out = tcn_out.transpose(1, 2)  # (batch, seq_len, tcn_channels[-1])
        
        # GRU
        gru_out, h_n = self.gru(tcn_out)
        h_n = h_n[-1]  # (batch, gru_hidden_size)
        
        # Classification
        out = self.dropout(h_n)
        out = self.fc1(out)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


def create_tcn_classifier(config):
    """
    Создает TCN классификатор по конфигурации
    
    Args:
        config: словарь с параметрами модели
    
    Returns:
        модель классификатора
    """
    model_type = config.get('model_type', 'tcn')
    
    if model_type == 'hybrid':
        model = HybridTCNClassifier(
            input_size=config['input_size'],
            tcn_channels=config['tcn_channels'],
            gru_hidden_size=config.get('gru_hidden_size', 64),
            num_classes=config['num_classes'],
            kernel_size=config.get('kernel_size', 3),
            dropout=config.get('dropout', 0.2)
        )
    else:
        model = TCNClassifier(
            input_size=config['input_size'],
            num_channels=config['tcn_channels'],
            num_classes=config['num_classes'],
            kernel_size=config.get('kernel_size', 3),
            dropout=config.get('dropout', 0.2)
        )
    
    return model


if __name__ == "__main__":
    # Тестирование модели
    config = {
        'input_size': 8,
        'tcn_channels': [32, 64, 64, 128],
        'num_classes': 5,
        'kernel_size': 3,
        'dropout': 0.2,
        'model_type': 'tcn'
    }
    
    model = create_tcn_classifier(config)
    
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
    
    # Тест гибридной модели
    config['model_type'] = 'hybrid'
    config['gru_hidden_size'] = 64
    model_hybrid = create_tcn_classifier(config)
    output_hybrid = model_hybrid(x)
    
    print("\nГибридная модель:", model_hybrid.__class__.__name__)
    print("Выход:", output_hybrid.shape)
    print("Количество параметров:", sum(p.numel() for p in model_hybrid.parameters()))


TCN (Temporal Convolutional Network) классификатор танцевальных фигур
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """Удаляет лишние элементы из конца для causal convolution"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Базовый блок TCN
    
    Состоит из:
    - Dilated causal convolution
    - Weight normalization
    - ReLU activation
    - Dropout
    - Residual connection
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, 
                 padding, dropout=0.2):
        """
        Args:
            n_inputs: количество входных каналов
            n_outputs: количество выходных каналов
            kernel_size: размер ядра свертки
            stride: шаг свертки
            dilation: dilation factor
            padding: padding
            dropout: вероятность dropout
        """
        super(TemporalBlock, self).__init__()
        
        # Первая свертка
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Вторая свертка
        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        self.init_weights()

    def init_weights(self):
        """Инициализация весов"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        Args:
            x: (batch_size, n_inputs, seq_len)
        
        Returns:
            (batch_size, n_outputs, seq_len)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    TCN (Temporal Convolutional Network)
    
    Стек TemporalBlock'ов с увеличивающимся dilation
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        Args:
            num_inputs: размерность входных признаков
            num_channels: список с количеством каналов для каждого слоя
            kernel_size: размер ядра свертки
            dropout: вероятность dropout
        """
        super(TemporalConvNet, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size, 
                stride=1, dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size, 
                dropout=dropout
            )]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_inputs, seq_len)
        
        Returns:
            (batch_size, num_channels[-1], seq_len)
        """
        return self.network(x)


class TCNClassifier(nn.Module):
    """
    Классификатор на основе TCN
    
    Архитектура:
    - TCN для извлечения темпоральных признаков
    - Global average pooling
    - Полносвязные слои для классификации
    """
    def __init__(self, input_size, num_channels, num_classes, 
                 kernel_size=3, dropout=0.2):
        """
        Args:
            input_size: размерность входных признаков
            num_channels: список с количеством каналов для каждого TCN слоя
            num_classes: количество классов
            kernel_size: размер ядра свертки
            dropout: вероятность dropout
        """
        super(TCNClassifier, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # TCN
        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Классификационные слои
        self.fc1 = nn.Linear(num_channels[-1], num_channels[-1] // 2)
        self.batch_norm = nn.BatchNorm1d(num_channels[-1] // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_channels[-1] // 2, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        
        Returns:
            (batch_size, num_classes)
        """
        # TCN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # TCN
        tcn_out = self.tcn(x)  # (batch, num_channels[-1], seq_len)
        
        # Global pooling
        pooled = self.global_pool(tcn_out)  # (batch, num_channels[-1], 1)
        pooled = pooled.squeeze(-1)  # (batch, num_channels[-1])
        
        # Classification
        out = self.fc1(pooled)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
    def get_embeddings(self, x):
        """
        Получить embeddings перед классификацией
        
        Args:
            x: (batch_size, seq_len, input_size)
        
        Returns:
            (batch_size, num_channels[-1] // 2)
        """
        with torch.no_grad():
            x = x.transpose(1, 2)
            tcn_out = self.tcn(x)
            pooled = self.global_pool(tcn_out).squeeze(-1)
            out = self.fc1(pooled)
            out = F.relu(out)
        return out


class HybridTCNClassifier(nn.Module):
    """
    Гибридный классификатор: TCN + GRU
    
    Комбинирует преимущества обоих подходов:
    - TCN для локальных темпоральных паттернов
    - GRU для долгосрочных зависимостей
    """
    def __init__(self, input_size, tcn_channels, gru_hidden_size, 
                 num_classes, kernel_size=3, dropout=0.2):
        """
        Args:
            input_size: размерность входных признаков
            tcn_channels: список каналов для TCN
            gru_hidden_size: размер скрытого состояния GRU
            num_classes: количество классов
            kernel_size: размер ядра для TCN
            dropout: вероятность dropout
        """
        super(HybridTCNClassifier, self).__init__()
        
        # TCN для извлечения локальных признаков
        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # GRU для моделирования последовательности
        self.gru = nn.GRU(
            input_size=tcn_channels[-1],
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # Классификация
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(gru_hidden_size, gru_hidden_size // 2)
        self.batch_norm = nn.BatchNorm1d(gru_hidden_size // 2)
        self.fc2 = nn.Linear(gru_hidden_size // 2, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        
        Returns:
            (batch_size, num_classes)
        """
        # TCN
        x_tcn = x.transpose(1, 2)  # (batch, input_size, seq_len)
        tcn_out = self.tcn(x_tcn)  # (batch, tcn_channels[-1], seq_len)
        tcn_out = tcn_out.transpose(1, 2)  # (batch, seq_len, tcn_channels[-1])
        
        # GRU
        gru_out, h_n = self.gru(tcn_out)
        h_n = h_n[-1]  # (batch, gru_hidden_size)
        
        # Classification
        out = self.dropout(h_n)
        out = self.fc1(out)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


def create_tcn_classifier(config):
    """
    Создает TCN классификатор по конфигурации
    
    Args:
        config: словарь с параметрами модели
    
    Returns:
        модель классификатора
    """
    model_type = config.get('model_type', 'tcn')
    
    if model_type == 'hybrid':
        model = HybridTCNClassifier(
            input_size=config['input_size'],
            tcn_channels=config['tcn_channels'],
            gru_hidden_size=config.get('gru_hidden_size', 64),
            num_classes=config['num_classes'],
            kernel_size=config.get('kernel_size', 3),
            dropout=config.get('dropout', 0.2)
        )
    else:
        model = TCNClassifier(
            input_size=config['input_size'],
            num_channels=config['tcn_channels'],
            num_classes=config['num_classes'],
            kernel_size=config.get('kernel_size', 3),
            dropout=config.get('dropout', 0.2)
        )
    
    return model


if __name__ == "__main__":
    # Тестирование модели
    config = {
        'input_size': 8,
        'tcn_channels': [32, 64, 64, 128],
        'num_classes': 5,
        'kernel_size': 3,
        'dropout': 0.2,
        'model_type': 'tcn'
    }
    
    model = create_tcn_classifier(config)
    
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
    
    # Тест гибридной модели
    config['model_type'] = 'hybrid'
    config['gru_hidden_size'] = 64
    model_hybrid = create_tcn_classifier(config)
    output_hybrid = model_hybrid(x)
    
    print("\nГибридная модель:", model_hybrid.__class__.__name__)
    print("Выход:", output_hybrid.shape)
    print("Количество параметров:", sum(p.numel() for p in model_hybrid.parameters()))


