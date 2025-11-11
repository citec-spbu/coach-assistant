"""
Классификатор танцевальных фигур

Система классификации танцевальных фигур на основе поз,
извлеченных из видео с помощью модуля DancePose.
"""

__version__ = '1.0.0'
__author__ = 'Dance Classifier Team'

# Основные модули
from .data_preparation import extract_poses, feature_extraction, dataset_builder
from .models import gru_classifier, tcn_classifier
from .training import train
from .inference import predict
from .utils import helpers

__all__ = [
    'extract_poses',
    'feature_extraction',
    'dataset_builder',
    'gru_classifier',
    'tcn_classifier',
    'train',
    'predict',
    'helpers'
]


Классификатор танцевальных фигур

Система классификации танцевальных фигур на основе поз,
извлеченных из видео с помощью модуля DancePose.
"""

__version__ = '1.0.0'
__author__ = 'Dance Classifier Team'

# Основные модули
from .data_preparation import extract_poses, feature_extraction, dataset_builder
from .models import gru_classifier, tcn_classifier
from .training import train
from .inference import predict
from .utils import helpers

__all__ = [
    'extract_poses',
    'feature_extraction',
    'dataset_builder',
    'gru_classifier',
    'tcn_classifier',
    'train',
    'predict',
    'helpers'
]


