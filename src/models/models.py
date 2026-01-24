"""
Модуль models.py

Содержит модели данных (датаклассы) для передачи между этапами конвейера.
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class DownloadTask:
    """Задание на загрузку изображения."""
    url: str
    breed: str
    metadata: Dict[str, Any]
    index: int


@dataclass
class DownloadedImage:
    """Загруженное изображение."""
    image: np.ndarray
    breed: str
    index: int
    metadata: Dict[str, Any]


@dataclass
class ProcessedImage:
    """Обработанное изображение с тремя версиями."""
    original: np.ndarray
    custom: np.ndarray
    library: np.ndarray
    breed: str
    index: int
    metadata: Dict[str, Any]