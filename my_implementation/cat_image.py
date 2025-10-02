"""
Модуль cat_image.py

Содержит классы CatImage, ColorCatImage, GrayscaleCatImage для инкапсуляции изображений кошек.
"""

from abc import ABC, abstractmethod

import numpy as np
import time
import functools

from .my_image_processing import MyImageProcessing
from implementation.image_processing import ImageProcessing


def time_logger(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Метод {func.__name__} выполнен за {end_time - start_time:.4f} секунд")
        return result
    return wrapper


class CatImage(ABC):
    """
    Абстрактный класс для изображений кошек.

    Инкапсулирует изображение, метаданные (url, порода), методы обработки
    и перегруженные операторы.
    """

    def __init__(self, url: str, breed: str, image_array: np.ndarray):
        self.url = url
        self.breed = breed
        self.custom_processor = MyImageProcessing()
        self.library_processor = ImageProcessing()
        self.image = self.prepare_image(image_array)

    @abstractmethod
    def prepare_image(self, image_array: np.ndarray) -> np.ndarray:
        """
        Абстрактный метод для подготовки изображения (цветное или ч/б).
        """
        pass

    @time_logger
    def custom_edge_detection(self) -> np.ndarray:
        """
        Выполняет выделение контуров самописным методом.

        Returns:
            np.ndarray: Изображение с выделенными контурами.
        """
        return self.custom_processor.edge_detection(self.image)

    @time_logger
    def library_edge_detection(self) -> np.ndarray:
        """
        Выполняет выделение контуров библиотечным методом (Canny).

        Returns:
            np.ndarray: Изображение с выделенными контурами.
        """
        return self.library_processor.edge_detection(self.image)

    def __add__(self, other: 'CatImage') -> 'CatImage':
        if isinstance(other, CatImage):
            new_image = np.clip(self.image.astype(np.int32) + 
                                other.image.astype(np.int32), 0, 255).astype(np.uint8)
            return self.__class__(self.url, f"{self.breed}+{other.breed}", new_image)
        return NotImplemented

    def __sub__(self, other: 'CatImage') -> 'CatImage':
        if isinstance(other, CatImage):
            new_image = np.clip(self.image.astype(np.int32) -
                                 other.image.astype(np.int32), 0, 255).astype(np.uint8)
            return self.__class__(self.url, f"{self.breed}-{other.breed}", new_image)
        return NotImplemented

    def __str__(self) -> str:
        return f"CatImage: breed={self.breed}, url={self.url}, shape={self.image.shape}"


class ColorCatImage(CatImage):
    """
    Класс для цветных изображений кошек.
    """

    def prepare_image(self, image_array: np.ndarray) -> np.ndarray:
        return image_array


class GrayscaleCatImage(CatImage):
    """
    Класс для ч/б изображений кошек.
    """

    def prepare_image(self, image_array: np.ndarray) -> np.ndarray:
        if image_array.ndim == 3:
            return self.custom_processor._rgb_to_grayscale(image_array).astype(np.uint8)
        return image_array