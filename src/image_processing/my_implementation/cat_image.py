"""
Модуль cat_image.py

Содержит классы CatImage, ColorCatImage, GrayscaleCatImage для инкапсуляции изображений кошек.
Адаптировано для конвейерной архитектуры.
"""

from abc import ABC, abstractmethod
import numpy as np


class CatImage(ABC):
    """
    Абстрактный класс для изображений кошек.
    
    Инкапсулирует изображение и метаданные (url, порода).
    Обработка изображений вынесена в отдельные этапы конвейера.
    """

    def __init__(self, url: str, breed: str, image_array: np.ndarray):
        """
        Инициализация изображения кошки.
        
        Args:
            url (str): URL изображения
            breed (str): Порода кошки
            image_array (np.ndarray): Массив изображения
        """
        self._url = url
        self._breed = breed
        self._image = self.prepare_image(image_array)

    @property
    def url(self) -> str:
        """URL изображения"""
        return self._url
    
    @property
    def breed(self) -> str:
        """Порода кошки"""
        return self._breed
    
    @property
    def image(self) -> np.ndarray:
        """Массив изображения"""
        return self._image
    
    @abstractmethod
    def prepare_image(self, image_array: np.ndarray) -> np.ndarray:
        """
        Абстрактный метод для подготовки изображения (цветное или ч/б).
        
        Args:
            image_array (np.ndarray): Входной массив изображения
            
        Returns:
            np.ndarray: Подготовленное изображение
        """
        pass

    def __add__(self, other: 'CatImage') -> 'CatImage':
        """
        Сложение двух изображений (перегрузка оператора +).
        
        Args:
            other (CatImage): Другое изображение
            
        Returns:
            CatImage: Новое изображение
            
        Raises:
            TypeError: Если other не является CatImage
        """
        if isinstance(other, CatImage):
            new_image = np.clip(self.image.astype(np.int32) + 
                                other.image.astype(np.int32), 0, 255).astype(np.uint8)
            return self.__class__(self.url, f"{self.breed}+{other.breed}", new_image)
        raise TypeError("Можно складывать только CatImage")
    
    def __sub__(self, other: 'CatImage') -> 'CatImage':
        """
        Вычитание двух изображений (перегрузка оператора -).
        
        Args:
            other (CatImage): Другое изображение
            
        Returns:
            CatImage: Новое изображение
            
        Raises:
            TypeError: Если other не является CatImage
        """
        if isinstance(other, CatImage):
            new_image = np.clip(self.image.astype(np.int32) -
                                other.image.astype(np.int32), 0, 255).astype(np.uint8)
            return self.__class__(self.url, f"{self.breed}-{other.breed}", new_image)
        raise TypeError("Можно вычитать только CatImage")

    def __str__(self) -> str:
        """
        Строковое представление объекта.
        
        Returns:
            str: Информация об изображении
        """
        return f"CatImage: breed={self.breed}, url={self.url}, shape={self.image.shape}"


class ColorCatImage(CatImage):
    """
    Класс для цветных изображений кошек.
    """

    def prepare_image(self, image_array: np.ndarray) -> np.ndarray:
        """
        Подготавливает цветное изображение.
        
        Args:
            image_array (np.ndarray): Входной массив изображения
            
        Returns:
            np.ndarray: Цветное изображение (3 канала)
        """
        if image_array.ndim == 3:
            return image_array
        else:
            return np.stack([image_array, image_array, image_array], axis=-1)


class GrayscaleCatImage(CatImage):
    """
    Класс для ч/б изображений кошек.
    """

    def prepare_image(self, image_array: np.ndarray) -> np.ndarray:
        """
        Подготавливает ч/б изображение.
        
        Args:
            image_array (np.ndarray): Входной массив изображения
            
        Returns:
            np.ndarray: Ч/б изображение (1 канал)
        """
        if image_array.ndim == 3:
            gray = np.dot(image_array[..., :3], [0.299, 0.587, 0.114])
            return gray.astype(np.uint8)
        return image_array