"""
Модуль cat_image_processor.py

Содержит класс CatImageProcessor для работы с API и обработки изображений.
"""

import os
import requests
from PIL import Image
import io
import numpy as np
import random
from typing import List, Dict, Any
import time
import functools

from .cat_image import ColorCatImage
from .cat_image import GrayscaleCatImage


def time_logger(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Метод {func.__name__} выполнен за {end_time - start_time:.4f} секунд")
        return result
    return wrapper


class CatImageProcessor:
    """
    Класс для обработки изображений кошек.

    Инкапсулирует функционал работы с API, скачивания, обработки и сохранения изображений.
    """

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._base_url = "https://api.thecatapi.com/v1/images/search"

    @property
    def api_key(self) -> str:
        return self._api_key
    
    @property
    def base_url(self) -> str:
        return self._base_url

    @time_logger
    def fetch_images(self, limit: int = 1) -> List[Dict[str, Any]]:
        """
        Скачивает данные изображений из API.

        Args:
            limit (int): Количество изображений.

        Returns:
            List[Dict[str, Any]]: Список данных изображений.
        """
        headers = {"x-api-key": self.api_key}
        params = {"limit": limit, "size": "med", "has_breeds": True}
        response = requests.get(self.base_url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    @time_logger
    def download_image(self, url: str) -> np.ndarray:
        """
        Скачивает изображение по URL.

        Args:
            url (str): URL изображения.

        Returns:
            np.ndarray: Массив изображения.
        """
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        return np.array(img)

    @time_logger
    def process_and_save(self, images_data: List[Dict[str, Any]], output_dir: str = "processed_cats"):
        """
        Обрабатывает и сохраняет изображения.

        Args:
            images_data (List[Dict[str, Any]]): Данные изображений.
            output_dir (str): Директория для сохранения.
        """
        os.makedirs(output_dir, exist_ok=True)
        for i, data in enumerate(images_data, 1):
            url = data["url"]
            breeds = data.get("breeds", [])
            breed_name = breeds[0]["name"] if breeds else "Unknown"
            dir = os.path.join(output_dir, breed_name)
            os.makedirs(dir, exist_ok=True)
            image_array = self.download_image(url)
            method = random.choice(['color', 'grayscale'])
            if method == 'grayscale':
                cat_img = GrayscaleCatImage(url, breed_name, image_array)
            else:
                cat_img = ColorCatImage(url, breed_name, image_array)
            original_path = os.path.join(dir, f"{i}_{breed_name}_original.png")
            Image.fromarray(cat_img.image).save(original_path)
            custom_edges = cat_img.custom_edge_detection()
            custom_path = os.path.join(dir, f"{i}_{breed_name}_custom_edges.png")
            Image.fromarray(custom_edges).save(custom_path)
            lib_edges = cat_img.library_edge_detection()
            lib_path = os.path.join(output_dir, f"{i}_{breed_name}_library_edges.png")
            Image.fromarray(lib_edges).save(lib_path)