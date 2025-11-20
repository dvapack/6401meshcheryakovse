"""
Модуль cat_image_processor.py

Содержит класс CatImageProcessor для работы с API и обработки изображений.
"""

import os
import io
import time
import functools
from typing import List, Dict, Any, Tuple, Callable

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import aiohttp
import aiofiles
import asyncio

import numpy as np
from PIL import Image

from .cat_image import ColorCatImage

def time_logger_async(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        print(f"Метод {func.__name__} выполнен за {end_time - start_time:.4f} секунд")
        return result
    return wrapper

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

    def __init__(self, api_key: str, output_dir: str = "processed_cats"):
        self._api_key = api_key
        self._base_url = "https://api.thecatapi.com/v1/images/search"
        self.output_dir = output_dir

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def base_url(self) -> str:
        return self._base_url

    @staticmethod
    async def _download_image(session: aiohttp.ClientSession, url: str, index: int) -> Tuple[int, np.ndarray]:
        """
        Асинхронно скачивает изображение по URL.

        Args:
            session (aiohttp.ClientSession): HTTP сессия
            url (str): URL изображения.

        Returns:
            Tuple[index, np.ndarray]: Кортеж (индекс, массив изображения).
        """
        print(f"_download_image для изображения {index}")
        async with session.get(url) as response:
            img_data = await response.read()
            img = Image.open(io.BytesIO(img_data))
            return index, np.array(img)

    @time_logger_async
    async def download_images(self, limit: int = 1) -> List[Tuple[Dict[str, Any], int, np.ndarray]]:
        """
        Асинхронно скачивает все изображения.

        Args:
            limit (int): Лимит по количеству изображений.

        Returns:
            List[Tuple[Dict[str, Any], int, np.ndarray]]: Список троек (данные, индекс, изображение).
        """
        headers = {"x-api-key": self.api_key}
        params = {"limit": limit, "size": "med", "has_breeds": str(True)}
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(self.base_url, headers=headers, params=params) as response:
                images_data = await response.json()

            tasks = []
            for i, data in enumerate(images_data):
                url = data["url"]
                task = self._download_image(session=session, url=url, index=i)
                tasks.append(task)

            download_results = await asyncio.gather(*tasks)

            results = []
            for i, data in enumerate(images_data):
                result_index, image_array = download_results[i]
                results.append((data, result_index, image_array))

        return results

    @staticmethod
    @time_logger
    def process_single_image(args: Tuple[Dict[str, Any], int, np.ndarray]) \
                            -> Tuple[int, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        """
        Обрабатывает одно изображение в отдельном процессе.

        Args:
            args: Кортеж (данные изображения, индекс, массив изображения)

        Returns:
            Tuple: (индекс, данные, оригинальное изображение, кастомные границы, библиотечные границы, тип обработки)
        """
        data, index, image_array = args
        breeds = data.get("breeds", [])
        breed_name = breeds[0]["name"] if breeds else "Unknown"
        url = data["url"]

        print(f"Обработка изображения {index} ({breed_name}) в процессе PID[{os.getpid()}]")
        cat_img = ColorCatImage(url, breed_name, image_array)

        custom_edges = cat_img.custom_edge_detection()
        lib_edges = cat_img.library_edge_detection()

        return index, data, image_array, custom_edges, lib_edges

    @time_logger_async
    async def process_images_parallel(
            self, downloaded_data: List[Tuple[Dict[str, Any], int, np.ndarray]]
            ) -> List[Tuple[int, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]]:
        """
        Параллельная обработка изображений с использованием ProcessPoolExecutor.

        Args:
            downloaded_data (List[Tuple[Dict[str, Any], np.ndarray]]): Загруженные данные и изображения.

        Returns:
            List[Tuple]: Результаты обработки
        """
        loop = asyncio.get_event_loop()
        tasks = []
        with ThreadPoolExecutor() as executor:
            for args in downloaded_data:
                future = loop.run_in_executor(executor, self.process_single_image, args)
                tasks.append(future)

            results = await asyncio.gather(*tasks)

        return results

    @time_logger_async
    async def save_image(self, file_path: str, image_array: np.ndarray) -> None:
        image = Image.fromarray(image_array)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')

        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(image_bytes.getvalue())

    @time_logger_async
    async def save_one_dir_async(self, index: int, data: Dict[str, Any],
                                    original_image: np.ndarray, custom_edges: np.ndarray,
                                    lib_edges: np.ndarray) -> None:
        """
        Асинхронно сохраняет одно изображение и результаты обработки.

        Args:
            index (int): Индекс изображения
            data (Dict[str, Any]): Данные изображения
            original_image (np.ndarray): Оригинальное изображение
            custom_edges (np.ndarray): Кастомные границы
            lib_edges (np.ndarray): Библиотечные границы
        """
        local_time = time.localtime()
        local_time = time.strftime("%Y-%m-%d_%H-%M-%S", local_time)
        breeds = data.get("breeds", [])
        breed_name = breeds[0]["name"] if breeds else "Unknown"
        breed_dir = os.path.join(self.output_dir, breed_name, local_time)
        os.makedirs(breed_dir, exist_ok=True)
        tasks = [
            self.save_image(
                os.path.join(breed_dir, f"{index}_{breed_name}_original.png"),
                original_image,
            ),
            self.save_image(
                os.path.join(breed_dir, f"{index}_{breed_name}_custom.jpg"),
                custom_edges,
            ),
            self.save_image(
                os.path.join(breed_dir, f"{index}_{breed_name}_cv2.jpg"),
                lib_edges,
            ),
        ]
        await asyncio.gather(*tasks)

    @time_logger_async
    async def save_images_async(self, processed_results: List[Tuple[int, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]]) -> None:
        """
        Асинхронно сохраняет все обработанные изображения.

        Args:
            processed_results: Результаты обработки изображений
        """
        tasks = []
        for index, data, original_image, custom_edges, lib_edges in processed_results:
            task = self.save_one_dir_async(index, data, original_image, custom_edges, lib_edges)
            tasks.append(task)

        await asyncio.gather(*tasks)


    @time_logger_async
    async def run_async(self, limit: int = 1) -> None:
        """
        Основной асинхронный метод для выполнения всего pipeline.

        Args:
            limit (int): Количество изображений для обработки.
        """
        downloaded_data = await self.download_images(limit=limit)
        processed_results = await self.process_images_parallel(downloaded_data)
        await self.save_images_async(processed_results)

    @time_logger_async
    async def process_and_save_async(self, limit: int = 1, output_dir: str = "processed_cats") -> None:
        """
        Асинхронная версия метода process_and_save для обратной совместимости.

        Args:
            limit (int): Количество изображений.
            output_dir (str): Директория для сохранения.
        """
        if output_dir:
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)

        await self.run_async(limit)