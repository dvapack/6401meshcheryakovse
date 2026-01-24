"""
Модуль download_worker.py

Воркер загрузки изображений.
Берёт задания из download_queue, скачивает изображение и кладёт результат в processing_queue.
"""

import asyncio
import logging
from typing import Optional

import aiohttp
import cv2
import numpy as np

from ...models.models import DownloadTask, DownloadedImage
from ..queues.image_queue import ImageQueue

logger = logging.getLogger(__name__)


class DownloadWorker:
    """
    Воркер загрузки изображений.
    """

    def __init__(
        self,
        download_queue: ImageQueue,
        processing_queue: ImageQueue,
        timeout: float = 30.0,
    ):
        self._download_queue = download_queue
        self._processing_queue = processing_queue
        self._timeout = timeout

    async def run(self) -> None:
        timeout = aiohttp.ClientTimeout(total=self._timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                task: Optional[DownloadTask] = await self._download_queue.get()
                try:
                    if task is None:
                        await self._processing_queue.put(None)
                        return

                    image = await self._download_image(session, task.url)
                    downloaded = DownloadedImage(
                        image=image,
                        breed=task.breed,
                        index=task.index,
                        metadata=task.metadata,
                    )
                    await self._processing_queue.put(downloaded)

                finally:
                    self._download_queue.task_done()

    async def _download_image(self, session: aiohttp.ClientSession, url: str) -> np.ndarray:
        logger.debug(f"Скачивание изображения: {url}")

        async with session.get(url) as response:
            response.raise_for_status()
            data = await response.read()

        arr = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Не удалось декодировать изображение")

        return image
