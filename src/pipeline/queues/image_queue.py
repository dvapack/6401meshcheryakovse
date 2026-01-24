"""
Модуль image_queue.py

Очередь для передачи данных между стадиями пайплайна.
"""

import asyncio
import logging
from typing import Generic, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


class ImageQueue(Generic[T]):
    """
    Очередь для обмена сообщениями между воркерами пайплайна.
    """

    def __init__(self, maxsize: int = 0, name: str = ""):
        """
        Args:
            maxsize (int): Максимальный размер очереди (0 — без ограничений)
            name (str): Имя очереди (для логирования)
        """
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=maxsize)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def qsize(self) -> int:
        size = self._queue.qsize()
        return size

    async def put(self, item: T) -> None:
        await self._queue.put(item)
        logger.debug(f"Элемент добавлен в очередь '{self._name}'")

    async def get(self) -> T:
        item = await self._queue.get()
        logger.debug(f"Элемент получен из очереди '{self._name}'")
        return item

    def task_done(self) -> None:
        self._queue.task_done()
        logger.debug(f"task_done() для очереди '{self._name}'")

    async def join(self) -> None:
        logger.debug(f"Ожидание завершения очереди '{self._name}'")
        await self._queue.join()
        logger.debug(f"Очередь '{self._name}' обработана полностью")
