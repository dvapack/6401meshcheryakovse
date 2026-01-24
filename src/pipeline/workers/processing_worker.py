"""
Модуль processing_worker.py

Воркер обработки изображений.
Берёт DownloadedImage из processing_queue, применяет 2 реализации обработки и кладёт ProcessedImage в saving_queue.
"""

import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

from ...models.models import DownloadedImage, ProcessedImage
from ...image_processing.my_implementation.my_image_processing import MyImageProcessing
from ...image_processing.library_implementation.image_processing import ImageProcessing
from ..queues.image_queue import ImageQueue

logger = logging.getLogger(__name__)


def _custom_edge_detection(image):
    processor = MyImageProcessing()
    return processor.edge_detection(image)


def _library_edge_detection(image):
    processor = ImageProcessing()
    return processor.edge_detection(image)


class ProcessingWorker:
    """
    Воркер обработки изображений.
    """

    def __init__(
        self,
        processing_queue: ImageQueue,
        saving_queue: ImageQueue,
        executor: ProcessPoolExecutor,
        timeout: float = 60.0,
    ):
        self._processing_queue = processing_queue
        self._saving_queue = saving_queue
        self._executor = executor
        self._timeout = timeout

    async def run(self) -> None:
        loop = asyncio.get_running_loop()

        while True:
            item: Optional[DownloadedImage] = await self._processing_queue.get()
            try:
                if item is None:
                    await self._saving_queue.put(None)
                    return

                logger.debug(f"Обработка изображения: breed={item.breed}, index={item.index}")

                task = self._process(loop, item)
                processed = await asyncio.wait_for(task, timeout=self._timeout)

                await self._saving_queue.put(processed)

            finally:
                self._processing_queue.task_done()

    async def _process(self, loop: asyncio.AbstractEventLoop, item: DownloadedImage) -> ProcessedImage:
        custom_fut = loop.run_in_executor(self._executor, _custom_edge_detection, item.image)
        library_fut = loop.run_in_executor(self._executor, _library_edge_detection, item.image)

        custom, library = await asyncio.gather(custom_fut, library_fut)

        return ProcessedImage(
            original=item.image,
            custom=custom,
            library=library,
            breed=item.breed,
            index=item.index,
            metadata=item.metadata,
        )
