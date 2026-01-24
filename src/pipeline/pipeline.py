"""
Модуль pipeline.py

Оркестратор конвейера: получение списка изображений, запуск воркеров и управление очередями.
"""

import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any, Dict, List

from ..api.cat_api_client import CatAPIClient
from ..config.logger_config import setup_logging
from ..config.pipeline_config import PipelineConfig
from ..models.models import DownloadTask
from .queues.image_queue import ImageQueue
from .workers.download_worker import DownloadWorker
from .workers.processing_worker import ProcessingWorker
from .workers.saving_worker import SavingWorker

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Конвейер обработки изображений кошек.
    """

    def __init__(self, config: PipelineConfig):
        self._config = config

    async def run(self) -> None:
        setup_logging(self._config.log_dir)

        download_workers = self._config.download_workers
        processing_workers = max(self._config.processing_workers, download_workers)
        saving_workers = max(self._config.saving_workers, processing_workers)

        if processing_workers != self._config.processing_workers:
            logger.warning(
                f"processing_workers увеличен до {processing_workers}, "
                f"чтобы корректно завершать pipeline"
            )
        if saving_workers != self._config.saving_workers:
            logger.warning(
                f"saving_workers увеличен до {saving_workers}, "
                f"чтобы корректно завершать pipeline"
            )

        download_queue = ImageQueue(maxsize=self._config.download_queue_size, name="download")
        processing_queue = ImageQueue(maxsize=self._config.processing_queue_size, name="processing")
        saving_queue = ImageQueue(maxsize=self._config.saving_queue_size, name="saving")

        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        executor = ProcessPoolExecutor(max_workers=processing_workers)

        tasks: List[asyncio.Task] = []
        try:
            for _ in range(download_workers):
                worker = DownloadWorker(
                    download_queue=download_queue,
                    processing_queue=processing_queue,
                    timeout=self._config.download_timeout,
                )
                tasks.append(asyncio.create_task(worker.run()))

            for _ in range(processing_workers):
                worker = ProcessingWorker(
                    processing_queue=processing_queue,
                    saving_queue=saving_queue,
                    executor=executor,
                    timeout=self._config.processing_timeout,
                )
                tasks.append(asyncio.create_task(worker.run()))

            for _ in range(saving_workers):
                worker = SavingWorker(
                    saving_queue=saving_queue,
                    output_dir=self._config.output_dir,
                    run_id=run_id,
                )
                tasks.append(asyncio.create_task(worker.run()))

            client = CatAPIClient(api_key=self._config.api_key, base_url=self._config.api_base_url)
            images = await client.get_cat_images(limit=self._config.max_images)

            await self._enqueue_download_tasks(download_queue, images)

            for _ in range(download_workers):
                await download_queue.put(None)

            await download_queue.join()
            await processing_queue.join()
            await saving_queue.join()

            await asyncio.gather(*tasks)

            logger.info("Pipeline завершён")

        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            executor.shutdown(wait=False)

    async def _enqueue_download_tasks(
        self,
        download_queue: ImageQueue,
        images: List[Dict[str, Any]],
    ) -> None:
        for i, item in enumerate(images):
            url = item.get("url")
            if not url:
                continue

            breed = self._extract_breed(item)
            task = DownloadTask(url=url, breed=breed, metadata=item, index=i)
            await download_queue.put(task)

    def _extract_breed(self, item: Dict[str, Any]) -> str:
        breeds = item.get("breeds")
        if isinstance(breeds, list) and breeds:
            name = breeds[0].get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        return "Unknown"
