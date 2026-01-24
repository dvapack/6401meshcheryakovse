"""
Модуль saving_worker.py

Воркер сохранения изображений.
Берёт ProcessedImage из saving_queue и сохраняет на диск (original/custom/library).
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from ...models.models import ProcessedImage
from ..queues.image_queue import ImageQueue

logger = logging.getLogger(__name__)


class SavingWorker:
    """
    Воркер сохранения изображений.
    """

    def __init__(
        self,
        saving_queue: ImageQueue,
        output_dir: str = "processed_cats",
        run_id: Optional[str] = None,
    ):
        self._saving_queue = saving_queue
        self._output_dir = Path(output_dir)

        if run_id:
            self._run_id = run_id
        else:
            self._run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    async def run(self) -> None:
        while True:
            item: Optional[ProcessedImage] = await self._saving_queue.get()
            try:
                if item is None:
                    return

                self._save_item(item)

            finally:
                self._saving_queue.task_done()

    def _save_item(self, item: ProcessedImage) -> None:
        breed_dir = self._output_dir / item.breed / self._run_id
        breed_dir.mkdir(parents=True, exist_ok=True)

        base = f"{item.index}_{item.breed}"
        original_path = breed_dir / f"{base}_original.png"
        custom_path = breed_dir / f"{base}_custom.jpg"
        library_path = breed_dir / f"{base}_cv2.jpg"

        ok1 = cv2.imwrite(str(original_path), item.original)
        ok2 = cv2.imwrite(str(custom_path), item.custom)
        ok3 = cv2.imwrite(str(library_path), item.library)

        if not (ok1 and ok2 and ok3):
            raise IOError("Не удалось сохранить одно или несколько изображений")

        logger.debug(f"Сохранено: {breed_dir}")
