"""
Модуль main.py

Точка входа в проект. Запускает конвейер обработки изображений.
"""

import asyncio
import logging
import sys

from src.config.pipeline_config import PipelineConfig
from src.pipeline.pipeline import Pipeline

logger = logging.getLogger(__name__)


async def main() -> None:
    config = PipelineConfig()

    if not config.api_key:
        print("Ошибка: не задан API_KEY.")
        sys.exit(1)

    pipeline = Pipeline(config)
    await pipeline.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Остановлено пользователем.")
