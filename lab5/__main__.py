"""
main.py

Лабораторная работа №4 по курсу "Технологии программирования на Python".

Программа скачивает изображения кошек из API, обрабатывает их выделением контуров
пользовательским и библиотечным методами, и сохраняет результаты.

Запуск:
    python main.py [limit]

Аргументы:
    limit: количество изображений для обработки (по умолчанию 1)
"""

import argparse
import os
import asyncio
import logging

from dotenv import load_dotenv

from .my_implementation.cat_image_processor import CatImageProcessor
from .configs import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

async def main_async(limit: int) -> None:
    """
    Асинхронная основная функция

    Args:
        limit: количество изображений для обработки
    """
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("Ошибка: API_KEY не найден в .env файле")
        return

    processor = CatImageProcessor(api_key)
    logger.info(f"Начинаем обработку {limit} изображений...")
    await processor.run_async(limit=limit)
    logger.info(f"Обработка {limit} изображений завершена успешно.")


def main() -> None:
    """
    Основная синхронная функция для запуска асинхронного кода
    """
    parser = argparse.ArgumentParser(
        description="Лабораторная работа №4: обработка изображений кошек.",
    )
    parser.add_argument(
        "limit",
        type=int,
        nargs='?',
        default=1,
        help="Количество изображений для обработки (по умолчанию 1)",
    )

    args = parser.parse_args()
    asyncio.run(main_async(args.limit))

if __name__ == "__main__":
    main()