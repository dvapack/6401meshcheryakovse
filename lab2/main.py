"""
main.py

Лабораторная работа №2 по курсу "Технологии программирования на Python".

Программа скачивает изображения кошек из API, обрабатывает их выделением контуров
пользовательским и библиотечным методами, и сохраняет результаты.

Запуск:
    python main.py [limit]

Аргументы:
    limit: количество изображений для обработки (по умолчанию 1)
"""

import argparse
import os

from dotenv import load_dotenv

from my_implementation import CatImageProcessor

def main() -> None:
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("Ошибка: API_KEY не найден в .env файле")
        return

    parser = argparse.ArgumentParser(
        description="Лабораторная работа №2: обработка изображений кошек.",
    )
    parser.add_argument(
        "limit",
        type=int,
        default=1,
        help="Количество изображений для обработки (по умолчанию 1)",
    )

    args = parser.parse_args()

    processor = CatImageProcessor(api_key)
    try:
        images_data = processor.fetch_images(limit=args.limit)
        processor.process_and_save(images_data)
        print("Обработка завершена.")
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
