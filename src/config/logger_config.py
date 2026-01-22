"""
Модуль logger_config.py

Настройка логирования для всего проекта.
Настраивает консольное и файловое логирование.
"""

import logging
from datetime import datetime
from pathlib import Path


def setup_logging(
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """
    Настраивает логирование для всего проекта и возвращает корневой логгер проекта.
    
    Args:
        log_dir (str): Директория для хранения логов
        console_level (int): Уровень логирования для консоли (INFO по умолчанию)
        file_level (int): Уровень логирования для файлов (DEBUG по умолчанию)
        
    Returns:
        logging.Logger: Настроенный корневой логгер
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel("INFO")
    console_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    log_file = Path(log_dir) / f"cat_pipeline_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(
        filename=log_file,
        mode="a",
        encoding='utf-8'
    )
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
        
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    logger.info(f"Логирование настроено. Файлы логов: {log_dir}")
    logger.debug(f"Консольный уровень: {console_level}, файловый уровень: {file_level}")
    
    return logger