"""
Модуль pipeline_config.py

Конфигурация конвейера обработки изображений.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class PipelineConfig:
    """
    Конфигурация конвейера.
    Все значения загружаются из переменных окружения или имеют значения по умолчанию.
    """
    api_key: str = os.getenv('API_KEY', '')
    api_base_url: str = os.getenv('API_BASE_URL', 'https://api.thecatapi.com/v1/images/search')
    
    download_workers: int = int(os.getenv('DOWNLOAD_WORKERS', '3'))
    processing_workers: int = int(os.getenv('PROCESSING_WORKERS', '2'))
    saving_workers: int = int(os.getenv('SAVING_WORKERS', '2'))
    
    download_queue_size: int = int(os.getenv('DOWNLOAD_QUEUE_SIZE', '10'))
    processing_queue_size: int = int(os.getenv('PROCESSING_QUEUE_SIZE', '10'))
    saving_queue_size: int = int(os.getenv('SAVING_QUEUE_SIZE', '5'))
    
    max_images: int = int(os.getenv('MAX_IMAGES', '10'))
    output_dir: str = os.getenv('OUTPUT_DIR', 'processed_cats')
    log_dir: str = os.getenv('LOG_DIR', 'logs')
    
    api_timeout: float = float(os.getenv('API_TIMEOUT', '30.0'))
    download_timeout: float = 30.0
    processing_timeout: float = 60.0