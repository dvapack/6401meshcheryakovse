"""
Модуль cat_api_client.py

Клиент для работы с TheCatAPI. Получает список изображений кошек с метаданными.
"""

import aiohttp
import asyncio
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CatAPIClient:
    """
    Клиент для работы с TheCatAPI.
    
    Отвечает за получение списка изображений кошек с метаданными.
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.thecatapi.com/v1/images/search"):
        """
        Инициализация клиента.
        
        Args:
            api_key (str): API ключ для доступа к TheCatAPI
            base_url (str): Базовый URL API
        """
        self._api_key = api_key
        self._base_url = base_url

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def base_url(self) -> str:
        return self._base_url
    
    async def get_cat_images(self, limit: int = 1) -> List[Dict[str, Any]]:
        """
        Получает список изображений кошек с метаданными.
        
        Args:
            limit (int): Количество изображений для получения (макс. 100)
            
        Returns:
            List[Dict[str, Any]]: Список словарей с метаданными изображений
                                  Каждый словарь содержит:
                                  - id: идентификатор изображения
                                  - url: URL изображения для загрузки
                                  - breeds: список пород (может быть пустым)
                                  - и другие метаданные
                                  
        Raises:
            aiohttp.ClientError: При ошибках сети или API
            ValueError: При некорректных параметрах
        """
        if limit <= 0 or limit > 100:
            raise ValueError("limit должен быть в диапазоне 1-100")
        
        headers = {"x-api-key": self.api_key}
        params = {
            "limit": limit,
            "has_breeds": 1,
            "size": "med"
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                logger.debug(f"Запрашиваем {limit} изображений с API")
                
                async with session.get(
                    self.base_url,
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    response.raise_for_status()
                    data = await response.json()
                    
                    logger.debug(f"Получено {len(data)} изображений от API")
                    return data
                    
            except aiohttp.ClientError as e:
                logger.error(f"Ошибка при запросе к API: {e}")
                raise
            except asyncio.TimeoutError:
                logger.error("Таймаут при запросе к API")
                raise
            except Exception as e:
                logger.error(f"Неожиданная ошибка: {e}")
                raise