"""
Модуль decorators.py

Декораторы для логирования времени выполнения функций.
"""

import asyncio
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)


def time_logger(func):
    """
    Декоратор для логирования времени выполнения функции.
    Поддерживает синхронные и асинхронные функции.
    """
    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed = time.time() - start
                logger.debug(f"{func.__name__} выполнена за {elapsed:.4f} сек")
        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.time() - start
                logger.debug(f"{func.__name__} выполнена за {elapsed:.4f} сек")
        return sync_wrapper
