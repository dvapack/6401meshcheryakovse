import time
import functools
import logging

logger = logging.getLogger(__name__)

def time_logger(func):
    """
    Декоратор для логирования времени выполнения функций.
    Работает как с синхронными, так и с асинхронными функциями.
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"Метод {func.__name__} выполнен за {end_time - start_time:.4f} секунд")
        return result
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"Метод {func.__name__} выполнен за {end_time - start_time:.4f} секунд")
        return result
    
    return async_wrapper if hasattr(func, '__await__') else sync_wrapper