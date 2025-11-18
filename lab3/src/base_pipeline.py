from typing import Generator, List, Any
from abc import ABC, abstractmethod

import time
import functools
import psutil
import os

import pandas as pd

def time_logger(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Метод {func.__name__} выполнен за {end_time - start_time:.4f} секунд")
        return result
    return wrapper

def memory_logger(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024
        
        result = func(*args, **kwargs)
        
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_used = memory_after - memory_before
        
        print(f"Метод {func.__name__} использовал {memory_used:.2f} МБ памяти")
        return result
    return wrapper

class BasePipeline(ABC):
    """Базовый класс для выполнения лабораторной работы"""

    @abstractmethod
    def get_data(self, columns: List[str]) -> pd.DataFrame | Generator[pd.DataFrame, None, None]:
        """
        Метод для загрузки данных для конкретной задачи
        
        Args:
            columns (List[str]): Необходимые для загрузки столбцы

        Returns:
            pd.DataFrame | Generator[pd.DataFrame, None, None]: Либо полный DataFrame, либо генератор чанков DataFrame
        """
        pass
    
    @abstractmethod
    def aggregate_data(self, data: pd.DataFrame | Generator[pd.DataFrame, None, None]) -> pd.DataFrame:
        """
        Метод для агрегации данных для конкретной задачи

        Args:
            data (pd.DataFrame | Generator[pd.DataFrame, None, None]): Данные для агрегирования
        
        Returns:
            pd.DataFrame: DataFrame с агрегрированными данными
        """
        pass
    
    @abstractmethod
    def task_job(self, data: pd.DataFrame) -> Any:
        """
        Метод для выполнения манипуляций над агрегированными данными

        Args:
            data (pd.DataFrame): Агрегированные данные

        Returns:
            Any: Результат задания
        """
        pass
    
    @abstractmethod
    def plot_results(self, data: Any):
        """
        Метод для отрисовки результатов работы
        
        Args:
            data (Any): Результат работы
        """
        pass
    
    @abstractmethod
    def run(self):
        """
        Обертка-метод для вызова выполнения задания
        """
        pass