from typing import Generator, List, Any
from abc import ABC, abstractmethod

from dataloader import Dataloader

import pandas as pd


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
    def run(self):
        """
        Обертка-метод для вызова выполнения задания
        """
        pass