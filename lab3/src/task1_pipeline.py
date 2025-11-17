from typing import Generator, List, Any
from abc import ABC, abstractmethod
import time
import functools

from base_pipeline import BasePipeline
from dataloader import Dataloader

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

class Task1Pipeline(BasePipeline, Dataloader):
    """Класс для выполнения задания 1"""

    def __init__(self, csv_path:str, parquet_file: str = "parquet_file.parquet"):
        Dataloader.__init__(self, csv_path, parquet_file)

    def get_data(self, columns: List[str] = ['Country.Name', 'per_capita']) -> pd.DataFrame | Generator[pd.DataFrame, None, None]:
        """
        Метод для загрузки данных для конкретной задачи
        
        Args:
            columns (List[str]): Необходимые для загрузки столбцы

        Returns:
            pd.DataFrame | Generator[pd.DataFrame, None, None]: Либо полный DataFrame, либо генератор чанков DataFrame
        """
        for chunk in self.read_csv_pandas_generator():
            chunk = chunk[chunk['Country.Population'] > 0]
            if len(chunk) > 0:
                chunk['per_capita'] = (
                    chunk['Emissions.Production.CO2.Total'] / 
                    chunk['Country.Population']
                    )
            yield chunk[columns]
    
    def aggregate_data(self, data: pd.DataFrame | Generator[pd.DataFrame, None, None]) -> pd.DataFrame:
        """
        Метод для агрегации данных для конкретной задачи

        Args:
            data (pd.DataFrame | Generator[pd.DataFrame, None, None]): Данные для агрегирования
        
        Returns:
            pd.DataFrame: DataFrame с агрегрированными данными
        """
        aggregated = pd.DataFrame()
        for chunk in data:
            aggregated = pd.concat([aggregated, chunk[['Country.Name', 'per_capita']]], 
                                       ignore_index=True)
        return aggregated
    
    def task_job(self, data: pd.DataFrame) -> Any:
        """
        Метод для выполнения манипуляций над агрегированными данными

        Args:
            data (pd.DataFrame): Агрегированные данные

        Returns:
            Any: Результат задания
        """
        pass
    
    def run(self):
        """
        Обертка-метод для вызова выполнения задания
        """
        
        pass
