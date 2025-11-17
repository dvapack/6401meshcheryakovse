from typing import Generator, List, Any

from .base_pipeline import BasePipeline, time_logger, memory_logger
from .dataloader import Dataloader

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Task3Pipeline(BasePipeline, Dataloader):
    """Класс для выполнения задания 3"""

    def __init__(self, csv_path: str, parquet_file: str = "parquet_file.parquet"):
        Dataloader.__init__(self, csv_path, parquet_file)

    @time_logger
    def get_data(self, columns: List[str] = []) -> pd.DataFrame | Generator[pd.DataFrame, None, None]:
        """
        Метод для загрузки данных для конкретной задачи
        
        Args:
            columns (List[str]): Необходимые для загрузки столбцы

        Returns:
            pd.DataFrame | Generator[pd.DataFrame, None, None]: Либо полный DataFrame, либо генератор чанков DataFrame
        """
        for chunk in self.read_csv_pandas_generator():
            yield chunk.groupby('Year').agg({'Country.GDP': 'sum', 'Emissions.Production.CO2.Total': 'sum'})

    @time_logger
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
            aggregated = aggregated.add(chunk, fill_value=0)
        return aggregated

    @time_logger
    def task_job(self, data: pd.DataFrame) -> Any:
        """
        Метод для выполнения манипуляций над агрегированными данными

        Args:
            data (pd.DataFrame): Агрегированные данные

        Returns:
            Any: Результат задания
        """
        result = []
        for year in sorted(data.index):
            result.append((year, data.loc[year, 'Country.GDP'], data.loc[year, 'Emissions.Production.CO2.Total']))
        return result
    
    def plot_results(self, data: Any):
        """
        Метод для отрисовки результатов работы
        
        Args:
            data (Any): Результат работы
        """
        plt.figure(figsize=(12, 8))
        
        years = [x[0] for x in data]
        gdp_values = [x[1] for x in data]
        emissions_values = [x[2] for x in data]
        
        gdp_norm = (gdp_values - np.min(gdp_values)) / (np.max(gdp_values) - np.min(gdp_values))
        emissions_norm = (emissions_values - np.min(emissions_values)) / (np.max(emissions_values) - np.min(emissions_values))
        
        plt.plot(years, gdp_norm, 'g-', marker='o', linewidth=2, label='Нормализованный ВВП')
        plt.plot(years, emissions_norm, 'r-', marker='s', linewidth=2, label='Нормализованные выбросы')
        plt.title('Задание 3: Динамика ВВП и выбросов CO2', fontsize=14)
        plt.xlabel('Год')
        plt.ylabel('Нормализованные значения')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @memory_logger
    def run(self):
        """
        Обертка-метод для вызова выполнения задания
        """
        self.plot_results(self.task_job(self.aggregate_data(self.get_data())))
