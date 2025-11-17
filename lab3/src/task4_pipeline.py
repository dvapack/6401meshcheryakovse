from typing import Generator, List, Any

from .base_pipeline import BasePipeline, time_logger, memory_logger
from .dataloader import Dataloader

import pandas as pd
import matplotlib.pyplot as plt


class Task4Pipeline(BasePipeline, Dataloader):
    """Класс для выполнения задания 4"""

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
        return self.read_parquet(columns=['Country.Population', 'Emissions.Production.CO2.Total'])

    @time_logger
    def aggregate_data(self, data: pd.DataFrame | Generator[pd.DataFrame, None, None]) -> pd.DataFrame:
        """
        Метод для агрегации данных для конкретной задачи

        Args:
            data (pd.DataFrame | Generator[pd.DataFrame, None, None]): Данные для агрегирования
        
        Returns:
            pd.DataFrame: DataFrame с агрегрированными данными
        """
        mask = (data['Country.Population'] > 0) & (data['Emissions.Production.CO2.Total'] > 0)
        df_clean = data[mask].dropna()
        return df_clean

    @time_logger
    def task_job(self, data: pd.DataFrame) -> Any:
        """
        Метод для выполнения манипуляций над агрегированными данными

        Args:
            data (pd.DataFrame): Агрегированные данные

        Returns:
            Any: Результат задания
        """
        populations = data['Country.Population'].tolist()
        emissions = data['Emissions.Production.CO2.Total'].tolist()
        correlation = data['Country.Population'].corr(data['Emissions.Production.CO2.Total'])
        
        return populations, emissions, correlation
    
    def plot_results(self, data: Any):
        """
        Метод для отрисовки результатов работы
        
        Args:
            data (Any): Результат работы
        """
        plt.figure(figsize=(12, 8))
        
        populations, emissions, correlation = data
        plt.scatter(populations, emissions, alpha=0.6, s=20, color='purple')
        plt.title(f'Задание 4: Корреляция население-выбросы\n(r = {correlation:.3f})', fontsize=14)
        plt.xlabel('Население страны')
        plt.ylabel('Выбросы CO2')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @memory_logger
    def run(self):
        """
        Обертка-метод для вызова выполнения задания
        """
        self.plot_results(self.task_job(self.aggregate_data(self.get_data())))
