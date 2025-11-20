from typing import Generator, List, Any

from .base_pipeline import BasePipeline, time_logger, memory_logger
from .dataloader import Dataloader

import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np


class Task2Pipeline(BasePipeline, Dataloader):
    """Класс для выполнения задания 2"""

    def __init__(self, csv_path: str, parquet_file: str = "parquet_file.parquet"):
        Dataloader.__init__(self, csv_path, parquet_file)

    @time_logger
    def get_data(self, columns: List[str] = ['Country.Name', 'Emissions.Production.CO2.Total']) -> pd.DataFrame | Generator[pd.DataFrame, None, None]:
        """
        Метод для загрузки данных для конкретной задачи
        
        Args:
            columns (List[str]): Необходимые для загрузки столбцы

        Returns:
            pd.DataFrame | Generator[pd.DataFrame, None, None]: Либо полный DataFrame, либо генератор чанков DataFrame
        """
        for chunk in self.read_csv_pandas_generator():
            yield chunk[columns]

    @time_logger  
    def aggregate_data(self, data: pd.DataFrame | Generator[pd.DataFrame, None, None]) -> pd.DataFrame:
        """
        Метод для агрегации данных для конкретной задачи

        Args:
            data (pd.DataFrame | Generator[pd.DataFrame, None, None]): Данные для агрегирования
        
        Returns:
            pd.DataFrame: DataFrame с агрегрированными данными
        """
        aggregated = pd.DataFrame(columns=['Country.Name', 'sum_x', 'sum_x2', 'count'])
        
        for chunk in data:
            chunk_agg = chunk.groupby('Country.Name').agg({
                'Emissions.Production.CO2.Total': ['sum', lambda x: (x**2).sum(), 'count']
            }).reset_index()
            
            chunk_agg.columns = ['Country.Name', 'sum_x', 'sum_x2', 'count']
            aggregated = pd.concat([aggregated, chunk_agg], ignore_index=True)
            aggregated = aggregated.groupby('Country.Name').agg({
                'sum_x': 'sum',
                'sum_x2': 'sum', 
                'count': 'sum'
            }).reset_index()
        
        aggregated['mean_x'] = aggregated['sum_x'] / aggregated['count']
        aggregated['mean_x2'] = aggregated['sum_x2'] / aggregated['count']
        aggregated['variance'] = aggregated['mean_x2'] - (aggregated['mean_x'] ** 2)
        
        aggregated['std'] = np.sqrt((aggregated['variance']).fillna(0))
        
        print(f"aggregated.memory_usage: {aggregated.memory_usage()}")
        return aggregated[['Country.Name', 'mean_x', 'std', 'count', 'variance']]

    @time_logger  
    def task_job(self, data: pd.DataFrame) -> Any:
        """
        Метод для выполнения манипуляций над агрегированными данными

        Args:
            data (pd.DataFrame): Агрегированные данные

        Returns:
            Any: Результат задания
        """
        sorted_var = data.sort_values('variance')
        
        lowest_var = list(sorted_var.head(3).itertuples(index=False, name=None))
        highest_var = list(sorted_var.tail(3).itertuples(index=False, name=None))
        
        selected_countries = [country for country, _, _, _, _ in lowest_var + highest_var]
        ci_data = {}
        
        for country in selected_countries:
            country_stats = data[data['Country.Name'] == country].iloc[0]
            mean = country_stats['mean_x']
            std = country_stats['std']
            n = country_stats['count']
            
            if n > 1:
                sem = std / np.sqrt(n)
                ci = norm.ppf(0.95) * sem
                ci_data[country] = (mean - ci, mean + ci)
            else:
                ci_data[country] = (mean, mean)
        
        return lowest_var, highest_var, ci_data

    def plot_results(self, data: Any):
        """
        Метод для отрисовки результатов работы
        
        Args:
            data (Any): Результат работы
        """
        plt.figure(figsize=(12, 8))
        
        lowest_var, highest_var, ci_data = data
        all_countries_task2 = [x[0] for x in lowest_var] + [x[0] for x in highest_var]
        
        means = [np.mean(ci_data[country]) for country in all_countries_task2]
        errors = [[means[i] - ci_data[country][0] for i, country in enumerate(all_countries_task2)], 
                [ci_data[country][1] - means[i] for i, country in enumerate(all_countries_task2)]]
        
        bars = plt.bar(all_countries_task2, means, yerr=errors, capsize=5, 
                    color=['blue']*3 + ['orange']*3, alpha=0.7)
        plt.title('Задание 2: Страны с наименьшим/наибольшим разбросом выбросов\n(95% доверительные интервалы)', fontsize=14)
        plt.ylabel('Средние выбросы CO2')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.yscale("log")
        plt.tight_layout()
        plt.show()

    @memory_logger
    def run(self):
        """
        Обертка-метод для вызова выполнения задания
        """
        self.plot_results(self.task_job(self.aggregate_data(self.get_data(['Country.Name', 'Emissions.Production.CO2.Total']))))
