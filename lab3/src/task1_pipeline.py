from typing import Generator, List, Any

from .base_pipeline import BasePipeline, time_logger, memory_logger
from .dataloader import Dataloader

import pandas as pd
import matplotlib.pyplot as plt


class Task1Pipeline(BasePipeline, Dataloader):
    """Класс для выполнения задания 1"""

    def __init__(self, csv_path: str, parquet_file: str = "parquet_file.parquet"):
        Dataloader.__init__(self, csv_path, parquet_file)

    @time_logger
    def get_data(self, columns: List[str] = ['Country.Name', 'per_capita', 'count']) -> pd.DataFrame | Generator[pd.DataFrame, None, None]:
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
            chunk['count'] = 1
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
        aggregated = pd.DataFrame(columns=['Country.Name', 'per_capita', 'count'])
        for chunk in data:
            aggregated = pd.concat([aggregated, chunk], ignore_index=True)
            print(f"aggregated.memory_usage: {aggregated.memory_usage}")
            aggregated = aggregated.groupby('Country.Name').agg({
                                                                'per_capita': 'sum',
                                                                'count': 'sum'
                                                            }).reset_index()
        aggregated['per_capita'] = aggregated['per_capita'] / aggregated['count']
        
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
        sorted_data = data.sort_values('per_capita')
        greenest = list(sorted_data.head(3).itertuples(index=False, name=None))
        dirtiest = list(sorted_data.tail(3).itertuples(index=False, name=None))
        return greenest, dirtiest
    
    def plot_results(self, data: Any):
        """
        Метод для отрисовки результатов работы
        
        Args:
            data (Any): Результат работы
        """
        plt.figure(figsize=(12, 8))
    
        greenest, dirtiest = data
        all_countries = greenest + dirtiest 
        countries = [x[0] for x in all_countries]
        emissions_per_capita = [x[1] for x in all_countries]
        
        plt.bar(countries, emissions_per_capita, color=['green']*3 + ['red']*3, alpha=0.7)
        plt.title('Задание 1: Самые "зеленые" и "грязные" страны\n(выбросы на душу населения)', fontsize=14)
        plt.yscale("log")
        plt.ylabel('Выбросы CO2 на душу населения')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(f"countries {countries}, emissions_per_capita {emissions_per_capita}")

    @memory_logger
    def run(self):
        """
        Обертка-метод для вызова выполнения задания
        """
        self.plot_results(self.task_job(self.aggregate_data(self.get_data(['Country.Name', 'per_capita', 'count']))))
