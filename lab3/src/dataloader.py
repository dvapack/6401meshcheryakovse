from typing import Generator, Tuple, List

import os
import time

import pandas as pd


class Dataloader:
    """
    Класс для работы с данными.
    Поддерживает:
        1. Загрузку чанками из csv с помощью Pandas
        2. Загрузку из csv с помощью Parquet
        3. Создание Parquet файла
        4. Сравнение скорости загрузки данных из SCV и Parquet
    """
    _chunksize = 1000

    def __init__(self, csv_path: str, parquet_path: str = "data/parquet_file.parquet"):
        self._csv_path = csv_path
        self._parquet_path = parquet_path

    @property
    def chunksize(self):
        """Chunksize property"""
        return self._chunksize
    
    @property
    def csv_path(self):
        """CSV_path property"""
        return self._csv_path
    
    @property
    def parquet_path(self):
        """Parquet_path property"""
        return self._parquet_path

    def read_csv_pandas_generator(self, chunksize: int = 1000) -> Generator[pd.DataFrame, None, None]:
        """
        Функция-генератор для чтения CSV файла через Pandas
        
        Args:
            chunksize (int): Размер чанков для прочтения

        Returns:
            Generator[pd.DataFrame, None, None]: Специальный итератор (генератор)
        """
        self._chunksize = chunksize
        for chunk in pd.read_csv(self.csv_path, chunksize=chunksize):
            yield chunk

    def _create_parquet_file(self) -> None:
        """Метод для создания Parquet файла"""
        print("Создание Parquet файла...")
        chunks = pd.read_csv(self.csv_path, chunksize=self.chunksize)
        df = pd.concat(chunks)
        df.to_parquet(self._parquet_path)
        print("Parquet файл создан")

    def read_parquet(self, columns: List[str]) -> pd.DataFrame:
        """
        Метод для чтения Parquet файла
        
        Args:
            columns (List[str]): Необходимые для прочтения столбцы

        Returns:
            pd.DataFrame: DataFrame с данными
        """
        if not os.path.exists(self._parquet_path):
            self._create_parquet_file()
        df = pd.read_parquet(self._parquet_path, columns=columns)
        return df
    
    def benchmark_reading_speed(self) -> Tuple[float, float]:
        """
        Сравнивает скорость чтения CSV и Parquet файлов

        Returns:
            Tuple[float, float]: Кортеж - (csv_time, parquet_time).
        """
        start_time = time.time()
        df_csv = pd.read_csv(self._csv_path)
        csv_time = time.time() - start_time
        print(f"CSV: {csv_time:.3f} сек, {len(df_csv)} строк")
        start_time = time.time()
        df_parquet = pd.read_parquet(self._parquet_path)
        parquet_time = time.time() - start_time
        print(f"Parquet: {parquet_time:.3f} сек, {len(df_parquet)} строк")
        return csv_time, parquet_time