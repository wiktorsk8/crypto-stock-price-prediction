from abc import ABC, abstractmethod
from typing import List

from src.data.api.APIService import parse_stock_data
from src.data.dtos.StockDTO import StockDTO
from src.data.handlers.DataSourceHandler import DataSourceHandler


class APIDataSourceHandler(DataSourceHandler, ABC):
    __stock_dtos = []

    def load_stock_data(self):
        stock_data = parse_stock_data()
        for stock in stock_data:
            self.__stock_dtos.append(StockDTO(
                stock['open'],
                stock['high'],
                stock['low'],
                stock['close'],
                stock['volume'],
                stock['date'],
            ))


    def get_stock_dtos(self) -> List[StockDTO]:
        return self.__stock_dtos
