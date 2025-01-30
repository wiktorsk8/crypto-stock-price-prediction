from abc import ABC, abstractmethod
from typing import List

from src.data.dtos.StockDTO import StockDTO


# Define the interface
class DataSourceHandler(ABC):
    @abstractmethod
    def load_stock_data(self):
        """Method that must be implemented by subclasses"""
        pass

    @abstractmethod
    def get_stock_dtos(self) -> List[StockDTO]:
        pass