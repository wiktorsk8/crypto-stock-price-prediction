from src.data.enums.HandlerTypeEnum import HandlerTypeEnum
from src.data.handlers.APIDataSourceHandler import APIDataSourceHandler
from src.data.handlers.DataSourceHandler import DataSourceHandler


def make(factory_type: HandlerTypeEnum)-> DataSourceHandler:
    if factory_type == HandlerTypeEnum.API:
        return APIDataSourceHandler()

