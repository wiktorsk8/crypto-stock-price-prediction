from dotenv import load_dotenv

from src.data.enums.HandlerTypeEnum import HandlerTypeEnum
from src.data.factories import DataSourceHandlerFactory

load_dotenv()

handler = DataSourceHandlerFactory.make(HandlerTypeEnum.API)
handler.load_stock_data()
