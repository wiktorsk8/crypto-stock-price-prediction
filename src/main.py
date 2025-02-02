from dotenv import load_dotenv

from src.data.enums.HandlerTypeEnum import HandlerTypeEnum
from src.data.factories import DataSourceHandlerFactory
from src.prediction_model.PredictionModel import PredictionModel

# initialize env
load_dotenv()

# load API data handler
handler = DataSourceHandlerFactory.make(HandlerTypeEnum.API)
handler.load_stock_data()

print(len(handler.get_stock_dtos()))

# initialize PredictionModel
model = PredictionModel(train_data_len=300, test_data_len=100)

# get data frame
dataFrame = model.load_data_frame(handler.get_stock_dtos())

# scale values
scaled_values = model.scale_values(dataFrame)

# prepare datasets
model.prepare_datasets(scaled_values)

# train model
model.train_model()

# get predicted prices
predicted_price = model.generate_price_for_test_data()

# display graph
model.plot_predicted_and_real_price(predicted_price=predicted_price)