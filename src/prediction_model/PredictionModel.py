from dataclasses import dataclass, asdict
from typing import List

from keras import Sequential
from keras.src.layers import Dropout, LSTM, Dense
from matplotlib import pyplot as plt
from pandas import DataFrame

from src.data.dtos.StockDTO import StockDTO
from src.data.handlers.DataSourceHandler import DataSourceHandler
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class PredictionModel():
    def __init__(self, train_data_len, test_data_len):
        self.train_data_len = train_data_len
        self.test_data_len = test_data_len

        self.__train_previous_days_sequence = np.ndarray
        self.__train_next_day_stock_price = np.ndarray

        self.__test_previous_days_sequence = np.ndarray
        self.__test_next_day_stock_price = np.ndarray

        self.__regressor = Sequential()

        self.__scaler = MinMaxScaler(feature_range=(0,1))

        self.stocks = []

    def load_data_frame(self, stocks: List[StockDTO]) -> DataFrame:
        stocks.reverse()
        self.stocks = stocks
        return pd.DataFrame([asdict(stock) for stock in stocks])


    def scale_values(self, data_frame: DataFrame) -> np.ndarray:
        dataset = data_frame
        train_set = dataset.iloc[:, 3:4].values
        return self.__scaler.fit_transform(train_set)


    def prepare_datasets(self, data_scaled):
        x_train = []
        y_train = []
        window = 10

        # 300 records out of 350 used for training
        for i in range(window, self.train_data_len):
            x_train.append(data_scaled[i - window:i, 0])
            y_train.append(data_scaled[i, 0])

        self.__train_previous_days_sequence, self.__train_next_day_stock_price = np.array(x_train), np.array(y_train)

        x_test = []
        y_test = []
        for i in range(self.train_data_len, self.train_data_len + self.test_data_len):
            x_test.append(data_scaled[i - window:i, 0])
            y_test.append(data_scaled[i, 0])

        self.__test_previous_days_sequence, self.__test_next_day_stock_price = np.array(x_test), np.array(y_test)


    def train_model(self):
        x_train = np.reshape(
            self.__train_previous_days_sequence,
            (self.__train_previous_days_sequence.shape[0], self.__train_previous_days_sequence.shape[1],
             1)
        )

        self.__regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        self.__regressor.add(Dropout(0.2))
        self.__regressor.add(LSTM(units=50, return_sequences=True))
        self.__regressor.add(Dropout(0.2))
        self.__regressor.add(LSTM(units=50, return_sequences=True))
        self.__regressor.add(Dropout(0.2))
        self.__regressor.add(LSTM(units=50))
        self.__regressor.add(Dropout(0.2))
        self.__regressor.add(Dense(units=1))
        self.__regressor.compile(optimizer='adam', loss='mean_squared_error')

        self.__regressor.fit(x_train, self.__train_next_day_stock_price, epochs=10, batch_size=32)


    def generate_price_for_test_data(self):
        predicted_price = self.__regressor.predict(self.__test_previous_days_sequence)
        predicted_price = self.__scaler.inverse_transform(predicted_price)
        return predicted_price


    def plot_predicted_and_real_price(self, predicted_price):
        # print(self.stocks[-50], self.stocks[-1])

        real_price = self.__scaler.inverse_transform(self.__test_next_day_stock_price.reshape(-1, 1))
        plt.plot(real_price, color='orange', label='real price')
        plt.plot(predicted_price, color='green', label='predicted price')
        plt.title('BTC stock price prediction')
        plt.xlabel('time')
        plt.ylabel('stock price')
        plt.legend()
        plt.show()
        print(f'RMSE {metrics.mean_squared_error(predicted_price, real_price, squared=False)}')
        print(f'MAE {metrics.mean_absolute_error(predicted_price, real_price)}')