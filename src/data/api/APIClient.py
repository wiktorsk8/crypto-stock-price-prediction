import requests
import os

class APIClient:
    def __init__(self):
        self.__api_base_url = os.getenv('API_BASE_URL')
        self.__api_key = os.getenv('API_KEY')

    def fetch_stock_data(self, stock: str, market: str):
        params = {
            'apikey': self.__api_key,
            'function': 'DIGITAL_CURRENCY_DAILY',
            'symbol': stock,
            'market': market,

        }
        r = requests.get(url=self.__api_base_url, params=params, headers={'Accept': 'application/json'})
        return r.json()