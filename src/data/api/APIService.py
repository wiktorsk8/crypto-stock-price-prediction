from src.data.api.APIClient import APIClient


def parse_stock_data():
    api = APIClient()
    stock_data = api.fetch_stock_data(stock='BTC', market='USD')
    parsed_data = []
    for date, stock in stock_data['Time Series (Digital Currency Daily)'].items():
        parsed_data.append({
            'date': date,
            'open': stock['1. open'],
            'high': stock['2. high'],
            'low': stock['3. low'],
            'close': stock['4. close'],
            'volume': stock['5. volume']
        })

    return parsed_data
