from dataclasses import dataclass


@dataclass
class StockDTO:
    date: str
    open: float
    low: float
    high: float
    volume: float
    close: float

