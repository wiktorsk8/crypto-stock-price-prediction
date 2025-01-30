from dataclasses import dataclass


@dataclass
class StockDTO:
    open: float
    close: float
    low: float
    high: float
    volume: float
    date: str
