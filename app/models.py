from pydantic import BaseModel
from typing import Optional


class StockData(BaseModel):
    ticker: str
    name: str
    sector: str
    current_price: float
    market_cap: Optional[float]
    market_cap_category: Optional[str]
    pe_ratio: Optional[float]
    rsi: Optional[float]
    macd: Optional[float]
    macd_signal: Optional[float]
    ma50: Optional[float]
    ma200: Optional[float]
    price_vs_ma50: Optional[float]
    price_vs_ma200: Optional[float]
    volume_spike: Optional[float]
    above_ma50: Optional[bool]
    above_ma200: Optional[bool]
    golden_cross: Optional[bool]


class ScreenResponse(BaseModel):
    total: int
    results: list[StockData]


class BacktestResult(BaseModel):
    ticker: str
    strategy: str
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    num_trades: int
    win_rate_pct: float