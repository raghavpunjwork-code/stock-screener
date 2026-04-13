from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional
import io, csv, json

from app.screener import screen_stocks, get_stock_info, DEFAULT_TICKERS
from app.models import ScreenResponse, StockData, BacktestResult
from app.backtester import backtest_ma_crossover

app = FastAPI(title="Stock Screener API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "Stock Screener API running"}


@app.get("/screen", response_model=ScreenResponse)
def screen(
    min_pe: Optional[float] = None, max_pe: Optional[float] = None,
    min_rsi: Optional[float] = None, max_rsi: Optional[float] = None,
    market_cap: Optional[str] = None, above_ma50: Optional[bool] = None,
    above_ma200: Optional[bool] = None, min_volume_spike: Optional[float] = None,
    sector: Optional[str] = None, limit: int = Query(20),
):
    results = screen_stocks(DEFAULT_TICKERS, min_pe, max_pe, min_rsi, max_rsi,
                            market_cap, above_ma50, above_ma200, min_volume_spike, sector)
    return ScreenResponse(total=len(results), results=[StockData(**r) for r in results[:limit]])


@app.get("/stock/{ticker}", response_model=StockData)
def get_stock(ticker: str):
    data = get_stock_info(ticker.upper())
    if not data:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")
    return StockData(**data)


@app.get("/backtest/{ticker}", response_model=BacktestResult)
def backtest(ticker: str, period: str = "5y"):
    try:
        return BacktestResult(**backtest_ma_crossover(ticker.upper(), period))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/export")
def export(format: str = "csv", min_pe: Optional[float] = None,
           max_pe: Optional[float] = None, min_rsi: Optional[float] = None,
           max_rsi: Optional[float] = None):
    results = screen_stocks(DEFAULT_TICKERS, min_pe, max_pe, min_rsi, max_rsi)
    if format == "json":
        return StreamingResponse(io.StringIO(json.dumps(results, indent=2)),
                                 media_type="application/json")
    if not results:
        raise HTTPException(status_code=404, detail="No results")
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
    output.seek(0)
    return StreamingResponse(output, media_type="text/csv")