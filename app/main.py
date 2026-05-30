from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional
import io, csv, json, asyncio

from app.screener import screen_stocks, get_stock_info, DEFAULT_TICKERS
from app.models import ScreenResponse, StockData, BacktestResult
from app.backtester import backtest_ma_crossover
from app.sentiment import get_trending_sentiment, get_ticker_sentiment

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
        raise HTTPException(status_code=404, detail="No data for " + ticker)
    return StockData(**data)


@app.get("/backtest/{ticker}", response_model=BacktestResult)
def backtest(ticker: str, period: str = "5y"):
    try:
        return BacktestResult(**backtest_ma_crossover(ticker.upper(), period))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/trending")
def trending(force: bool = False):
    return get_trending_sentiment(force=force)


@app.get("/trending/stream")
async def trending_stream():
    async def generator():
        while True:
            try:
                data = get_trending_sentiment()
                yield "data: " + json.dumps(data) + "\n\n"
            except Exception as e:
                yield "data: " + json.dumps({"error": str(e)}) + "\n\n"
            await asyncio.sleep(60)
    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.get("/sentiment/{ticker}")
def ticker_sentiment(ticker: str):
    try:
        return get_ticker_sentiment(ticker.upper())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@app.get("/debug/yfinance")
def debug_yfinance():
    """Test yfinance connectivity - shows exact errors."""
    import traceback
    results = {}
    try:
        import yfinance as yf
        results["import"] = "ok"
        try:
            t = yf.Ticker("AAPL")
            h = t.history(period="2d")
            results["history_rows"] = len(h)
            if not h.empty:
                results["latest_close"] = float(h["Close"].iloc[-1])
            else:
                results["history_error"] = "empty dataframe"
        except Exception as e:
            results["history_error"] = str(e)
            results["history_traceback"] = traceback.format_exc()[-500:]
    except Exception as e:
        results["import_error"] = str(e)
    return results
