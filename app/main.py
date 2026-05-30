from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional
import io, csv, json, asyncio, os, time
import requests as req_lib
from datetime import datetime, timedelta

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

FINNHUB_KEY = os.environ.get("FINNHUB_KEY", "")
FINNHUB_BASE = "https://finnhub.io/api/v1"

WATCHLIST = [
    "AAPL","MSFT","NVDA","TSLA","META","AMZN","GOOGL",
    "AMD","PLTR","SPY","COIN","JPM","QQQ","GME","SMCI"
]

_mkt_cache: dict = {}
_detail_cache: dict = {}
MKT_TTL = 120
DETAIL_TTL = 3600  # 1 hour — analyst data doesn't change often


def fh_get(path: str, params: dict) -> dict:
    """Call Finnhub API with key."""
    if not FINNHUB_KEY:
        raise HTTPException(503, "FINNHUB_KEY not configured")
    params["token"] = FINNHUB_KEY
    r = req_lib.get(FINNHUB_BASE + path, params=params, timeout=8)
    r.raise_for_status()
    return r.json()


# ── /market-data ─────────────────────────────────────────────────
@app.get("/market-data")
def market_data():
    """Live quotes for the buzz watchlist via Finnhub."""
    global _mkt_cache
    now = time.time()
    if _mkt_cache and (now - _mkt_cache.get("ts", 0)) < MKT_TTL:
        return _mkt_cache["data"]

    results = []
    for ticker in WATCHLIST:
        try:
            d = fh_get("/quote", {"symbol": ticker})
            if d and d.get("c"):
                results.append({
                    "ticker": ticker,
                    "price": d["c"],
                    "prev_close": d["pc"],
                    "change_pct": d["dp"],
                    "high": d["h"],
                    "low": d["l"],
                    "open": d["o"],
                })
        except Exception:
            pass

    if not results:
        raise HTTPException(503, "No data from Finnhub")

    data = {"quotes": results, "source": "Finnhub", "ts": now}
    _mkt_cache = {"ts": now, "data": data}
    return data


# ── /stock-detail/{ticker} ────────────────────────────────────────
@app.get("/stock-detail/{ticker}")
def stock_detail(ticker: str):
    """
    Analyst price targets, recommendation trends, key metrics, and
    recent news for a single ticker — all via Finnhub free tier.
    Cached for 1 hour.
    """
    t = ticker.upper()
    global _detail_cache
    now = time.time()
    if t in _detail_cache and (now - _detail_cache[t].get("ts", 0)) < DETAIL_TTL:
        return _detail_cache[t]["data"]

    result = {"ticker": t}

    # 1. Price target
    try:
        pt = fh_get("/stock/price-target", {"symbol": t})
        current = None
        try:
            q = fh_get("/quote", {"symbol": t})
            current = q.get("c")
        except Exception:
            pass
        upside = None
        if current and pt.get("targetMean"):
            upside = round((pt["targetMean"] - current) / current * 100, 1)
        result["price_target"] = {
            "mean": pt.get("targetMean"),
            "high": pt.get("targetHigh"),
            "low": pt.get("targetLow"),
            "upside_pct": upside,
            "last_updated": pt.get("lastUpdated"),
        }
    except Exception:
        result["price_target"] = None

    # 2. Analyst recommendations (most recent period)
    try:
        recs = fh_get("/stock/recommendation", {"symbol": t})
        if recs:
            latest = recs[0]
            sb = latest.get("strongBuy", 0)
            b  = latest.get("buy", 0)
            h  = latest.get("hold", 0)
            s  = latest.get("sell", 0)
            ss = latest.get("strongSell", 0)
            total = sb + b + h + s + ss
            bull_pct = round((sb + b) / total * 100) if total else 0
            bear_pct = round((s + ss) / total * 100) if total else 0
            # consensus label
            if bull_pct >= 60:
                consensus = "Strong Buy" if sb / max(total,1) > 0.3 else "Buy"
            elif bear_pct >= 50:
                consensus = "Sell"
            else:
                consensus = "Hold"
            result["recommendations"] = {
                "strong_buy": sb, "buy": b, "hold": h,
                "sell": s, "strong_sell": ss,
                "total": total,
                "bull_pct": bull_pct,
                "bear_pct": bear_pct,
                "consensus": consensus,
                "period": latest.get("period"),
            }
    except Exception:
        result["recommendations"] = None

    # 3. Key metrics
    try:
        m = fh_get("/stock/metric", {"symbol": t, "metric": "all"})
        met = m.get("metric", {})
        result["metrics"] = {
            "pe_ttm":             met.get("peBasicExclExtraTTM"),
            "eps_ttm":            met.get("epsBasicExclExtraTTM"),
            "revenue_growth_yoy": met.get("revenueGrowthTTMYoy"),
            "gross_margin":       met.get("grossMarginTTM"),
            "week52_high":        met.get("52WeekHigh"),
            "week52_low":         met.get("52WeekLow"),
            "beta":               met.get("beta"),
            "market_cap":         met.get("marketCapitalization"),
            "dividend_yield":     met.get("dividendYieldIndicatedAnnual"),
            "rsi14":              met.get("rsi14"),
        }
    except Exception:
        result["metrics"] = None

    # 4. Recent news (last 7 days, up to 5 articles)
    try:
        to_date   = datetime.utcnow().strftime("%Y-%m-%d")
        from_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
        news = fh_get("/company-news", {"symbol": t, "from": from_date, "to": to_date})
        result["news"] = [
            {
                "headline": n.get("headline", ""),
                "source":   n.get("source", ""),
                "url":      n.get("url", ""),
                "datetime": n.get("datetime", 0),
                "summary":  (n.get("summary") or "")[:200],
            }
            for n in (news or [])[:5]
        ]
    except Exception:
        result["news"] = []

    _detail_cache[t] = {"ts": now, "data": result}
    return result


# ── existing endpoints ────────────────────────────────────────────
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
