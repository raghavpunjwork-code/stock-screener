import yfinance as yf
import pandas as pd
from typing import Optional
from app.indicators import compute_rsi, compute_macd, compute_moving_averages

DEFAULT_TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM",
    "JNJ","V","PG","MA","HD","CVX","MRK","ABBV","PEP","KO",
    "AVGO","COST","MCD","WMT","BAC","DIS","ADBE","CRM","NFLX",
    "INTC","AMD","PYPL","QCOM","TXN","NEE","UNH","TMO","AMGN",
    "HON","LOW","IBM","GS","CAT","SBUX","BA","MMM","BRK-B","NKE",
]


def get_stock_info(ticker: str) -> Optional[dict]:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty or len(hist) < 50:
            return None
        info = stock.info
        close = hist["Close"]
        rsi = compute_rsi(close)
        macd, signal = compute_macd(close)
        ma50, ma200 = compute_moving_averages(close)
        current_price = float(close.iloc[-1])
        avg_volume = float(hist["Volume"].tail(20).mean())
        latest_volume = float(hist["Volume"].iloc[-1])
        volume_spike = latest_volume / avg_volume if avg_volume > 0 else 1.0
        market_cap = info.get("marketCap")
        pe_ratio = info.get("trailingPE")

        def safe(series):
            v = series.iloc[-1]
            return float(v) if not pd.isna(v) else None

        return {
            "ticker": ticker,
            "name": info.get("shortName", ticker),
            "sector": info.get("sector", "Unknown"),
            "current_price": round(current_price, 2),
            "market_cap": market_cap,
            "market_cap_category": "large" if market_cap and market_cap >= 10e9 else "mid" if market_cap and market_cap >= 2e9 else "small",
            "pe_ratio": round(pe_ratio, 2) if pe_ratio else None,
            "rsi": round(safe(rsi), 2) if safe(rsi) else None,
            "macd": round(safe(macd), 4) if safe(macd) else None,
            "macd_signal": round(safe(signal), 4) if safe(signal) else None,
            "ma50": round(safe(ma50), 2) if safe(ma50) else None,
            "ma200": round(safe(ma200), 2) if safe(ma200) else None,
            "price_vs_ma50": round((current_price - safe(ma50)) / safe(ma50) * 100, 2) if safe(ma50) else None,
            "price_vs_ma200": round((current_price - safe(ma200)) / safe(ma200) * 100, 2) if safe(ma200) else None,
            "volume_spike": round(volume_spike, 2),
            "above_ma50": current_price > safe(ma50) if safe(ma50) else None,
            "above_ma200": current_price > safe(ma200) if safe(ma200) else None,
            "golden_cross": False,
        }
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None


def screen_stocks(tickers=DEFAULT_TICKERS, min_pe=None, max_pe=None,
                  min_rsi=None, max_rsi=None, market_cap=None,
                  above_ma50=None, above_ma200=None,
                  min_volume_spike=None, sector=None):
    results = []
    for ticker in tickers:
        data = get_stock_info(ticker)
        if not data:
            continue
        if min_pe and (not data["pe_ratio"] or data["pe_ratio"] < min_pe): continue
        if max_pe and (not data["pe_ratio"] or data["pe_ratio"] > max_pe): continue
        if min_rsi and (not data["rsi"] or data["rsi"] < min_rsi): continue
        if max_rsi and (not data["rsi"] or data["rsi"] > max_rsi): continue
        if market_cap and data["market_cap_category"] != market_cap: continue
        if above_ma50 is not None and data["above_ma50"] != above_ma50: continue
        if above_ma200 is not None and data["above_ma200"] != above_ma200: continue
        if min_volume_spike and (not data["volume_spike"] or data["volume_spike"] < min_volume_spike): continue
        if sector and data["sector"].lower() != sector.lower(): continue
        results.append(data)
    return results