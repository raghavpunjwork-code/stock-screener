"""
sentiment.py - Market Buzz via price momentum using Stooq API.

Stooq (stooq.com) provides free OHLCV data via CSV, works from any server,
no authentication, no IP blocking. Falls back to yfinance if Stooq fails.
"""

import os
import time
import csv
import io
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import concurrent.futures

# ── Watchlist ────────────────────────────────────────────────────────────────
WATCHLIST_FULL = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA",
    "AMD", "INTC", "QCOM", "AVGO", "ARM", "SMCI",
    "NFLX", "CRM", "ADBE", "ORCL",
    "JPM", "BAC", "GS", "WFC", "V", "MA",
    "PLTR", "COIN", "HOOD", "SOFI", "MSTR", "GME", "RIVN",
    "XOM", "CVX", "LLY", "UNH", "PFE",
    "SPY", "QQQ",
]

_cache: Dict = {}
CACHE_TTL = 300

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; StockScreener/1.0)",
    "Accept": "text/csv,text/plain,*/*",
}


def _fetch_stooq(ticker: str) -> Optional[dict]:
    """Fetch recent price data from Stooq CSV API."""
    symbol = ticker.lower() + ".us"
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    try:
        r = requests.get(url, headers=HEADERS, timeout=5)
        if r.status_code != 200 or len(r.text) < 50:
            return None

        lines = [l for l in r.text.strip().splitlines() if l and "No data" not in l]
        if len(lines) < 3:
            return None

        reader = csv.DictReader(io.StringIO("\n".join(lines)))
        rows = list(reader)
        # Stooq returns newest-first; take last 5 trading days
        rows = [r for r in rows if r.get("Close") and r["Close"] != "null"][-6:]
        if len(rows) < 2:
            return None

        current_close = float(rows[-1]["Close"])
        prev_close = float(rows[-2]["Close"])
        open_5d = float(rows[0]["Close"])
        change_pct = (current_close - prev_close) / prev_close * 100
        week_change = (current_close - open_5d) / open_5d * 100

        curr_vol = float(rows[-1].get("Volume") or 0)
        vols = [float(r.get("Volume") or 0) for r in rows[:-1] if r.get("Volume")]
        avg_vol = sum(vols) / len(vols) if vols else curr_vol
        vol_spike = curr_vol / avg_vol if avg_vol > 0 else 1.0

        return {
            "ticker": ticker,
            "change_pct": round(change_pct, 2),
            "week_change": round(week_change, 2),
            "vol_spike": round(vol_spike, 2),
            "current_price": round(current_close, 2),
            "volume": int(curr_vol),
        }
    except Exception:
        return None


def _fetch_yfinance_fallback(ticker: str) -> Optional[dict]:
    """Fallback: try yfinance if Stooq fails."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="5d")
        if hist is None or hist.empty or len(hist) < 2:
            return None
        close = hist["Close"]
        volume = hist["Volume"]
        current_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        change_pct = (current_close - prev_close) / prev_close * 100
        week_change = (current_close - float(close.iloc[0])) / float(close.iloc[0]) * 100
        curr_vol = float(volume.iloc[-1])
        avg_vol = float(volume.iloc[:-1].mean()) if len(volume) > 1 else curr_vol
        vol_spike = curr_vol / avg_vol if avg_vol > 0 else 1.0
        return {
            "ticker": ticker,
            "change_pct": round(change_pct, 2),
            "week_change": round(week_change, 2),
            "vol_spike": round(vol_spike, 2),
            "current_price": round(current_close, 2),
            "volume": int(curr_vol),
        }
    except Exception:
        return None


def _fetch_momentum(ticker: str) -> Optional[dict]:
    m = _fetch_stooq(ticker)
    if m is None:
        m = _fetch_yfinance_fallback(ticker)
    return m


def _sentiment_from_momentum(m: dict) -> dict:
    chg = m["change_pct"]
    vol = m["vol_spike"]
    weight = min(vol / 2.0, 2.0)
    score = max(-1.0, min(1.0, (chg / 10.0) * weight))

    if score > 0.05:
        label = "bullish"
        bull_pct = min(round(50 + abs(chg) * 3, 1), 90.0)
        bear_pct = round(max(5, 25 - abs(chg) * 2), 1)
    elif score < -0.05:
        label = "bearish"
        bear_pct = min(round(50 + abs(chg) * 3, 1), 90.0)
        bull_pct = round(max(5, 25 - abs(chg) * 2), 1)
    else:
        label = "neutral"
        bull_pct = 35.0
        bear_pct = 35.0

    neut_pct = round(max(0, 100 - bull_pct - bear_pct), 1)
    return {
        "label": label,
        "score": round(score, 3),
        "bull_pct": bull_pct,
        "bear_pct": bear_pct,
        "neut_pct": neut_pct,
    }


def _fetch_stocktwits_trending() -> List[str]:
    try:
        r = requests.get(
            "https://api.stocktwits.com/api/2/trending/symbols.json",
            headers={"User-Agent": "StockScreener/1.0"},
            timeout=6,
        )
        if r.status_code == 200:
            return [s["symbol"] for s in r.json().get("symbols", []) if s.get("symbol")][:15]
    except Exception:
        pass
    return []


def get_trending_sentiment(force: bool = False) -> dict:
    global _cache
    now = time.time()
    if not force and _cache and (now - _cache.get("ts", 0)) < CACHE_TTL:
        return _cache["data"]

    st_trending = _fetch_stocktwits_trending()
    extra = [t for t in st_trending if t not in WATCHLIST_FULL]
    tickers = WATCHLIST_FULL + extra[:10]

    results = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_fetch_momentum, t): t for t in tickers}
            for future in concurrent.futures.as_completed(futures, timeout=20):
                try:
                    m = future.result()
                    if m is None:
                        continue
                    sent = _sentiment_from_momentum(m)
                    mention_score = max(1, round(m["vol_spike"] * 2))
                    direction = "↑" if m["change_pct"] >= 0 else "↓"
                    feed_text = (
                        f"{direction} {m['change_pct']:+.2f}% today"
                        f" · Vol spike {m['vol_spike']:.1f}x"
                        f" · 5-day {m['week_change']:+.2f}%"
                    )
                    results.append({
                        "ticker": m["ticker"],
                        "mentions": mention_score,
                        "reddit_mentions": 0,
                        "stocktwits_mentions": 1 if m["ticker"] in st_trending else 0,
                        "news_mentions": 0,
                        "bullish_pct": sent["bull_pct"],
                        "bearish_pct": sent["bear_pct"],
                        "neutral_pct": sent["neut_pct"],
                        "overall": sent["label"],
                        "score": sent["score"],
                        "change_pct": m["change_pct"],
                        "vol_spike": m["vol_spike"],
                        "price": m["current_price"],
                        "feed": [{
                            "text": feed_text,
                            "source": "Stooq / Yahoo Finance",
                            "sub": "Price & Volume",
                            "sentiment": sent["label"],
                            "url": f"https://finance.yahoo.com/quote/{m['ticker']}",
                            "ts": datetime.utcnow().isoformat() + "Z",
                        }],
                    })
                except Exception:
                    pass
    except Exception:
        pass  # TimeoutError or other — return whatever we collected

    results.sort(key=lambda x: x["vol_spike"], reverse=True)
    results = results[:20]

    data = {
        "trending": results,
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "posts_analyzed": len(results),
        "model": "price-momentum",
        "sources": ["Stooq", "StockTwits" if st_trending else None],
    }
    data["sources"] = [s for s in data["sources"] if s]
    _cache = {"ts": now, "data": data}
    return data


def get_ticker_sentiment(ticker: str) -> dict:
    m = _fetch_momentum(ticker)
    if not m:
        return {
            "ticker": ticker, "overall": "neutral", "score": 0.0,
            "bullish_pct": 33.3, "bearish_pct": 33.3, "neutral_pct": 33.4,
            "articles_analyzed": 0, "articles": [], "model": "price-momentum",
        }
    sent = _sentiment_from_momentum(m)
    return {
        "ticker": ticker,
        "overall": sent["label"],
        "score": sent["score"],
        "bullish_pct": sent["bull_pct"],
        "bearish_pct": sent["bear_pct"],
        "neutral_pct": sent["neut_pct"],
        "change_pct": m["change_pct"],
        "week_change": m["week_change"],
        "vol_spike": m["vol_spike"],
        "price": m["current_price"],
        "articles_analyzed": 1,
        "articles": [{
            "title": f"{m['change_pct']:+.2f}% today, {m['vol_spike']:.1f}x volume",
            "url": f"https://finance.yahoo.com/quote/{ticker}",
            "source": "Stooq",
            "sentiment": sent["label"],
            "bull": max(sent["score"], 0),
            "bear": max(-sent["score"], 0),
            "ts": datetime.utcnow().isoformat() + "Z",
        }],
        "model": "price-momentum",
    }
