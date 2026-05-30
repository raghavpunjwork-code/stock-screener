"""
sentiment.py - Market Buzz via price momentum & volume analysis.

Uses yfinance price/volume data (same as the screener - confirmed working on Render).
Tickers with unusual volume spikes are "trending"; price direction = sentiment.
No external news APIs or auth needed.

Optional sources (best-effort, silently skipped if blocked):
  - StockTwits trending symbols
  - NewsAPI headlines (set NEWSAPI_KEY env var)
"""

import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import concurrent.futures


# ── Watchlist ────────────────────────────────────────────────────────────────
WATCHLIST = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA",
    # Semiconductors / AI
    "AMD", "INTC", "QCOM", "AVGO", "ARM", "SMCI",
    # Software
    "NFLX", "CRM", "ADBE", "ORCL",
    # Financials
    "JPM", "BAC", "GS", "WFC", "V", "MA",
    # High-volatility / meme
    "PLTR", "COIN", "HOOD", "SOFI", "MSTR", "GME", "AMC", "RIVN", "LCID",
    # Energy / Healthcare
    "XOM", "CVX", "LLY", "UNH", "PFE",
    # ETFs
    "SPY", "QQQ",
]

# ── Cache ────────────────────────────────────────────────────────────────────
_cache: Dict = {}
CACHE_TTL = 300  # 5 minutes


# ── Price momentum fetcher ───────────────────────────────────────────────────

def _fetch_momentum(ticker: str) -> Optional[dict]:
    """Fetch 5-day price + volume data for a ticker. Returns None on failure."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="5d", interval="1d")
        if hist is None or hist.empty or len(hist) < 2:
            return None

        close = hist["Close"]
        volume = hist["Volume"]

        current_price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2])
        change_pct = (current_price - prev_price) / prev_price * 100

        # Volume spike: today vs 4-day average
        avg_vol = float(volume.iloc[:-1].mean()) if len(volume) > 1 else float(volume.iloc[-1])
        curr_vol = float(volume.iloc[-1])
        vol_spike = curr_vol / avg_vol if avg_vol > 0 else 1.0

        # 5-day trend
        week_change = (current_price - float(close.iloc[0])) / float(close.iloc[0]) * 100

        return {
            "ticker": ticker,
            "change_pct": round(change_pct, 2),
            "week_change": round(week_change, 2),
            "vol_spike": round(vol_spike, 2),
            "current_price": round(current_price, 2),
            "volume": int(curr_vol),
            "avg_volume": int(avg_vol),
        }
    except Exception:
        return None


def _sentiment_from_momentum(m: dict) -> dict:
    """Derive sentiment label and score from price change."""
    chg = m["change_pct"]
    vol = m["vol_spike"]

    # Weight by volume spike (high vol = more conviction)
    weight = min(vol / 2.0, 2.0)
    score = (chg / 10.0) * weight
    score = max(-1.0, min(1.0, score))

    if score > 0.05:
        label = "bullish"
        bull = min(abs(score), 1.0)
        bear = 0.0
    elif score < -0.05:
        label = "bearish"
        bull = 0.0
        bear = min(abs(score), 1.0)
    else:
        label = "neutral"
        bull = 0.0
        bear = 0.0

    # Percentage breakdown
    if label == "bullish":
        bull_pct = round(50 + abs(chg) * 3, 1)
        bear_pct = round(max(5, 25 - abs(chg) * 2), 1)
    elif label == "bearish":
        bear_pct = round(50 + abs(chg) * 3, 1)
        bull_pct = round(max(5, 25 - abs(chg) * 2), 1)
    else:
        bull_pct = 35.0
        bear_pct = 35.0

    bull_pct = min(bull_pct, 90.0)
    bear_pct = min(bear_pct, 90.0)
    neut_pct = round(max(0, 100 - bull_pct - bear_pct), 1)

    return {
        "label": label,
        "score": round(score, 3),
        "bull": bull,
        "bear": bear,
        "bull_pct": bull_pct,
        "bear_pct": bear_pct,
        "neut_pct": neut_pct,
    }


def _fetch_stocktwits_trending() -> List[str]:
    """Best-effort StockTwits trending. Returns [] if blocked."""
    try:
        import requests
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


# ── Main aggregator ──────────────────────────────────────────────────────────

def get_trending_sentiment(force: bool = False) -> dict:
    global _cache
    now = time.time()
    if not force and _cache and (now - _cache.get("ts", 0)) < CACHE_TTL:
        return _cache["data"]

    # Augment watchlist with StockTwits trending (best-effort)
    st_trending = _fetch_stocktwits_trending()
    extra = [t for t in st_trending if t not in WATCHLIST]
    tickers = WATCHLIST + extra[:10]

    # Fetch momentum data in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(_fetch_momentum, t): t for t in tickers}
        for future in concurrent.futures.as_completed(futures, timeout=40):
            try:
                m = future.result()
                if m is None:
                    continue
                sent = _sentiment_from_momentum(m)

                # Mentions score: volume spike is the "buzz" proxy
                mention_score = max(1, round(m["vol_spike"] * 2))

                # Feed entry
                direction = "↑" if m["change_pct"] >= 0 else "↓"
                feed_text = (
                    f"{direction} {m['change_pct']:+.2f}% today"
                    f" · Vol spike {m['vol_spike']:.1f}x"
                    f" · 5-day {m['week_change']:+.2f}%"
                )
                feed_entry = {
                    "text": feed_text,
                    "source": "Yahoo Finance",
                    "sub": "Price & Volume",
                    "sentiment": sent["label"],
                    "url": f"https://finance.yahoo.com/quote/{m['ticker']}",
                    "ts": datetime.utcnow().isoformat() + "Z",
                }

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
                    "feed": [feed_entry],
                })
            except Exception:
                pass

    # Sort by volume spike (most unusual activity first)
    results.sort(key=lambda x: x["vol_spike"], reverse=True)
    results = results[:20]

    data = {
        "trending": results,
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "posts_analyzed": len(results),
        "model": "price-momentum",
        "sources": ["Yahoo Finance", "StockTwits" if st_trending else None],
    }
    data["sources"] = [s for s in data["sources"] if s]

    _cache = {"ts": now, "data": data}
    return data


def get_ticker_sentiment(ticker: str) -> dict:
    """Single-ticker deep-dive via price momentum."""
    m = _fetch_momentum(ticker)
    if not m:
        return {
            "ticker": ticker,
            "overall": "neutral",
            "score": 0.0,
            "bullish_pct": 33.3,
            "bearish_pct": 33.3,
            "neutral_pct": 33.4,
            "articles_analyzed": 0,
            "articles": [],
            "model": "price-momentum",
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
            "title": f"{m['change_pct']:+.2f}% today, {m['vol_spike']:.1f}x volume spike",
            "url": f"https://finance.yahoo.com/quote/{ticker}",
            "source": "Yahoo Finance",
            "sentiment": sent["label"],
            "bull": sent["bull"],
            "bear": sent["bear"],
            "ts": datetime.utcnow().isoformat() + "Z",
        }],
        "model": "price-momentum",
    }
