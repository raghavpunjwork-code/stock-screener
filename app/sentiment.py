"""
sentiment.py - Stock sentiment engine using Yahoo Finance news via yfinance.

Sources:
  - Yahoo Finance news (via yfinance) - works from any server, no auth needed
  - StockTwits trending symbols - public API (fallback if blocked)
  - NewsAPI - optional; set NEWSAPI_KEY env var

Model:
  - VADER sentiment (vaderSentiment) - lightweight, no GPU needed
  - FinBERT fallback if transformers/torch installed (not on Render free tier)
"""

import os
import time
import json
from datetime import datetime
from typing import List, Dict, Optional
from functools import lru_cache

# ── Sentiment model ──────────────────────────────────────────────────────────

_pipeline = None
_model_status = "unloaded"


def _load_pipeline():
    global _pipeline, _model_status
    if _model_status != "unloaded":
        return
    _model_status = "loading"
    try:
        from transformers import pipeline as hf_pipeline
        _pipeline = hf_pipeline("text-classification", model="ProsusAI/finbert", truncation=True)
        _model_status = "finbert"
    except Exception:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _pipeline = SentimentIntensityAnalyzer()
            _model_status = "vader"
        except Exception:
            _model_status = "error"


@lru_cache(maxsize=1024)
def _run_sentiment(text: str) -> dict:
    if _model_status == "unloaded":
        _load_pipeline()
    try:
        if _model_status == "finbert":
            result = _pipeline(text[:512])[0]
            label = result["label"].lower()
            score = result["score"]
            return {
                "label": "positive" if label == "positive" else "negative" if label == "negative" else "neutral",
                "bull": score if label == "positive" else 0.0,
                "bear": score if label == "negative" else 0.0,
            }
        elif _model_status == "vader":
            scores = _pipeline.polarity_scores(text)
            compound = scores["compound"]
            return {
                "label": "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral",
                "bull": max(compound, 0),
                "bear": max(-compound, 0),
            }
    except Exception:
        pass
    return {"label": "neutral", "bull": 0.0, "bear": 0.0}


# ── Watchlist ────────────────────────────────────────────────────────────────
# Tickers we always scan - mix of large caps + high-buzz names
WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AMD",
    "NFLX", "CRM", "ADBE", "INTC", "QCOM", "AVGO", "ARM",
    "JPM", "BAC", "GS", "WFC",
    "SPY", "QQQ",
    "PLTR", "COIN", "HOOD", "SOFI", "SMCI", "MSTR",
    "GME", "AMC", "RIVN", "LCID",
    "XOM", "CVX", "LLY", "UNH", "JNJ",
]

# ── Yahoo Finance news fetcher ───────────────────────────────────────────────

def _fetch_yf_news(ticker: str) -> List[dict]:
    """Fetch recent Yahoo Finance news headlines for a ticker via yfinance."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker)
        news = info.news or []
        results = []
        for item in news[:8]:
            content = item.get("content", {})
            title = content.get("title", "") or item.get("title", "")
            summary = content.get("summary", "") or ""
            url = ""
            cp = content.get("canonicalUrl", {})
            if isinstance(cp, dict):
                url = cp.get("url", "")
            if not url:
                url = "https://finance.yahoo.com/quote/" + ticker
            pub_date = content.get("pubDate", "") or ""
            if title:
                results.append({
                    "title": title,
                    "summary": summary[:200],
                    "url": url,
                    "ts": pub_date,
                    "source": "Yahoo Finance",
                    "ticker": ticker,
                })
        return results
    except Exception as e:
        return []


def _fetch_stocktwits_trending() -> List[str]:
    """Fetch trending symbols from StockTwits (best effort)."""
    try:
        import requests as req
        r = req.get(
            "https://api.stocktwits.com/api/2/trending/symbols.json",
            headers={"User-Agent": "StockScreener/1.0"},
            timeout=8,
        )
        if r.status_code == 200:
            symbols = r.json().get("symbols", [])
            return [s["symbol"] for s in symbols if s.get("symbol")][:20]
    except Exception:
        pass
    return []


# ── Cache ────────────────────────────────────────────────────────────────────

_cache: Dict = {}
CACHE_TTL = 300  # 5 minutes (news doesn't change every minute)


def get_trending_sentiment(force: bool = False) -> dict:
    """
    Scan the watchlist via Yahoo Finance news, score sentiment, return top tickers.
    Results cached for CACHE_TTL seconds.
    """
    global _cache
    if not force and _cache and (time.time() - _cache.get("ts", 0)) < CACHE_TTL:
        return _cache["data"]

    _load_pipeline()

    # Try to augment watchlist with StockTwits trending
    st_trending = _fetch_stocktwits_trending()
    tickers_to_scan = list(dict.fromkeys(WATCHLIST + [t for t in st_trending if t not in WATCHLIST]))

    bucket = {}

    def ensure(t):
        if t not in bucket:
            bucket[t] = {
                "ticker": t,
                "mentions": 0,
                "news_count": 0,
                "stocktwits_count": 0,
                "pos": 0, "neg": 0, "neu": 0,
                "score_sum": 0.0,
                "feed": [],
            }
        return bucket[t]

    # Add StockTwits trending as a signal
    for sym in st_trending:
        d = ensure(sym)
        d["mentions"] += 1
        d["stocktwits_count"] += 1
        d["pos"] += 1
        d["score_sum"] += 0.5

    # Fetch Yahoo Finance news for each ticker
    import concurrent.futures
    def scan_ticker(ticker):
        return ticker, _fetch_yf_news(ticker)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(scan_ticker, t): t for t in tickers_to_scan}
        for future in concurrent.futures.as_completed(futures, timeout=45):
            try:
                ticker, articles = future.result()
                if not articles:
                    continue
                d = ensure(ticker)
                for article in articles:
                    text = article["title"] + " " + article.get("summary", "")
                    sent = _run_sentiment(text.strip()[:512])
                    d["mentions"] += 1
                    d["news_count"] += 1
                    d["score_sum"] += sent["bull"] - sent["bear"]
                    if sent["label"] == "positive":
                        d["pos"] += 1
                    elif sent["label"] == "negative":
                        d["neg"] += 1
                    else:
                        d["neu"] += 1
                    if len(d["feed"]) < 4:
                        d["feed"].append({
                            "text": article["title"][:140],
                            "source": article["source"],
                            "sub": "",
                            "sentiment": sent["label"],
                            "url": article["url"],
                            "ts": article["ts"],
                        })
            except Exception:
                pass

    # Build results
    results = []
    for d in bucket.values():
        if d["mentions"] == 0:
            continue
        total = d["pos"] + d["neg"] + d["neu"] or 1
        score = d["score_sum"] / max(d["mentions"], 1)
        results.append({
            "ticker": d["ticker"],
            "mentions": d["mentions"],
            "reddit_mentions": 0,
            "stocktwits_mentions": d["stocktwits_count"],
            "news_mentions": d["news_count"],
            "bullish_pct": round(d["pos"] / total * 100, 1),
            "bearish_pct": round(d["neg"] / total * 100, 1),
            "neutral_pct": round(d["neu"] / total * 100, 1),
            "overall": "bullish" if score > 0.05 else "bearish" if score < -0.05 else "neutral",
            "score": round(score, 3),
            "feed": d["feed"][:3],
        })

    results.sort(key=lambda x: x["mentions"], reverse=True)
    results = results[:20]

    total_posts = sum(r["news_mentions"] for r in results)
    data = {
        "trending": results,
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "posts_analyzed": total_posts,
        "model": _model_status,
        "sources": ["Yahoo Finance News", "StockTwits" if st_trending else None],
    }
    data["sources"] = [s for s in data["sources"] if s]

    _cache = {"ts": time.time(), "data": data}
    return data


def get_ticker_sentiment(ticker: str) -> dict:
    """Deep-dive sentiment for a single ticker using Yahoo Finance news."""
    _load_pipeline()
    articles = _fetch_yf_news(ticker)
    scored = []
    pos = neg = neu = 0
    score_sum = 0.0
    for article in articles:
        text = (article["title"] + " " + article.get("summary", "")).strip()[:512]
        sent = _run_sentiment(text)
        score_sum += sent["bull"] - sent["bear"]
        if sent["label"] == "positive":
            pos += 1
        elif sent["label"] == "negative":
            neg += 1
        else:
            neu += 1
        scored.append({
            "title": article["title"],
            "url": article["url"],
            "source": article["source"],
            "sentiment": sent["label"],
            "bull": round(sent["bull"], 3),
            "bear": round(sent["bear"], 3),
            "ts": article["ts"],
        })
    total = pos + neg + neu or 1
    avg = score_sum / max(len(articles), 1)
    return {
        "ticker": ticker,
        "overall": "bullish" if avg > 0.05 else "bearish" if avg < -0.05 else "neutral",
        "score": round(avg, 3),
        "bullish_pct": round(pos / total * 100, 1),
        "bearish_pct": round(neg / total * 100, 1),
        "neutral_pct": round(neu / total * 100, 1),
        "articles_analyzed": len(articles),
        "articles": scored,
        "model": _model_status,
    }
