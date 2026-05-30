"""
sentiment.py — Social sentiment engine for trending stocks.

Sources:
  - Reddit (r/wallstreetbets, r/stocks, r/investing, r/stockmarket) — public JSON API, no auth
  - StockTwits — public trending + per-ticker stream API, no auth
  - NewsAPI — optional; set NEWSAPI_KEY env var (free tier: newsapi.org)
  - Twitter/X — requires paid API ($100+/mo). Set TWITTER_BEARER_TOKEN env var to enable.

Model:
  - FinBERT (ProsusAI/finbert) — loaded lazily on first request, ~400 MB download.
  - VADER fallback if transformers/torch not installed.
"""

import re
import os
import json
import time
import requests
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from collections import defaultdict
from functools import lru_cache

# ──────────────────────────────────────────────
# Sentiment model (FinBERT, lazy-loaded)
# ──────────────────────────────────────────────

_pipeline = None
_model_status = "unloaded"   # "unloaded" | "loading" | "finbert" | "vader" | "error"


def _load_pipeline():
    global _pipeline, _model_status
    if _model_status in ("finbert", "vader"):
        return
    _model_status = "loading"
    try:
        from transformers import pipeline as hf_pipeline
        _pipeline = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            max_length=512,
            truncation=True,
        )
        _model_status = "finbert"
        print("[sentiment] FinBERT loaded.")
    except Exception as e:
        print(f"[sentiment] FinBERT unavailable ({e}). Falling back to VADER.")
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _pipeline = SentimentIntensityAnalyzer()
            _model_status = "vader"
            print("[sentiment] VADER loaded.")
        except Exception as e2:
            print(f"[sentiment] VADER unavailable ({e2}). Sentiment disabled.")
            _model_status = "error"


@lru_cache(maxsize=1024)
def _run_sentiment(text: str) -> tuple:
    """Cache-friendly: returns (label, bull_score, bear_score)."""
    if _model_status == "unloaded":
        _load_pipeline()

    if _model_status == "finbert":
        try:
            result = _pipeline(text[:512])[0]
            label = result["label"]       # "positive" | "negative" | "neutral"
            score = float(result["score"])
            bull = score if label == "positive" else 0.0
            bear = score if label == "negative" else 0.0
            return label, bull, bear
        except Exception:
            pass

    if _model_status == "vader":
        try:
            scores = _pipeline.polarity_scores(text)
            c = scores["compound"]
            if c >= 0.05:
                return "positive", abs(c), 0.0
            elif c <= -0.05:
                return "negative", 0.0, abs(c)
            else:
                return "neutral", 0.0, 0.0
        except Exception:
            pass

    return "neutral", 0.0, 0.0


def analyze_sentiment(text: str) -> Dict:
    label, bull, bear = _run_sentiment(text.strip()[:400])
    return {"label": label, "bullish": bull, "bearish": bear}


# ──────────────────────────────────────────────
# Ticker extraction
# ──────────────────────────────────────────────

_IGNORE = {
    "I","A","AN","THE","AND","OR","BUT","NOT","FOR","WITH","FROM","THIS","THAT",
    "THEY","WILL","HAVE","BEEN","WERE","INTO","ARE","WAS","HAS","HAD","HOW",
    "WHO","WHY","ALL","ANY","NEW","OUT","NOW","UP","SO","DO","GO","ON","AT",
    "IN","IT","IS","IF","AS","BE","BY","TO","OF","MY","WE","HE","SHE","YOU",
    "ETF","CEO","IPO","ATH","EPS","PE","YTD","YOY","QOQ","DD","WSB","IMO",
    "EOD","YOLO","FOMO","BUY","SELL","LOL","OMG","WTF","HODL","AFAIK","TBH",
    "USA","US","UK","EU","AI","EV","FED","SEC","NYSE","NASDAQ","GDP","CPI",
    "FDA","IRS","CNBC","FOMC","SPY","QQQ","DIA","IWM",   # ETFs — tracked separately
}


def extract_tickers(text: str, known: set = None) -> List[str]:
    """Extract tickers via $TICKER pattern and known list."""
    upper = text.upper()
    found = set()
    # $TICKER (most reliable signal on social media)
    found.update(re.findall(r"\$([A-Z]{1,5})\b", upper))
    # Match against known tickers in text
    if known:
        for t in known:
            if re.search(rf"\b{re.escape(t)}\b", upper):
                found.add(t)
    return [t for t in found if t not in _IGNORE and 1 < len(t) <= 5]


# ──────────────────────────────────────────────
# Reddit scraper (public JSON, no auth needed)
# ──────────────────────────────────────────────

_REDDIT_HEADERS = {"User-Agent": "StockScreener/2.0 sentiment-analysis-bot"}
_SUBREDDITS = ["wallstreetbets", "stocks", "investing", "stockmarket"]


def fetch_reddit_posts(limit: int = 100) -> List[Dict]:
    posts = []
    per_sub = max(limit // len(_SUBREDDITS), 10)
    for sub in _SUBREDDITS:
        try:
            url = f"https://www.reddit.com/r/{sub}/hot.json?limit={per_sub}"
            r = requests.get(url, headers=_REDDIT_HEADERS, timeout=8)
            if r.status_code != 200:
                continue
            for child in r.json()["data"]["children"]:
                p = child["data"]
                posts.append({
                    "source": "reddit",
                    "sub": f"r/{sub}",
                    "title": p.get("title", ""),
                    "body": p.get("selftext", "")[:250],
                    "score": p.get("score", 0),
                    "url": f"https://reddit.com{p.get('permalink', '')}",
                    "ts": datetime.fromtimestamp(
                        p.get("created_utc", time.time())
                    ).isoformat(),
                })
        except Exception as e:
            print(f"[sentiment] Reddit [{sub}] error: {e}")
    return posts


# ──────────────────────────────────────────────
# StockTwits (public API, no auth for trending)
# ──────────────────────────────────────────────

def fetch_stocktwits_trending() -> List[Dict]:
    try:
        r = requests.get(
            "https://api.stocktwits.com/api/2/trending/symbols.json",
            timeout=8
        )
        if r.status_code != 200:
            return []
        return [
            {
                "ticker": s.get("symbol", ""),
                "title": s.get("title", ""),
                "watchlist_count": s.get("watchlist_count", 0),
            }
            for s in r.json().get("symbols", [])[:25]
            if s.get("symbol")
        ]
    except Exception as e:
        print(f"[sentiment] StockTwits trending error: {e}")
        return []


def fetch_stocktwits_stream(ticker: str, limit: int = 20) -> List[Dict]:
    try:
        r = requests.get(
            f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json",
            timeout=8
        )
        if r.status_code != 200:
            return []
        msgs = []
        for m in r.json().get("messages", [])[:limit]:
            sentiment_tag = (m.get("entities") or {}).get("sentiment") or {}
            st_label = sentiment_tag.get("basic", "")   # "Bullish" | "Bearish" | ""
            msgs.append({
                "source": "stocktwits",
                "ticker": ticker,
                "text": m.get("body", ""),
                "st_label": st_label,
                "ts": m.get("created_at", ""),
            })
        return msgs
    except Exception as e:
        print(f"[sentiment] StockTwits stream [{ticker}] error: {e}")
        return []


# ──────────────────────────────────────────────
# NewsAPI (optional — requires NEWSAPI_KEY)
# ──────────────────────────────────────────────

def fetch_news(ticker: str = None) -> List[Dict]:
    api_key = os.environ.get("NEWSAPI_KEY", "")
    if not api_key:
        return []
    try:
        from newsapi import NewsApiClient
        client = NewsApiClient(api_key=api_key)
        q = ticker if ticker else "stock market S&P 500 investing"
        articles = client.get_everything(
            q=q, language="en", sort_by="publishedAt", page_size=20,
            from_param=(datetime.utcnow() - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S"),
        )
        return [{
            "source": "news",
            "ticker": ticker,
            "title": a.get("title", ""),
            "text": a.get("description") or a.get("title", ""),
            "url": a.get("url", ""),
            "ts": a.get("publishedAt", ""),
        } for a in articles.get("articles", [])]
    except Exception as e:
        print(f"[sentiment] NewsAPI error: {e}")
        return []


# ──────────────────────────────────────────────
# Twitter/X — requires paid API
# ──────────────────────────────────────────────

def fetch_twitter_recent(query: str = "stocks OR investing", max_results: int = 30) -> List[Dict]:
    """
    Twitter/X Basic API ($100+/month). Set TWITTER_BEARER_TOKEN env var.
    Returns empty list if no token is set.
    """
    token = os.environ.get("TWITTER_BEARER_TOKEN", "")
    if not token:
        return []
    try:
        headers = {"Authorization": f"Bearer {token}"}
        params = {
            "query": f"({query}) lang:en -is:retweet",
            "max_results": max_results,
            "tweet.fields": "created_at,public_metrics",
        }
        r = requests.get(
            "https://api.twitter.com/2/tweets/search/recent",
            headers=headers, params=params, timeout=10
        )
        if r.status_code != 200:
            print(f"[sentiment] Twitter error {r.status_code}: {r.text[:200]}")
            return []
        tweets = r.json().get("data", [])
        return [{
            "source": "twitter",
            "text": t.get("text", ""),
            "ts": t.get("created_at", ""),
        } for t in tweets]
    except Exception as e:
        print(f"[sentiment] Twitter error: {e}")
        return []


# ──────────────────────────────────────────────
# Main aggregator
# ──────────────────────────────────────────────

_cache: Dict = {"data": None, "ts": 0.0}
CACHE_TTL = 60   # seconds — refresh every minute for near-live feel


def get_trending_sentiment(force: bool = False) -> Dict:
    """
    Aggregate Reddit + StockTwits (+ Twitter/News if configured) into
    a ranked list of trending tickers with FinBERT sentiment scores.
    Results are cached for CACHE_TTL seconds.
    """
    global _cache
    now = time.time()
    if not force and _cache["data"] and (now - _cache["ts"]) < CACHE_TTL:
        return _cache["data"]

    # Ensure model is loaded
    if _model_status == "unloaded":
        _load_pipeline()

    from app.screener import DEFAULT_TICKERS
    known = set(DEFAULT_TICKERS)

    # ── Fetch ──────────────────────────────────
    reddit_posts   = fetch_reddit_posts(100)
    st_trending    = fetch_stocktwits_trending()
    twitter_posts  = fetch_twitter_recent()        # empty unless TWITTER_BEARER_TOKEN set
    news_posts     = fetch_news()                  # empty unless NEWSAPI_KEY set

    sources_active = ["Reddit", "StockTwits"]
    if twitter_posts:
        sources_active.append("Twitter/X")
    if news_posts:
        sources_active.append("News")

    # ── Accumulate per-ticker ──────────────────
    bucket: Dict[str, Dict] = defaultdict(lambda: {
        "mentions": 0, "reddit": 0, "stocktwits": 0, "twitter": 0, "news": 0,
        "pos": 0, "neg": 0, "neu": 0, "score_sum": 0.0,
        "feed": [],
    })

    def _add(ticker, label, bull, bear, feed_item):
        d = bucket[ticker]
        d["mentions"] += 1
        d["score_sum"] += bull - bear
        if label == "positive":
            d["pos"] += 1
        elif label == "negative":
            d["neg"] += 1
        else:
            d["neu"] += 1
        if len(d["feed"]) < 5:
            d["feed"].append(feed_item)

    # Reddit
    for post in reddit_posts:
        text = f"{post['title']} {post['body']}"
        tickers = extract_tickers(text, known)
        if not tickers:
            continue
        sent = analyze_sentiment(post["title"])   # title only — faster + cleaner
        for t in tickers:
            bucket[t]["reddit"] += 1
            _add(t, sent["label"], sent["bullish"], sent["bearish"], {
                "text": post["title"][:140],
                "source": "Reddit",
                "sub": post["sub"],
                "sentiment": sent["label"],
                "url": post["url"],
                "ts": post["ts"],
            })

    # StockTwits trending (symbol appears on trending = implicit bullish)
    st_tickers = set()
    for item in st_trending:
        t = item["ticker"]
        st_tickers.add(t)
        bucket[t]["stocktwits"] += 1
        _add(t, "positive", 0.7, 0.0, {
            "text": f"Trending on StockTwits — {item['title']}",
            "source": "StockTwits",
            "sub": "",
            "sentiment": "positive",
            "url": f"https://stocktwits.com/symbol/{t}",
            "ts": datetime.utcnow().isoformat(),
        })

    # StockTwits per-ticker stream for top trending
    for t in list(st_tickers)[:8]:
        for msg in fetch_stocktwits_stream(t, 15):
            st_label = msg["st_label"]
            if st_label == "Bullish":
                label, bull, bear = "positive", 0.8, 0.0
            elif st_label == "Bearish":
                label, bull, bear = "negative", 0.0, 0.8
            else:
                sent = analyze_sentiment(msg["text"])
                label, bull, bear = sent["label"], sent["bullish"], sent["bearish"]
            bucket[t]["stocktwits"] += 1
            _add(t, label, bull, bear, {
                "text": msg["text"][:140],
                "source": "StockTwits",
                "sub": "",
                "sentiment": label,
                "url": f"https://stocktwits.com/symbol/{t}",
                "ts": msg["ts"],
            })

    # Twitter/X
    for post in twitter_posts:
        tickers = extract_tickers(post["text"], known)
        sent = analyze_sentiment(post["text"])
        for t in tickers:
            bucket[t]["twitter"] += 1
            _add(t, sent["label"], sent["bullish"], sent["bearish"], {
                "text": post["text"][:140],
                "source": "Twitter/X",
                "sub": "",
                "sentiment": sent["label"],
                "url": "",
                "ts": post["ts"],
            })

    # News
    for post in news_posts:
        tickers = extract_tickers(post["title"], known)
        sent = analyze_sentiment(post["text"])
        for t in tickers:
            bucket[t]["news"] += 1
            _add(t, sent["label"], sent["bullish"], sent["bearish"], {
                "text": post["title"][:140],
                "source": "News",
                "sub": "",
                "sentiment": sent["label"],
                "url": post.get("url", ""),
                "ts": post["ts"],
            })

    # ── Build ranked results ───────────────────
    results = []
    for ticker, d in bucket.items():
        total = d["pos"] + d["neg"] + d["neu"]
        if total == 0:
            continue
        bull_pct = round(d["pos"] / total * 100, 1)
        bear_pct = round(d["neg"] / total * 100, 1)
        neut_pct = round(d["neu"] / total * 100, 1)
        score    = round(d["score_sum"] / max(d["mentions"], 1), 3)
        overall  = "bullish" if score > 0.08 else "bearish" if score < -0.08 else "neutral"
        results.append({
            "ticker":              ticker,
            "mentions":            d["mentions"],
            "reddit_mentions":     d["reddit"],
            "stocktwits_mentions": d["stocktwits"],
            "twitter_mentions":    d["twitter"],
            "news_mentions":       d["news"],
            "bullish_pct":         bull_pct,
            "bearish_pct":         bear_pct,
            "neutral_pct":         neut_pct,
            "overall":             overall,
            "score":               score,
            "feed":                d["feed"][:3],
        })

    results.sort(key=lambda x: x["mentions"], reverse=True)

    output = {
        "trending":       results[:20],
        "last_updated":   datetime.utcnow().isoformat() + "Z",
        "sources_active": sources_active,
        "posts_analyzed": len(reddit_posts) + len(twitter_posts) + len(news_posts),
        "model":          _model_status,
        "twitter_status": "active" if twitter_posts else "requires_paid_api",
        "news_status":    "active" if news_posts else "set_NEWSAPI_KEY_env_var",
    }

    _cache["data"] = output
    _cache["ts"]   = now
    return output


def get_ticker_sentiment(ticker: str) -> Dict:
    """Deep-dive sentiment for a single ticker (StockTwits stream + news)."""
    if _model_status == "unloaded":
        _load_pipeline()

    ticker = ticker.upper()
    messages = []

    for msg in fetch_stocktwits_stream(ticker, 30):
        st_label = msg["st_label"]
        if st_label == "Bullish":
            label, bull, bear = "positive", 0.8, 0.0
        elif st_label == "Bearish":
            label, bull, bear = "negative", 0.0, 0.8
        else:
            sent = analyze_sentiment(msg["text"])
            label, bull, bear = sent["label"], sent["bullish"], sent["bearish"]
        messages.append({**msg, "sentiment": label, "bull": bull, "bear": bear})

    for post in fetch_news(ticker):
        sent = analyze_sentiment(post["text"])
        messages.append({**post, "sentiment": sent["label"],
                         "bull": sent["bullish"], "bear": sent["bearish"]})

    total = len(messages)
    pos = sum(1 for m in messages if m["sentiment"] == "positive")
    neg = sum(1 for m in messages if m["sentiment"] == "negative")
    neu = total - pos - neg
    score = sum(m["bull"] - m["bear"] for m in messages) / max(total, 1)

    return {
        "ticker":      ticker,
        "total":       total,
        "bullish_pct": round(pos / max(total, 1) * 100, 1),
        "bearish_pct": round(neg / max(total, 1) * 100, 1),
        "neutral_pct": round(neu / max(total, 1) * 100, 1),
        "score":       round(score, 3),
        "overall":     "bullish" if score > 0.08 else "bearish" if score < -0.08 else "neutral",
        "messages":    messages[:20],
    }
