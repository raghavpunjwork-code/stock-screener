import yfinance as yf
import pandas as pd
import numpy as np
from app.indicators import compute_moving_averages


def backtest_ma_crossover(ticker: str, period: str = "5y") -> dict:
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty or len(hist) < 200:
        raise ValueError(f"Not enough data for {ticker}")
    close = hist["Close"]
    ma50, ma200 = compute_moving_averages(close)
    signals = pd.DataFrame({"price": close, "ma50": ma50, "ma200": ma200, "signal": 0}, index=hist.index)
    for i in range(1, len(signals)):
        if signals["ma50"].iloc[i] > signals["ma200"].iloc[i] and signals["ma50"].iloc[i-1] <= signals["ma200"].iloc[i-1]:
            signals.iloc[i, signals.columns.get_loc("signal")] = 1
        elif signals["ma50"].iloc[i] < signals["ma200"].iloc[i] and signals["ma50"].iloc[i-1] >= signals["ma200"].iloc[i-1]:
            signals.iloc[i, signals.columns.get_loc("signal")] = -1
    position, entry_price, cash, shares, trades, portfolio = 0, 0, 10000.0, 0, [], []
    for _, row in signals.iterrows():
        if row["signal"] == 1 and position == 0:
            shares = cash / row["price"]; entry_price = row["price"]; cash = 0; position = 1
        elif row["signal"] == -1 and position == 1:
            cash = shares * row["price"]
            trades.append({"pnl_pct": (row["price"] - entry_price) / entry_price * 100})
            shares = 0; position = 0
        portfolio.append(cash + shares * row["price"])
    if position == 1:
        final = float(close.iloc[-1]); cash = shares * final
        trades.append({"pnl_pct": (final - entry_price) / entry_price * 100})
    ps = pd.Series(portfolio, index=signals.index)
    total_return = (ps.iloc[-1] - 10000) / 10000 * 100
    peak = ps.cummax(); drawdown = float(((ps - peak) / peak * 100).min())
    daily_ret = ps.pct_change().dropna()
    excess = daily_ret - 0.05 / 252
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() != 0 else 0
    win_rate = sum(1 for t in trades if t["pnl_pct"] > 0) / len(trades) * 100 if trades else 0
    return {"ticker": ticker, "strategy": "MA Crossover (50/200)", "total_return_pct": round(total_return, 2),
            "max_drawdown_pct": round(drawdown, 2), "sharpe_ratio": round(sharpe, 3),
            "num_trades": len(trades), "win_rate_pct": round(win_rate, 2)}