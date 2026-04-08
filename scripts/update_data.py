# scripts/update_data.py
# -*- coding: utf-8 -*-

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf


# =========================================================
# Config
# =========================================================
START_DATE = "2010-01-01"
END_DATE = None  # None = today
DATA_DIR = "data"
CACHE_DIR = "cache"

ALL_TICKERS = [
    "SPY",
    "QQQ",
    "^TWII",
    "0050.TW",
    "^VIX",
    "TLT",
    "DX-Y.NYB",
    "GLD",
    "CL=F",
]

TARGET_MARKETS = [
    ("SPY", "S&P 500 ETF"),
    ("QQQ", "Nasdaq 100 ETF"),
    ("^TWII", "Taiwan Weighted Index"),
    ("0050.TW", "Yuanta Taiwan 50 ETF"),
]

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# =========================================================
# Utilities
# =========================================================
def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def clip_0_100(series: pd.Series) -> pd.Series:
    return series.clip(lower=0, upper=100)


def rolling_percentile(series: pd.Series, window: int = 252) -> pd.Series:
    """
    Convert each point into percentile rank within trailing window, output 0~100.
    """
    def pct_rank(x):
        s = pd.Series(x)
        return s.rank(pct=True).iloc[-1] * 100

    return series.rolling(window, min_periods=max(30, window // 4)).apply(
        pct_rank,
        raw=False
    )


def classification_bucket(x: float) -> str:
    if pd.isna(x):
        return np.nan
    if x < 25:
        return "Calm"
    elif x < 50:
        return "Caution"
    elif x < 75:
        return "Fear"
    return "Extreme Fear"


def opportunity_bucket(x: float) -> str:
    if pd.isna(x):
        return np.nan
    if x < 25:
        return "Low Opportunity"
    elif x < 50:
        return "Neutral"
    elif x < 75:
        return "Watchlist"
    return "High Opportunity"


def make_signal(opportunity_index: float, price: float, ma200: float) -> str:
    """
    Simple decision helper for website display.
    """
    if pd.isna(opportunity_index):
        return "No Data"

    if pd.isna(ma200):
        if opportunity_index >= 75:
            return "Scale In"
        elif opportunity_index >= 50:
            return "Watchlist"
        return "No Action"

    if opportunity_index >= 75 and price > ma200:
        return "Strong Buy Zone"
    elif opportunity_index >= 75 and price <= ma200:
        return "Scale In Carefully"
    elif opportunity_index >= 50:
        return "Watchlist"
    return "No Action"


# =========================================================
# Data download
# =========================================================
def get_prices(
    tickers: List[str],
    start: str = START_DATE,
    end: Optional[str] = END_DATE,
    cache_file: str = os.path.join(CACHE_DIR, "yahoo_prices.csv"),
    force_refresh: bool = True,
) -> pd.DataFrame:
    """
    Download adjusted close prices from yfinance.
    """
    if os.path.exists(cache_file) and not force_refresh:
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return ensure_datetime_index(df)

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    close = {}
    for t in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if (t, "Close") in raw.columns:
                close[t] = raw[(t, "Close")]
            else:
                raise ValueError(f"Cannot find Close for ticker: {t}")
        else:
            if "Close" in raw.columns and len(tickers) == 1:
                close[t] = raw["Close"]
            else:
                raise ValueError(f"Unexpected columns for ticker: {t}")

    df = pd.DataFrame(close)
    df = ensure_datetime_index(df)
    df = df.ffill().dropna(how="all")
    df.to_csv(cache_file, encoding="utf-8-sig")
    return df


# =========================================================
# Feature engineering
# =========================================================
@dataclass
class FearIndexConfig:
    pct_window: int = 252
    trend_ma: int = 125
    drawdown_lb: int = 63
    safe_haven_lb: int = 20
    dxy_lb: int = 20
    gold_lb: int = 20
    oil_lb: int = 20
    oversold_lb: int = 5
    smooth_span: int = 3
    weights: Dict[str, float] = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                "vix": 0.25,
                "trend": 0.20,
                "drawdown": 0.15,
                "safe_haven": 0.15,
                "dxy": 0.10,
                "gold": 0.075,
                "oil": 0.075,
            }


def build_market_fear_index_for_symbol(
    prices: pd.DataFrame,
    target_symbol: str,
    config: FearIndexConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build custom fear index for a given target market symbol.
    Shared external factors: ^VIX, TLT, DX-Y.NYB, GLD, CL=F
    """
    required = [target_symbol, "^VIX", "TLT", "DX-Y.NYB", "GLD", "CL=F"]
    missing = [c for c in required if c not in prices.columns]
    if missing:
        raise ValueError(f"Missing required columns for {target_symbol}: {missing}")

    df = prices.copy().ffill()

    raw = pd.DataFrame(index=df.index)

    # 1) VIX level percentile
    raw["vix_raw"] = df["^VIX"]

    # 2) Trend pressure: below moving average means more fear
    ma = df[target_symbol].rolling(config.trend_ma).mean()
    raw["trend_raw"] = -(df[target_symbol] / ma - 1.0)

    # 3) Drawdown pressure
    roll_max = df[target_symbol].rolling(config.drawdown_lb).max()
    raw["drawdown_raw"] = -(df[target_symbol] / roll_max - 1.0)

    # 4) Safe haven
    raw["safe_haven_raw"] = -(
        df[target_symbol].pct_change(config.safe_haven_lb)
        - df["TLT"].pct_change(config.safe_haven_lb)
    )

    # 5) Dollar strength
    raw["dxy_raw"] = df["DX-Y.NYB"].pct_change(config.dxy_lb)

    # 6) Gold strength
    raw["gold_raw"] = df["GLD"].pct_change(config.gold_lb)

    # 7) Oil weakness
    raw["oil_raw"] = -(df["CL=F"].pct_change(config.oil_lb))

    scores = pd.DataFrame(index=raw.index)
    scores["vix"] = clip_0_100(rolling_percentile(raw["vix_raw"], config.pct_window))
    scores["trend"] = clip_0_100(rolling_percentile(raw["trend_raw"], config.pct_window))
    scores["drawdown"] = clip_0_100(rolling_percentile(raw["drawdown_raw"], config.pct_window))
    scores["safe_haven"] = clip_0_100(rolling_percentile(raw["safe_haven_raw"], config.pct_window))
    scores["dxy"] = clip_0_100(rolling_percentile(raw["dxy_raw"], config.pct_window))
    scores["gold"] = clip_0_100(rolling_percentile(raw["gold_raw"], config.pct_window))
    scores["oil"] = clip_0_100(rolling_percentile(raw["oil_raw"], config.pct_window))

    w = config.weights
    scores["fear_raw"] = (
        w["vix"] * scores["vix"]
        + w["trend"] * scores["trend"]
        + w["drawdown"] * scores["drawdown"]
        + w["safe_haven"] * scores["safe_haven"]
        + w["dxy"] * scores["dxy"]
        + w["gold"] * scores["gold"]
        + w["oil"] * scores["oil"]
    ) / sum(w.values())

    scores["fear_index"] = scores["fear_raw"].ewm(
        span=config.smooth_span,
        adjust=False
    ).mean()
    scores["regime"] = scores["fear_index"].apply(classification_bucket)

    return raw, scores


def build_stress_opportunity_indices_for_symbol(
    prices: pd.DataFrame,
    target_symbol: str,
    config: FearIndexConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw, scores = build_market_fear_index_for_symbol(prices, target_symbol, config)

    df = prices.copy().ffill()

    short_term_oversold_raw = -(df[target_symbol].pct_change(config.oversold_lb))

    opp = pd.DataFrame(index=scores.index)
    opp["stress_component"] = scores["fear_index"]
    opp["drawdown_component"] = scores["drawdown"]
    opp["vix_component"] = scores["vix"]
    opp["oversold_component"] = clip_0_100(
        rolling_percentile(short_term_oversold_raw, config.pct_window)
    )

    opp["opportunity_raw"] = (
        0.35 * opp["stress_component"]
        + 0.30 * opp["drawdown_component"]
        + 0.20 * opp["vix_component"]
        + 0.15 * opp["oversold_component"]
    )

    opp["opportunity_index"] = opp["opportunity_raw"].ewm(
        span=config.smooth_span,
        adjust=False
    ).mean()
    opp["opportunity_regime"] = opp["opportunity_index"].apply(opportunity_bucket)

    merged = scores.join(
        opp[["opportunity_raw", "opportunity_index", "opportunity_regime"]],
        how="left"
    )

    return raw, merged


# =========================================================
# JSON builders
# =========================================================
def build_latest_json(
    prices: pd.DataFrame,
    market_results: Dict[str, Dict[str, pd.DataFrame]],
) -> Dict:
    latest_markets = []

    all_last_dates = []

    for symbol, meta in TARGET_MARKETS:
        merged = market_results[symbol]["merged"].dropna(
            subset=["fear_index", "opportunity_index"]
        )
        if merged.empty:
            continue

        latest_row = merged.iloc[-1]
        latest_date = merged.index[-1]
        all_last_dates.append(latest_date)

        price_series = prices[symbol].dropna()
        if price_series.empty:
            continue

        latest_price = float(price_series.iloc[-1])
        ma200_series = price_series.rolling(200).mean().dropna()
        ma200 = float(ma200_series.iloc[-1]) if not ma200_series.empty else np.nan

        latest_markets.append({
            "symbol": symbol,
            "name": meta,
            "price": round(latest_price, 2),
            "fear_index": round(float(latest_row["fear_index"]), 2),
            "regime": str(latest_row["regime"]),
            "opportunity_index": round(float(latest_row["opportunity_index"]), 2),
            "opportunity_regime": str(latest_row["opportunity_regime"]),
            "signal": make_signal(
                float(latest_row["opportunity_index"]),
                latest_price,
                ma200
            )
        })

    updated_at = None
    if all_last_dates:
        updated_at = str(max(all_last_dates).date())

    return {
        "updated_at": updated_at,
        "markets": latest_markets
    }


def build_history_json(
    prices: pd.DataFrame,
    market_results: Dict[str, Dict[str, pd.DataFrame]],
) -> Dict[str, List[Dict]]:
    history = {}

    for symbol, _ in TARGET_MARKETS:
        merged = market_results[symbol]["merged"]

        tmp = prices[[symbol]].join(
            merged[["fear_index", "opportunity_index"]],
            how="inner"
        ).dropna()

        rows = []
        for idx, row in tmp.iterrows():
            rows.append({
                "date": str(idx.date()),
                "price": round(float(row[symbol]), 2),
                "fear_index": round(float(row["fear_index"]), 2),
                "opportunity_index": round(float(row["opportunity_index"]), 2),
            })

        history[symbol] = rows

    return history


# =========================================================
# Main
# =========================================================
def main(force_refresh: bool = True) -> None:
    print("Downloading market data...")
    prices = get_prices(
        tickers=ALL_TICKERS,
        start=START_DATE,
        end=END_DATE,
        force_refresh=force_refresh,
    )

    print("Downloaded columns:", list(prices.columns))

    config = FearIndexConfig()
    market_results: Dict[str, Dict[str, pd.DataFrame]] = {}

    for symbol, name in TARGET_MARKETS:
        print(f"Processing {symbol} - {name} ...")
        raw, merged = build_stress_opportunity_indices_for_symbol(
            prices=prices,
            target_symbol=symbol,
            config=config,
        )

        market_results[symbol] = {
            "raw": raw,
            "merged": merged,
        }

    print("Building JSON outputs...")
    latest_json = build_latest_json(prices, market_results)
    history_json = build_history_json(prices, market_results)

    latest_path = os.path.join(DATA_DIR, "latest.json")
    history_path = os.path.join(DATA_DIR, "history.json")

    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(latest_json, f, ensure_ascii=False, indent=2)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_json, f, ensure_ascii=False, indent=2)

    print(f"Saved: {latest_path}")
    print(f"Saved: {history_path}")

    print("\nLatest summary:")
    print(json.dumps(latest_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(force_refresh=True)