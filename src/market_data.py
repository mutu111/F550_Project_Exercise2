#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:48:50 2026

@author: fabriziocoiai
"""
# market_data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
import pandas as pd
import requests


@dataclass
class MarketSnapshot:
    date: pd.Timestamp
    price: float
    shares_outstanding: Optional[float]
    market_cap: Optional[float]


def _normalize_close_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    if df.empty:
        raise RuntimeError(f"No price data returned for {ticker}.")

    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", ticker) in df.columns:
            series = df[("Close", ticker)]
        elif "Close" in df.columns.get_level_values(0):
            series = df.xs("Close", axis=1, level=0).iloc[:, 0]
        else:
            raise RuntimeError(f"Could not find Close column for {ticker}. Columns={list(df.columns)}")
        return series.dropna().astype(float)

    if "Close" in df.columns:
        series = df["Close"]
    elif df.shape[1] == 1:
        series = df.iloc[:, 0]
    else:
        raise RuntimeError(f"Could not find Close column for {ticker}. Columns={list(df.columns)}")

    return series.dropna().astype(float)


def _stooq_symbol(ticker: str) -> str:
    t = ticker.strip().lower()
    if "." in t:
        return t
    return f"{t}.us"


def _fetch_stooq_prices(ticker: str, start: str, end: str) -> pd.Series:
    symbol = _stooq_symbol(ticker)
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    if not resp.text.strip():
        raise RuntimeError(f"Stooq returned an empty response for {ticker}.")

    try:
        df = pd.read_csv(
            pd.io.common.StringIO(resp.text),
            parse_dates=["Date"],
        )
    except pd.errors.EmptyDataError as e:
        raise RuntimeError(f"Stooq returned no CSV data for {ticker}.") from e
    if df.empty or "Close" not in df.columns:
        raise RuntimeError(f"Stooq returned no usable price data for {ticker}.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df[(df["Date"] >= pd.Timestamp(start)) & (df["Date"] < pd.Timestamp(end))]
    df = df.set_index("Date").sort_index()
    return _normalize_close_series(df, ticker)


def _fetch_csv_prices(
    ticker: str,
    start: str,
    end: str,
    csv_path: Optional[str] = None,
) -> pd.Series:
    path = Path(csv_path) if csv_path else Path("data") / f"{ticker.upper()}.csv"
    if not path.exists():
        raise RuntimeError(f"CSV file not found for {ticker}: {path}")

    df = pd.read_csv(path)
    normalized = {str(c).strip().lower(): c for c in df.columns}

    if "date" not in normalized:
        raise RuntimeError(f"CSV must contain a Date column. Columns={list(df.columns)}")

    close_col = None
    for name in ("close", "close/last", "adj close", "adj_close", "adjusted_close"):
        if name in normalized:
            close_col = normalized[name]
            break
    if close_col is None:
        raise RuntimeError(
            f"CSV must contain Close, Close/Last, or Adj Close column. Columns={list(df.columns)}"
        )

    df = df.rename(columns={normalized["date"]: "Date", close_col: "Close"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = (
        df["Close"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df[(df["Date"] >= pd.Timestamp(start)) & (df["Date"] < pd.Timestamp(end))]
    df = df.set_index("Date").sort_index()
    return _normalize_close_series(df[["Close"]], ticker)


def _fetch_alpha_vantage_prices(
    ticker: str,
    start: str,
    end: str,
    api_key: Optional[str] = None,
) -> pd.Series:
    key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
    if not key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY is not set.")

    resp = requests.get(
        "https://www.alphavantage.co/query",
        params={
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": "full",
            "datatype": "csv",
            "apikey": key,
        },
        timeout=30,
    )
    resp.raise_for_status()
    text = resp.text.strip()
    if not text:
        raise RuntimeError(f"Alpha Vantage returned an empty response for {ticker}.")

    low = text.lower()
    if "thank you for using alpha vantage" in low:
        raise RuntimeError(f"Alpha Vantage limit notice: {text}")
    if "error message" in low or "invalid api call" in low:
        raise RuntimeError(f"Alpha Vantage error for {ticker}: {text}")

    try:
        df = pd.read_csv(pd.io.common.StringIO(text))
    except pd.errors.EmptyDataError as e:
        raise RuntimeError(f"Alpha Vantage returned no CSV data for {ticker}.") from e

    normalized = {str(c).strip().lower(): c for c in df.columns}
    if "timestamp" not in normalized:
        raise RuntimeError(
            f"Alpha Vantage returned an unexpected payload for {ticker}. Columns={list(df.columns)}"
        )
    if "close" not in normalized:
        raise RuntimeError(
            f"Alpha Vantage CSV missing close column for {ticker}. Columns={list(df.columns)}"
        )

    df = df.rename(columns={normalized["timestamp"]: "Date", normalized["close"]: "Close"})
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df[(df["Date"] >= pd.Timestamp(start)) & (df["Date"] < pd.Timestamp(end))]
    df = df.set_index("Date")
    return _normalize_close_series(df[["Close"]], ticker)


def get_price_series(
    ticker: str,
    start: str,
    end: str,
    provider: str = "auto",
    csv_path: Optional[str] = None,
    alpha_vantage_api_key: Optional[str] = None,
) -> pd.Series:
    errors = []

    if provider in {"auto", "csv"}:
        try:
            return _fetch_csv_prices(ticker, start, end, csv_path=csv_path)
        except Exception as e:
            errors.append(f"csv failed: {e}")
            if provider == "csv":
                raise RuntimeError("; ".join(errors)) from e

    if provider in {"auto", "alpha_vantage"}:
        try:
            return _fetch_alpha_vantage_prices(
                ticker,
                start,
                end,
                api_key=alpha_vantage_api_key,
            )
        except Exception as e:
            errors.append(f"alpha_vantage failed: {e}")
            if provider == "alpha_vantage":
                raise RuntimeError("; ".join(errors)) from e

    if provider in {"auto", "stooq"}:
        try:
            return _fetch_stooq_prices(ticker, start, end)
        except Exception as e:
            errors.append(f"stooq failed: {e}")
            if provider == "stooq":
                raise RuntimeError("; ".join(errors)) from e

    if errors:
        raise RuntimeError("; ".join(errors))

    raise ValueError(f"Unsupported provider: {provider}")
