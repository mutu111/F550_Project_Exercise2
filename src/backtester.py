#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:51:00 2026

@author: fabriziocoiai
"""
# backtester.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
import numpy as np


@dataclass
class BacktestConfig:
    initial_cash: float = 100_000.0
    transaction_cost_bps: float = 10.0
    trade_size_units: float = 5.0
    allow_short: bool = False


def compute_sharpe(returns):
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return 0.0
    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(returns.mean() / std)


def compute_max_drawdown(pv):
    cum_max = pv.cummax()
    drawdown = (pv - cum_max) / cum_max
    return drawdown.min()


def compute_volatility(returns):
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return 0.0
    std = returns.std()
    if np.isnan(std):
        return 0.0
    return float(std)


def run_signal_backtest(
    prices,
    signals_by_date,
    holding_period_days: int = 20,
) -> Dict[str, float]:
    if isinstance(prices, pd.DataFrame):
        if "Close" in prices.columns:
            prices = prices["Close"]
        elif prices.shape[1] == 1:
            prices = prices.iloc[:, 0]
        else:
            raise ValueError(f"prices DataFrame must have Close or be single-column. Got {prices.columns}")

    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().astype(float)
    prices = prices[~prices.index.duplicated(keep="first")]

    if prices.empty:
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "volatility": 0.0,
        }

    ordered_events = []
    for dt, action in sorted(signals_by_date.items()):
        ts = pd.Timestamp(dt)
        if ts not in prices.index:
            continue
        normalized = str(action).strip().lower()
        if normalized not in {"buy", "hold", "sell"}:
            raise ValueError(f"Unsupported signal action: {action}")
        ordered_events.append((ts, normalized))

    if not ordered_events:
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "volatility": 0.0,
        }

    strategy_returns = []

    for i, (start_date, action) in enumerate(ordered_events):
        start_loc = prices.index.get_loc(start_date)
        horizon_loc = min(start_loc + holding_period_days, len(prices.index) - 1)

        if i + 1 < len(ordered_events):
            next_event_date = ordered_events[i + 1][0]
            next_event_loc = prices.index.get_loc(next_event_date)
            end_loc = min(horizon_loc, next_event_loc)
        else:
            end_loc = horizon_loc

        if end_loc <= start_loc:
            continue

        start_price = float(prices.iloc[start_loc])
        end_price = float(prices.iloc[end_loc])
        raw_return = (end_price / start_price) - 1.0

        if action == "buy":
            strategy_return = raw_return
        elif action == "sell":
            strategy_return = -raw_return
        else:
            strategy_return = 0.0

        strategy_returns.append(strategy_return)

    returns = np.asarray(strategy_returns, dtype=float)
    if returns.size == 0:
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "volatility": 0.0,
        }

    total_return = float(returns.mean())
    sharpe = compute_sharpe(returns)
    volatility = compute_volatility(returns)

    if not np.isfinite(total_return):
        total_return = 0.0
    if not np.isfinite(sharpe):
        sharpe = 0.0
    if not np.isfinite(volatility):
        volatility = 0.0

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "volatility": volatility,
    }


class EventBacktester:
    """
    Event-based signal evaluator that measures each decision over a
    fixed holding window or until the next event date.
    """
     
    def __init__(self, prices: pd.Series, cfg: Optional[BacktestConfig] = None):
        # Convert DataFrame to Series defensively
        if isinstance(prices, pd.DataFrame):
            if "Close" in prices.columns:
                prices = prices["Close"]
            elif prices.shape[1] == 1:
                prices = prices.iloc[:, 0]
            else:
                raise ValueError(f"prices DataFrame must have Close or be single-column. Got {prices.columns}")

        prices.index = pd.to_datetime(prices.index)
        self.prices = prices.sort_index().astype(float)
        self.cfg = cfg or BacktestConfig()  
        
    def run(
        self,
        *,
        ticker: str,
        agent,
        valuation_inputs_by_event_date: Dict[pd.Timestamp, object],
        holding_period_days: int = 20,
    ) -> Dict[str, float]:
        signals_by_date: Dict[pd.Timestamp, str] = {}

        for dt in self.prices.index:
            dt = pd.Timestamp(dt)
            vin = valuation_inputs_by_event_date.get(dt)
            if vin is not None:
                decision = agent.decide(vin)
                signals_by_date[dt] = decision.action

        return run_signal_backtest(
            self.prices,
            signals_by_date,
            holding_period_days=holding_period_days,
        )
        
        
