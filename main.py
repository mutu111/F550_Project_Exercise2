#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:51:39 2026

@author: fabriziocoiai
"""
# main.py
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
from pypdf import PdfReader

from src.env_utils import load_dotenv
from src.llm_backend import OpenAIBackend
from src.market_data import get_price_series
from src.valuation_agent import ValuationAgent, ValuationInputs, ValuationAgentConfig
from src.backtester import (
    EventBacktester,
    BacktestConfig,
)
from src.filing_rag import FilingRAG


BASE_DIR = Path(__file__).resolve().parent


def extract_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages)


def build_nvda_quarter_table() -> pd.DataFrame:
    rows = [
        {
            "end": "2023-01-29",
            "filed": "2023-02-22",
            "revenue": 6.051e9,
            "op_income": 0.601e9,
            "net_income": 1.414e9,
            "fcf": 1.738e9,
            "shares_outstanding": 24.70e9,
        },
        {
            "end": "2023-04-30",
            "filed": "2023-05-24",
            "revenue": 7.192e9,
            "op_income": 2.140e9,
            "net_income": 2.043e9,
            "fcf": 2.627e9,
            "shares_outstanding": 24.71e9,
        },
        {
            "end": "2023-07-30",
            "filed": "2023-08-23",
            "revenue": 13.507e9,
            "op_income": 6.800e9,
            "net_income": 6.188e9,
            "fcf": 6.055e9,
            "shares_outstanding": 24.68e9,
        },
        {
            "end": "2023-10-29",
            "filed": "2023-11-21",
            "revenue": 18.120e9,
            "op_income": 10.417e9,
            "net_income": 9.243e9,
            "fcf": 7.279e9,
            "shares_outstanding": 24.67e9,
        },
        {
            "end": "2024-01-28",
            "filed": "2024-02-21",
            "revenue": 22.103e9,
            "op_income": 13.615e9,
            "net_income": 12.285e9,
            "fcf": 11.242e9,
            "shares_outstanding": 24.62e9,
        },
        {
            "end": "2024-04-28",
            "filed": "2024-05-22",
            "revenue": 26.044e9,
            "op_income": 16.909e9,
            "net_income": 14.881e9,
            "fcf": 14.924e9,
            "shares_outstanding": 24.65e9,
        },
        {
            "end": "2024-07-28",
            "filed": "2024-08-28",
            "revenue": 30.040e9,
            "op_income": 18.642e9,
            "net_income": 16.599e9,
            "fcf": 14.300e9,
            "shares_outstanding": 24.68e9,
        },
    ]
    df = pd.DataFrame(rows)
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    return df


def align_to_trading_day(prices: pd.Series, dt: pd.Timestamp) -> pd.Timestamp:
    dt = pd.Timestamp(dt)
    idx = prices.index
    if dt in idx:
        return dt
    later = idx[idx >= dt]
    if len(later) == 0:
        return pd.Timestamp(idx[-1])
    return pd.Timestamp(later[0])


def main():
    load_dotenv()

    ticker = "NVDA"
    start = "2024-02-01"
    end = "2024-10-15"
    price_provider = "csv"
    price_csv_path = BASE_DIR / "data" / "NVDA_2024.csv"
    evaluation_dates = [
        pd.Timestamp("2024-02-05"),
        pd.Timestamp("2024-08-05"),
    ]
    sentiment_map = {
        "2024-02": 0.258,
        "2024-08": 0.445,
    }
    filing_pdf_map = {
        "2024-02": {
            "path": BASE_DIR / "SEC_Filings" / "nvda_202402.pdf",
            "doc_id": "nvda_202402",
            "filed": pd.Timestamp("2024-02-01"),
        },
        "2024-08": {
            "path": BASE_DIR / "SEC_Filings" / "nvda_202408.pdf",
            "doc_id": "nvda_202408",
            "filed": pd.Timestamp("2024-08-01"),
        },
    }

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY first.")

    prices = get_price_series(
        ticker,
        start=start,
        end=end,
        provider=price_provider,
        csv_path=price_csv_path,
    )
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().astype(float)
    prices = prices[~prices.index.duplicated(keep="first")]

    llm = OpenAIBackend(
        chat_model="gpt-5.2",
        embed_model="text-embedding-3-small",
        temperature=0.2,
        max_output_tokens=800,
    )

    quarter_table = build_nvda_quarter_table()

    filing_rag = FilingRAG()

    print("Adding local SEC PDFs to RAG...")
    for meta in filing_pdf_map.values():
        filing_rag.add_document(
            llm,
            doc_id=meta["doc_id"],
            ticker=ticker,
            filed=meta["filed"],
            source="10-Q PDF",
            text=extract_pdf_text(meta["path"]),
        )
    print("RAG documents added.")

    agent_with_sentiment = ValuationAgent(
        llm=llm,
        config=ValuationAgentConfig(
            use_llm=True,
            use_sentiment=True,
            cheap_threshold_ps=3.0,
            expensive_threshold_ps=8.0,
            cheap_threshold_pe=15.0,
            expensive_threshold_pe=30.0,
            cheap_threshold_pfcf=12.0,
            expensive_threshold_pfcf=25.0,
        ),
        filing_rag=filing_rag,
    )
    agent_without_sentiment = ValuationAgent(
        llm=llm,
        config=ValuationAgentConfig(
            use_llm=True,
            use_sentiment=False,
            cheap_threshold_ps=3.0,
            expensive_threshold_ps=8.0,
            cheap_threshold_pe=15.0,
            expensive_threshold_pe=30.0,
            cheap_threshold_pfcf=12.0,
            expensive_threshold_pfcf=25.0,
        ),
        filing_rag=filing_rag,
    )

    valuation_inputs_by_event_date = {}
    decisions_log = []
    for fd in evaluation_dates:
        fd = pd.Timestamp(fd)
        if fd < prices.index.min() or fd > prices.index.max():
            continue

        trade_dt = align_to_trading_day(prices, fd)
        month_key = trade_dt.strftime("%Y-%m")

        if month_key.startswith("2024-02"):
            sentiment_score = sentiment_map["2024-02"]
            filing_doc_id = filing_pdf_map["2024-02"]["doc_id"]
        elif month_key.startswith("2024-08"):
            sentiment_score = sentiment_map["2024-08"]
            filing_doc_id = filing_pdf_map["2024-08"]["doc_id"]
        else:
            sentiment_score = None
            filing_doc_id = None

        window = prices.loc[:trade_dt].tail(5)
        recent_prices = window.tolist()

        val = prices.loc[trade_dt]
        price = float(val.iloc[0]) if hasattr(val, "iloc") else float(val)

        vin = ValuationInputs(
            asof=trade_dt,
            ticker=ticker,
            price=price,
            market_cap=None,
            shares_outstanding=None,
            quarter_table=quarter_table,
            sentiment_score=sentiment_score,
            recent_prices=recent_prices,
            filing_doc_id=filing_doc_id,
        )
        valuation_inputs_by_event_date[pd.Timestamp(trade_dt)] = vin

        decision = agent_with_sentiment.decide(vin)
        decisions_log.append({
            "date": trade_dt,
            "type": "with_sentiment",
            "action": decision.action,
            "confidence": decision.confidence,
            "thesis": decision.thesis,
        })

        decision2 = agent_without_sentiment.decide(vin)
        decisions_log.append({
            "date": trade_dt,
            "type": "without_sentiment",
            "action": decision2.action,
            "confidence": decision2.confidence,
            "thesis": decision2.thesis,
        })

    print("\n=== AGENT DECISIONS ===")
    for d in decisions_log:
        print(f"\nDate: {d['date'].date()} | {d['type']}")
        print("Action:", d["action"])
        print("Confidence:", round(d["confidence"], 3))
        print("Thesis:", d["thesis"])

    bt = EventBacktester(
        prices=prices,
        cfg=BacktestConfig(
            initial_cash=100_000.0,
            transaction_cost_bps=10.0,
            trade_size_units=5.0,
            allow_short=True,
        ),
    )

    results_with = bt.run(
        ticker=ticker,
        agent=agent_with_sentiment,
        valuation_inputs_by_event_date=valuation_inputs_by_event_date,
    )

    results_without = bt.run(
        ticker=ticker,
        agent=agent_without_sentiment,
        valuation_inputs_by_event_date=valuation_inputs_by_event_date,
    )

    backtest_results = {
        "with_sentiment": results_with,
        "without_sentiment": results_without,
    }

    print("\n=== BACKTEST RESULTS ===")
    print(backtest_results)


if __name__ == "__main__":
    main()
