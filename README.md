# F550 Project

## Exercise 2

This repository contains the code for `F550 Project - Exercise 2`.
The project builds a valuation-focused agent that combines market data, SEC filing retrieval, optional sentiment input, and an event-based backtest to compare signals with and without sentiment.

## Repo Structure

```text
Code Archive-Fund Agent/
|-- data/
|   |-- NVDA_2024.csv
|   |-- data.csv
|   `-- README.txt
|-- SEC_Filings/
|   |-- nvda_202402.pdf
|   `-- nvda_202408.pdf
|-- src/
|   |-- __init__.py
|   |-- backtester.py
|   |-- env_utils.py
|   |-- filing_rag.py
|   |-- llm_backend.py
|   |-- market_data.py
|   |-- sec_fundamentals.py
|   `-- valuation_agent.py
|-- main.py
|-- README.md
|-- requirements.txt
|-- pyproject.toml
|-- environment.yml
`-- .env.example
```

## Setup

Create the conda environment:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate f550-ex2
```

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## API Key Setup

Set the API key in PowerShell before running the project:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

Notes:

- `OPENAI_API_KEY` is required to run the project

## Run

Run the project with:

```bash
python main.py
```
