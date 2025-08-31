README.md (front page) – copy/paste this skeleton                                                  # Inter-Uni Datathon 2025 – Hotham vs Perisher vs Buller

## Problem
Forecast weekly visitors & temperature, compare resorts on cost/crowds/features, and recommend the “luxury boys trip” plan.

## Data
- Climate & visitation: <where it comes from / file names / links>
- Any manual inputs (prices, helicopter, lessons): source list.

## Method
- ARIMA(1,1,1)(1,1,1)[52] for weekly seasonality.
- Train window: YYYY-MM to YYYY-MM; forecast horizon: 2026.
- Evaluation: (MAPE / RMSE) on last season (if computed).

## How to run
```bash
python -m venv .venv && source .venv/bin/activate   # win: .venv\Scripts\activate
pip install -r requirements.txt
python main.py --data data/ --out outputs/
