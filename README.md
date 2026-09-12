# Personal Finance Assistant – Fixed & Runnable

This corrected version preserves the original project's three ML features:

1. Predictive Expense Categorization — Logistic Regression + TF-IDF
2. Budget Recommendations — Linear Regression
3. Expense Forecasting — Exponential Smoothing

## Project structure

```text
Personal_Finance_Assistant_AI_Agent_Fixed/
├── main.py
├── requirements.txt
├── run_windows.bat
├── run_linux_mac.sh
├── data/
│   ├── expense_data.csv
│   ├── budget_data.csv
│   └── expense_forecast_data.csv
└── utils/
    ├── __init__.py
    ├── data_processing.py
    ├── classification.py
    ├── regression.py
    └── forecasting.py
```

## Run

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install:
```bash
pip install -r requirements.txt
```

Start:
```bash
streamlit run main.py
```

Open `http://localhost:8501`.

## Sample real-time tests

- `Uber ride` → Transport
- `Grocery shopping` → Groceries
- `Netflix subscription` → Entertainment
- Monthly income `50000` → ML-based recommended budget
- Forecast `3` months → future expense estimates and chart

The supplied ZIP only contained `main.py`, `README.md`, `requirements.txt`, and `LICENSE`, while `main.py` referenced missing `utils/` and `data/` files. This version supplies those missing runtime components and sample data so the application is self-contained.
