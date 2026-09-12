
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def forecast_expenses(forecast_data, months):
    df = forecast_data.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    series = df.set_index("date")["expense"].astype(float)
    model = ExponentialSmoothing(
        series, trend="add", seasonal=None,
        initialization_method="estimated"
    ).fit()
    future = model.forecast(int(months))
    future.index = pd.date_range(
        series.index.max() + pd.offsets.MonthBegin(1),
        periods=int(months), freq="MS"
    )
    return future
