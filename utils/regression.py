
import numpy as np
from sklearn.linear_model import LinearRegression

def recommend_budget(income, budget_data):
    if income <= 0:
        return 0.0
    X = budget_data[["income"]].values
    y = budget_data["recommended_budget"].values
    model = LinearRegression()
    model.fit(X, y)
    return float(max(0, model.predict(np.array([[income]]))[0]))
