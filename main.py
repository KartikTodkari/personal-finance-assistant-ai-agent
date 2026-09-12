
import streamlit as st
import pandas as pd
from utils.data_processing import load_data
from utils.classification import predict_expense_category
from utils.regression import recommend_budget
from utils.forecasting import forecast_expenses

st.set_page_config(page_title="Personal Finance Assistant", page_icon="💰", layout="wide")

st.markdown("""
<style>
.main {background:#f7f9fc;}
.hero {padding:18px 22px;border-radius:14px;background:linear-gradient(135deg,#1557a6,#2878d8);color:white;margin-bottom:18px;}
.card {padding:18px;border-radius:14px;background:white;border:1px solid #e4eaf2;box-shadow:0 2px 10px rgba(0,0,0,.04);}
</style>
""", unsafe_allow_html=True)

expense_data = load_data("data/expense_data.csv")
budget_data = load_data("data/budget_data.csv")
forecast_data = load_data("data/expense_forecast_data.csv")

st.markdown('<div class="hero"><h1>💰 Personal Finance Assistant</h1><p>AI-powered expense categorization, budget recommendations and expense forecasting.</p></div>', unsafe_allow_html=True)

st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Choose a feature",
    ["Expense Categorization", "Budget Recommendations", "Expense Forecasting"]
)

if option == "Expense Categorization":
    st.header("🔎 Predictive Expense Categorization")
    st.write("Enter a transaction description and the Logistic Regression model predicts its category.")
    description = st.text_input("Expense description", placeholder="e.g. Uber ride to office")
    if st.button("Classify Expense", type="primary"):
        category = predict_expense_category(description)
        st.success(f"Predicted Category: {category}")
    st.caption("Example: Uber ride → Transport | Grocery shopping → Groceries | Netflix subscription → Entertainment")

elif option == "Budget Recommendations":
    st.header("📊 Budget Recommendation")
    income = st.number_input("Monthly income", min_value=0.0, value=50000.0, step=1000.0)
    if st.button("Get Budget Recommendation", type="primary"):
        recommendation = recommend_budget(income, budget_data)
        savings = max(0, income - recommendation)
        c1, c2 = st.columns(2)
        c1.metric("Recommended monthly budget", f"₹{recommendation:,.2f}")
        c2.metric("Estimated amount left", f"₹{savings:,.2f}")
        st.info("This is a machine-learning estimate based on the sample budget dataset; it is not financial advice.")

else:
    st.header("📈 Expense Forecasting")
    months = st.slider("Months to forecast", 1, 12, 3)
    if st.button("Forecast Expenses", type="primary"):
        result = forecast_expenses(forecast_data, months)
        display = result.rename("Forecasted Expense").to_frame()
        st.line_chart(display)
        st.dataframe(display.style.format({"Forecasted Expense": "₹{:,.2f}"}), use_container_width=True)

st.divider()
st.caption("Educational project. Predictions are estimates and should not be treated as professional financial advice.")
