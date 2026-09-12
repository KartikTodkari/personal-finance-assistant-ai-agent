
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

_model = None
_vectorizer = None

def _load_model():
    global _model, _vectorizer
    if _model is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "expense_data.csv")
        df = pd.read_csv(path)
        _vectorizer = TfidfVectorizer(ngram_range=(1,2), lowercase=True)
        X = _vectorizer.fit_transform(df["description"].astype(str))
        _model = LogisticRegression(max_iter=1000)
        _model.fit(X, df["category"])
    return _model, _vectorizer

def predict_expense_category(description):
    if not description or not description.strip():
        return "Please enter an expense description."
    model, vectorizer = _load_model()
    X = vectorizer.transform([description.strip()])
    return model.predict(X)[0]
