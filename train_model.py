"""Train a small churn model and save it as model.joblib.

Run once before the Streamlit app:
    python train_model.py

The data is synthetic so the demo works without a dataset.
Swap in your own data and keep the same column names in app.py.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

rng = np.random.default_rng(42)
n = 1000

df = pd.DataFrame({
    "tenure": rng.integers(0, 73, n),
    "MonthlyCharges": rng.uniform(20, 150, n).round(2),
    "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
})

# Short tenure, high charges and month-to-month contracts churn more often.
risk = (
    -0.08 * df["tenure"]
    + 0.02 * df["MonthlyCharges"]
    + df["Contract"].map({"Month-to-month": 1.5, "One year": 0.0, "Two year": -1.5})
)
df["churn"] = (risk + rng.normal(0, 1, n) > 0).astype(int)

X = df[["tenure", "MonthlyCharges", "Contract"]]
y = df["churn"]

preprocess = ColumnTransformer(
    transformers=[("contract", OneHotEncoder(handle_unknown="ignore"), ["Contract"])],
    remainder="passthrough",
)

model = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=1000)),
])

model.fit(X, y)
print("Training accuracy:", round(model.score(X, y), 3))

joblib.dump(model, "model.joblib")
print("Saved model.joblib")
