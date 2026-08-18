"""
Training script for the UCI Breast Cancer Wisconsin (Diagnostic) dataset.

The dataset is UCI dataset ID 17. The ucimlrepo package can be used to
retrieve it directly from UCI when running this script in an environment
with internet access.

For this submission package, the saved model files are already included.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

DATASET_SOURCE = "UCI Machine Learning Repository, dataset ID 17"

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series((data.target == 0).astype(int), name="diagnosis")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42))
    ]),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(
        n_estimators=200, random_state=42
    )
}

Path("model").mkdir(exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, Path("model") / f"{name}.joblib")

print("All five models trained and saved.")
