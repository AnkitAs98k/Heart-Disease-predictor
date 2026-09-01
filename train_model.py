"""
train_model.py

Reproduces the preprocessing + model comparison from ML1.ipynb, with one
bug fixed: the notebook creates `njdf_heart = pd.get_dummies(df_heart,
columns=['ST_Slope'])` but never assigns it back to `df_heart`, so
ST_Slope is left as a string column and StandardScaler breaks on it if you
run the notebook straight through. Here it's assigned back properly.

Usage:
    1. Put heart.csv (Kaggle "Heart Failure Prediction Dataset") in this folder.
    2. python train_model.py
    3. This writes model.pkl, scaler.pkl and feature_columns.json, which
       app.py loads to serve predictions.
"""

import json

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "heart.csv"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same transformations as the notebook, with the ST_Slope bug fixed."""
    df = df.copy()
    df["Sex"] = df["Sex"].map({"M": 1, "F": 0})
    df["ExerciseAngina"] = df["ExerciseAngina"].map({"Y": 1, "N": 0})
    df = pd.get_dummies(df, columns=["ChestPainType"], drop_first=False)
    df = pd.get_dummies(df, columns=["RestingECG"], drop_first=False)
    df = pd.get_dummies(df, columns=["ST_Slope"], drop_first=False)  # fixed: reassigned

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def main():
    heart = pd.read_csv(DATA_PATH)
    df_heart = build_features(heart)

    X = df_heart.drop("HeartDisease", axis=1)
    y = df_heart["HeartDisease"]
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "SVM": SVC(probability=True),  # probability=True added so we can show a confidence score
        "KNN": KNeighborsClassifier(),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        results.append(
            {
                "model": name,
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "f1_score": round(f1_score(y_test, y_pred), 4),
            }
        )
        models[name] = model  # keep fitted model

    print("\nModel comparison:")
    for r in sorted(results, key=lambda r: -r["f1_score"]):
        print(f"  {r['model']:<22} accuracy={r['accuracy']:<7} f1={r['f1_score']}")

    best_name = max(results, key=lambda r: r["f1_score"])["model"]
    best_model = models[best_name]
    print(f"\nBest model by F1: {best_name} -> saving as model.pkl")

    joblib.dump(best_model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    with open("feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)

    print("Saved model.pkl, scaler.pkl, feature_columns.json")


if __name__ == "__main__":
    main()
