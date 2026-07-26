import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    SCRIPT_DIR / "Synthetic_Financial_Datasets_For_Fraud_Detection_resample.csv",
    SCRIPT_DIR / "Cleaned Data.csv",
    SCRIPT_DIR / "cleaned_fraud_data.csv",
]
MODEL_PATH = SCRIPT_DIR / "gradient_boost_model_from_notebook.pkl"


def find_data_file():
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("No compatible dataset file was found in the BankFraudAI folder.")


def prepare_features(df):
    print("Preparing features from the notebook workflow...")

    df = df.copy()

    df["errorBalanceOrig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    df["isMerchantDest"] = df["nameDest"].astype(str).str.startswith("M").astype(int)

    df["origDrainRatio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["isOrigDrained"] = ((df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)).astype(int)

    log_cols = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    for col in log_cols:
        df[f"{col}_log"] = np.log1p(df[col])

    if "type" in df.columns:
        df = pd.get_dummies(df, columns=["type"], drop_first=True)

    drop_cols = ["nameOrig", "nameDest", "isFlaggedFraud", "isFraud"] + log_cols
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["isFraud"]

    return X, y


def train_gradient_boosting_from_notebook():
    data_path = find_data_file()
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("Balancing training data with SMOTE...")
    smote = SMOTE(sampling_strategy="auto", random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print("Training HistGradientBoostingClassifier...")
    gb_clf = HistGradientBoostingClassifier(
        max_iter=50,
        max_depth=5,
        random_state=42,
    )
    gb_clf.fit(X_train_bal, y_train_bal)

    y_pred = gb_clf.predict(X_test)
    y_proba = gb_clf.predict_proba(X_test)[:, 1]

    print("\nGradient Boosting metrics:")
    print("Precision :", round(precision_score(y_test, y_pred, zero_division=0), 4))
    print("Recall    :", round(recall_score(y_test, y_pred, zero_division=0), 4))
    print("F1-Score  :", round(f1_score(y_test, y_pred, zero_division=0), 4))
    print("ROC-AUC   :", round(roc_auc_score(y_test, y_proba), 4))

    print("\nConfusion Matrix:")
    print(pd.DataFrame(confusion_matrix(y_test, y_pred), index=["Actual Legit", "Actual Fraud"], columns=["Pred Legit", "Pred Fraud"]))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraudulent"]))

    with MODEL_PATH.open("wb") as fh:
        pickle.dump(gb_clf, fh)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_gradient_boosting_from_notebook()
