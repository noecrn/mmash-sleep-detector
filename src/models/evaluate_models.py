import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogReg": LogisticRegression(max_iter=1000),
        "SVM": SVC(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }

    metrics = {
        "Accuracy": accuracy_score,
        "Precision": lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score,
        "F1-score": f1_score,
    }

    rows = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        row = {"Model": name}
        for metric_name, metric_fn in metrics.items():
            row[metric_name] = metric_fn(y_test, y_pred)
        rows.append(row)

    return pd.DataFrame(rows)