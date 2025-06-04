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
        "Recall": lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0),
        "F1-score": lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0),
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


# Evaluate trained model on all feature data
def evaluate_model():
    from joblib import load
    from sklearn.metrics import classification_report
    import pandas as pd

    model = load("models/random_forest_model.joblib")
    df = pd.read_csv("data/features/all_users.csv")
    X = df.drop(columns=["is_sleeping", "timestamp", "user_id"])
    y = df["is_sleeping"]

    y_pred = model.predict(X)
    print(classification_report(y, y_pred, zero_division=0))