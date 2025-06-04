from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import pandas as pd

def train_model(df, eval_during_training=False):
    X = df.drop(columns=["timestamp", "is_sleeping", "user_id"])
    y = df["is_sleeping"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    model = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    if eval_during_training:
        y_pred = model.predict(X_test)
        print("\n📊 Training evaluation metrics:")
        print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(model, "models/random_forest_model.joblib")
    print("✅ Model saved to models/random_forest_model.joblib")