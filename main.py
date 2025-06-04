import sys
import pandas as pd
from src.models.train import train_model

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|eval]")
        return

    command = sys.argv[1]

    if command == "train":
        print("🔨 Building dataset...")
        import os

        csv_path = "data/features/all_users.csv"
        if not os.path.exists(csv_path):
            print("❌ Dataset not found. Please run 'make prepare' first.")
            return

        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df.dropna(inplace=True)

        print("🏋️ Training model...")
        train_model(df)

    elif command == "eval":
        import os
        from src.models.evaluate_models import evaluate_model

        csv_path = "data/features/all_users.csv"
        model_path = "models/random_forest_model.joblib"

        if not os.path.exists(csv_path):
            print("❌ Dataset not found. Please run 'make prepare' first.")
            return

        if not os.path.exists(model_path):
            print("❌ Model not found. Please run 'make train' first.")
            return

        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df.dropna(inplace=True)

        print("📈 Evaluating model...")
        evaluate_model()
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()