import pandas as pd
from pathlib import Path
from src.data.load_data import load_rr_data, load_actigraph_data, load_user_data

def prepare_user_data(user_id: str, raw_base="../data/raw", out_base="../data/processed"):
    """
    Preprocesses and merges RR and Actigraph data for a given user.
    
    Args:
    	user_id (str): Identifier for the user (e.g. "user_1").
		raw_base (str): Base directory for raw data.
		out_base (str): Base directory for processed data.
	"""									
    user_dir = Path(raw_base) / user_id
    out_path = Path(out_base) / f"{user_id}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Charger les données
    rr_df = load_rr_data(user_dir)
    act_df = load_actigraph_data(user_dir)

    # Convertir RR → bpm
    rr_df["bpm"] = 60 / rr_df["ibi_s"]
    rr_df = rr_df.set_index("timestamp")
    rr_df = rr_df[~rr_df.index.duplicated(keep="first")]
    rr_interp = rr_df["bpm"].resample("1s").mean().interpolate().reset_index()

    # Fusion
    merged = pd.merge(act_df, rr_interp, on="timestamp", how="left")

    # Sauvegarde
    merged.to_csv(out_path, index=False)
    print(f"✅ Données fusionnées sauvegardées : {out_path}")
    
def window_features(df: pd.DataFrame, user_id: str, freq: str = "60s") -> pd.DataFrame:
    """
    Extracts rolling window features from processed user data.

    Args:
        df (pd.DataFrame): processed data with 'timestamp' column.
        freq (str): window duration (e.g. '60s', '30s').

    Returns:
        pd.DataFrame: one row per window with extracted features.
    """
    # Vérifie que toutes les colonnes nécessaires sont là
    expected_cols = ["HR", "Vector Magnitude", "Steps"]
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        print(f"⚠️ Colonnes manquantes pour {user_id} : {missing}")
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    df = df.sort_index()

    features = df.resample(freq).agg({
        "HR": ["mean", "std"],
        "Vector Magnitude": ["mean", "std"],
        "Steps": "sum"
    })

    # Flatten MultiIndex columns
    features.columns = ["_".join(col).lower() for col in features.columns]
    features = features.reset_index()
    features["user_id"] = user_id
    return features

def build_dataset(processed_dir: str = "data/processed", raw_dir: str = "data/raw", out_path: str = "data/features/all_users.csv") -> None:
    """
    Builds the full dataset by extracting features per user and adding the sleep label.

    Args:
        processed_dir (str): path to the processed user data
        raw_dir (str): path to the raw user folders (to load sleep.csv)
        out_path (str): path to save the final dataset
    """
    processed_dir = Path(processed_dir)
    raw_dir = Path(raw_dir)
    all_users = []

    for user_path in raw_dir.glob("user_*"):
        user_id = user_path.stem
        df = load_user_data(raw_dir / user_id)
        print(f"{user_id} → {df.shape if df is not None else 'None'}")
        print(f"{user_id} columns: {df.columns}" if df is not None else f"{user_id} columns: None")

        if df is None or df.empty:
            print(f"❌ Données manquantes ou vides pour {user_id}")
            continue

        print(f"🔄 Traitement des données pour {user_id}...")

        feats = window_features(df, user_id=user_id)
        print("✅ Features extracted: {feats.shape}")

        # Save per-user features
        processed_dir.mkdir(parents=True, exist_ok=True)
        # feats.to_csv(processed_dir / f"{user_id}.csv", index=False)

        # Load sleep.csv and build sleep intervals
        sleep_path = raw_dir / user_id / "sleep.csv"
        if sleep_path.exists():
            sleep_df = pd.read_csv(sleep_path)
            
            sleep_df["In Bed Date"] = sleep_df["In Bed Date"].astype(str)
            sleep_df["In Bed Time"] = sleep_df["In Bed Time"].astype(str)
            sleep_df["Out Bed Date"] = sleep_df["Out Bed Date"].astype(str)
            sleep_df["Out Bed Time"] = sleep_df["Out Bed Time"].astype(str)
            
            sleep_df["start"] = pd.to_datetime("2023-01-0" + sleep_df["In Bed Date"].astype(str) + " " + sleep_df["In Bed Time"])
            sleep_df["end"] = pd.to_datetime("2023-01-0" + sleep_df["Out Bed Date"].astype(str) + " " + sleep_df["Out Bed Time"])

            # Label each window
            feats["is_sleeping"] = feats["timestamp"].apply(
                lambda ts: any((ts >= start) & (ts <= end) for start, end in zip(sleep_df["start"], sleep_df["end"]))
            )
        else:
            print(f"⚠️ sleep.csv manquant pour {user_id}")
            feats["is_sleeping"] = False

        all_users.append(feats)

    # Concat and save
    full_df = pd.concat(all_users, ignore_index=True)
    # out_path = Path(out_path)
    # out_path.parent.mkdir(parents=True, exist_ok=True)
    # full_df.to_csv(out_path, index=False)
    # print(f"✅ Dataset complet sauvegardé : {out_path}")
    return full_df
