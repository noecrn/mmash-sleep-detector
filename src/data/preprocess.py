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
    Extracts rolling window features from processed user data, including additional rolling mean, std, and diff features for selected columns.

    Args:
        df (pd.DataFrame): processed data with 'timestamp' column.
        freq (str): window duration (e.g. '60s', '30s').

    Returns:
        pd.DataFrame: one row per window with extracted features and additional rolling/diff features.
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

    # Additional engineered features with corrected column names
    rolling_cols = {
        "hr_mean": "hr_mean_roll3",
        "vector magnitude_mean": "vector_magnitude_mean_roll3",
        "steps_sum": "steps_sum_roll3"
    }

    for col, new_col in rolling_cols.items():
        if col in features.columns:
            features[new_col] = features[col].rolling(window=3, min_periods=1).mean()
        else:
            print(f"⚠️ Column not found for rolling calculation: {col}")

    diff_cols = {
        "hr_mean": "hr_mean_diff",
        "vector magnitude_mean": "vector_magnitude_mean_diff",
        "steps_sum": "steps_sum_diff"
    }

    for col, new_col in diff_cols.items():
        if col in features.columns:
            features[new_col] = features[col].diff().fillna(0)
        else:
            print(f"⚠️ Column not found for diff calculation: {col}")

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
    
    # Add counter for debugging
    user_count = 0
    
    for user_path in raw_dir.glob("user_*"):
        user_count += 1
        user_id = user_path.stem
        print(f"\nProcessing {user_id}...")
        
        df = load_user_data(raw_dir / user_id)
        if df is None or df.empty:
            print(f"❌ Skipping {user_id} - No data")
            continue
            
        feats = window_features(df, user_id=user_id)
        if feats.empty:
            print(f"❌ Skipping {user_id} - No features generated")
            continue

        # Load sleep.csv and add is_sleeping label
        sleep_path = raw_dir / user_id / "sleep.csv"
        if sleep_path.exists():
            sleep_df = pd.read_csv(sleep_path)

            # Convertir en datetime
            sleep_df["start"] = pd.to_datetime("2023-01-0" + sleep_df["In Bed Date"].astype(str) + " " + sleep_df["In Bed Time"])
            sleep_df["end"] = pd.to_datetime("2023-01-0" + sleep_df["Out Bed Date"].astype(str) + " " + sleep_df["Out Bed Time"])

            # Marquer les timestamps comme "sleeping" si dans un intervalle
            feats["is_sleeping"] = feats["timestamp"].apply(
                lambda ts: any((ts >= start) & (ts <= end) for start, end in zip(sleep_df["start"], sleep_df["end"]))
            )
        else:
            print(f"⚠️ sleep.csv manquant pour {user_id}")
            feats["is_sleeping"] = False
            
        all_users.append(feats)
        print(f"✅ Added features for {user_id}")
    
    if user_count == 0:
        raise ValueError(f"No user directories found in {raw_dir}")
        
    if len(all_users) == 0:
        raise ValueError(f"No valid user data processed. Checked {user_count} users")
        
    # Concat and save
    full_df = pd.concat(all_users, ignore_index=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_path, index=False)
    print(f"✅ Full dataset saved to {out_path}")
    return full_df
