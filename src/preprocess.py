"""
preprocess.py — SONA (Security of Network Access)
==================================================
Data loading and feature engineering pipeline for SONA.
Run this after downloading NSL-KDD dataset files into the data/ folder.

Download from: https://www.unb.ca/cic/datasets/nsl.html
  - KDDTrain+.txt
  - KDDTest+.txt
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# ── Column names from NSL-KDD documentation ──────────────────────────────────

COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty_level"
]

# NSL-KDD attack families mapped to 5 broad categories
ATTACK_MAP = {
    "normal": "Normal",
    # DoS attacks
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "mailbomb": "DoS",
    # Probe attacks
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L (Remote to Local)
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "worm": "R2L",
    # U2R (User to Root)
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "httptunnel": "U2R", "ps": "U2R",
    "sqlattack": "U2R", "xterm": "U2R",
}

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load NSL-KDD train and test files."""
    print("Loading data...")
    train = pd.read_csv(train_path, header=None, names=COLUMNS)
    test  = pd.read_csv(test_path,  header=None, names=COLUMNS)
    print(f"  Train: {train.shape[0]:,} rows | Test: {test.shape[0]:,} rows")
    return train, test


def map_labels(df: pd.DataFrame, binary: bool = False) -> pd.DataFrame:
    """
    Map raw attack labels to categories.
    binary=True  → 'Normal' vs 'Attack' (2 classes)
    binary=False → 'Normal','DoS','Probe','R2L','U2R' (5 classes)
    """
    df = df.copy()
    df["attack_category"] = df["label"].str.lower().map(ATTACK_MAP)
    df["attack_category"] = df["attack_category"].fillna("Other")

    if binary:
        df["target"] = (df["attack_category"] != "Normal").astype(int)
    else:
        df["target"] = df["attack_category"]

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that improve model performance."""
    df = df.copy()

    # Ratio features — capture relative traffic patterns
    df["bytes_ratio"]     = df["src_bytes"] / (df["dst_bytes"] + 1)
    df["error_rate_diff"] = df["serror_rate"] - df["rerror_rate"]

    # Binary: was there any data sent both ways?
    df["bidirectional"]   = ((df["src_bytes"] > 0) & (df["dst_bytes"] > 0)).astype(int)

    # High connection count (potential scan/flood)
    df["high_count"]      = (df["count"] > 100).astype(int)

    # Connection with no data (potential probe)
    df["zero_bytes"]      = ((df["src_bytes"] == 0) & (df["dst_bytes"] == 0)).astype(int)

    return df


def encode_and_scale(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Encode categoricals and scale numerics.
    Returns scaled arrays + dict of fitted encoders (for inference).
    """
    train = train.copy()
    test  = test.copy()

    encoders = {}

    # Label-encode each categorical column
    for col in CATEGORICAL_COLS:
        if col in feature_cols:
            le = LabelEncoder()
            train[col] = le.fit_transform(train[col].astype(str))
            # Handle unseen categories in test gracefully
            test[col]  = test[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0]
                if x in le.classes_ else -1
            )
            encoders[col] = le

    # Standard scale all numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    X_test  = scaler.transform(test[feature_cols])
    encoders["scaler"] = scaler

    return X_train, X_test, encoders


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return all feature columns (drop label/target columns)."""
    drop = {"label", "difficulty_level", "attack_category", "target"}
    return [c for c in df.columns if c not in drop]


def prepare_pipeline(
    train_path: str = "data/KDDTrain+.txt",
    test_path:  str = "data/KDDTest+.txt",
    binary: bool = False,
    save_encoders: bool = True,
) -> dict:
    """
    Full preprocessing pipeline. Returns dict with:
      X_train, X_test, y_train, y_test, feature_cols, encoders, label_encoder
    """
    train_df, test_df = load_data(train_path, test_path)

    # Map labels
    train_df = map_labels(train_df, binary=binary)
    test_df  = map_labels(test_df,  binary=binary)

    # Feature engineering
    train_df = engineer_features(train_df)
    test_df  = engineer_features(test_df)

    feature_cols = get_feature_cols(train_df)
    print(f"  Features: {len(feature_cols)}")

    # Encode targets
    y_train_raw = train_df["target"]
    y_test_raw  = test_df["target"]

    if not binary:
        label_enc = LabelEncoder()
        y_train = label_enc.fit_transform(y_train_raw)
        y_test  = label_enc.transform(y_test_raw)
    else:
        label_enc = None
        y_train = y_train_raw.values
        y_test  = y_test_raw.values

    X_train, X_test, encoders = encode_and_scale(train_df, test_df, feature_cols)

    if save_encoders:
        os.makedirs("models", exist_ok=True)
        joblib.dump(encoders,   "models/encoders.pkl")
        joblib.dump(feature_cols, "models/feature_cols.pkl")
        if label_enc:
            joblib.dump(label_enc, "models/label_encoder.pkl")
        print("  Encoders saved to models/")

    print("\nClass distribution (train):")
    unique, counts = np.unique(y_train, return_counts=True)
    if label_enc:
        labels = label_enc.inverse_transform(unique)
    else:
        labels = ["Normal", "Attack"]
    for lbl, cnt in zip(labels, counts):
        print(f"  {lbl:10s}: {cnt:>7,}  ({cnt/len(y_train)*100:.1f}%)")

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_cols": feature_cols,
        "encoders": encoders,
        "label_encoder": label_enc,
    }


if __name__ == "__main__":
    data = prepare_pipeline(binary=False)
    print("\nPreprocessing complete.")
    print(f"X_train shape: {data['X_train'].shape}")
    print(f"X_test  shape: {data['X_test'].shape}")
