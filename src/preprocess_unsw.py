"""
preprocess_unsw.py — SONA v3: UNSW-NB15 preprocessing pipeline
=================================================================
Replaces NSL-KDD (1999) with UNSW-NB15 (2015) — a modern dataset built
from real normal traffic + synthetic contemporary attacks, covering
9 attack families instead of NSL-KDD's dated categories:

    Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic,
    Reconnaissance, Shellcode, Worms

WHY THIS FEATURE SUBSET:
UNSW-NB15 has 49 raw features, but many (jitter, TCP sequence numbers,
precise round-trip timing) require deep packet inspection tools like
Bro/Argus that aren't practical to reproduce from raw Scapy captures.
This pipeline uses a deliberately chosen ~23-feature subset that:
  1. Has strong discriminative power for the 9 attack classes
  2. Can realistically be computed live from captured packets
     (this keeps SONA's live-capture pipeline upgradeable to match —
     that's the next step after this trains successfully)

Download the dataset from Kaggle before running this:
  https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
  Files needed: UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv
  Place both in data/unsw/
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os


# ── Feature selection ─────────────────────────────────────────────────────────
# Chosen because they're derivable from live packet/flow tracking, unlike
# UNSW-NB15's jitter/TCP-sequence/RTT features which need specialised tools.

NUMERIC_COLS = [
    "dur",              # flow duration
    "spkts", "dpkts",   # packets sent / received
    "sbytes", "dbytes", # bytes sent / received
    "rate",             # packets per second
    "sttl", "dttl",     # time-to-live (src/dst) — easy from IP header
    "sload", "dload",   # bits per second (src/dst)
    "swin", "dwin",     # TCP window size (src/dst)
    "smean", "dmean",   # mean packet size (src/dst)
    "ct_srv_src",       # connections to same service from same src (2s window)
    "ct_state_ttl",     # connections with same state + ttl
    "ct_dst_ltm",       # connections to same dst (last 100 connections)
    "ct_src_dport_ltm", # connections from same src to same dst port
    "ct_dst_sport_ltm", # connections to same dst from same src port
    "ct_dst_src_ltm",   # connections between same src/dst pair
    "ct_src_ltm",       # connections from same src
    "ct_srv_dst",       # connections to same service at same dst
    "is_sm_ips_ports",  # src/dst use same IP and port (analogous to NSL-KDD 'land')
]

CATEGORICAL_COLS = ["proto", "service", "state"]

TARGET_COL = "attack_cat"


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load UNSW-NB15 pre-split train/test CSVs."""
    print("Loading UNSW-NB15 data...")
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    print(f"  Train: {train.shape[0]:,} rows | Test: {test.shape[0]:,} rows")
    return train, test


def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise attack_cat values — the raw CSV has inconsistent casing
    and stray whitespace (e.g. ' Fuzzers', 'Backdoor' vs 'Backdoors').
    """
    df = df.copy()
    df[TARGET_COL] = (
        df[TARGET_COL]
        .astype(str)
        .str.strip()
        .replace({
            "Backdoor":  "Backdoors",
            "nan":       "Normal",
        })
    )
    df.loc[df[TARGET_COL].isna() | (df[TARGET_COL] == ""), TARGET_COL] = "Normal"
    return df


def clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise proto/service/state — UNSW-NB15 uses '-' for 'unknown'."""
    df = df.copy()
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].replace("-", "none")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few derived features, mirroring the approach used for NSL-KDD."""
    df = df.copy()
    df["bytes_ratio"]   = df["sbytes"] / (df["dbytes"] + 1)
    df["pkts_ratio"]    = df["spkts"]  / (df["dpkts"]  + 1)
    df["load_diff"]     = df["sload"] - df["dload"]
    df["bidirectional"] = ((df["sbytes"] > 0) & (df["dbytes"] > 0)).astype(int)
    df["zero_bytes"]    = ((df["sbytes"] == 0) & (df["dbytes"] == 0)).astype(int)
    return df


def get_feature_cols() -> list[str]:
    """All features used for training, in a fixed order."""
    engineered = ["bytes_ratio", "pkts_ratio", "load_diff",
                  "bidirectional", "zero_bytes"]
    return NUMERIC_COLS + CATEGORICAL_COLS + engineered


def encode_and_scale(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Encode categoricals, scale numerics — same pattern as the NSL-KDD pipeline."""
    train = train.copy()
    test  = test.copy()
    encoders = {}

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        test[col] = test[col].astype(str).map(
            lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
        )
        encoders[col] = le

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    X_test  = scaler.transform(test[feature_cols])
    encoders["scaler"] = scaler

    return X_train, X_test, encoders


def prepare_pipeline(
    train_path: str = "data/unsw/UNSW_NB15_training-set.csv",
    test_path:  str = "data/unsw/UNSW_NB15_testing-set.csv",
    save_encoders: bool = True,
    model_dir: str = "models_unsw",
) -> dict:
    """Full preprocessing pipeline for UNSW-NB15."""
    train_df, test_df = load_data(train_path, test_path)

    train_df = clean_labels(train_df)
    test_df  = clean_labels(test_df)
    train_df = clean_categoricals(train_df)
    test_df  = clean_categoricals(test_df)
    train_df = engineer_features(train_df)
    test_df  = engineer_features(test_df)

    feature_cols = get_feature_cols()
    print(f"  Features: {len(feature_cols)}")

    label_enc = LabelEncoder()
    y_train = label_enc.fit_transform(train_df[TARGET_COL])
    # Some test-set categories may differ slightly — map unseen to Normal's class
    def safe_transform(val):
        return label_enc.transform([val])[0] if val in label_enc.classes_ else \
               label_enc.transform(["Normal"])[0]
    y_test = np.array([safe_transform(v) for v in test_df[TARGET_COL]])

    X_train, X_test, encoders = encode_and_scale(train_df, test_df, feature_cols)

    if save_encoders:
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(encoders,     os.path.join(model_dir, "encoders.pkl"))
        joblib.dump(feature_cols, os.path.join(model_dir, "feature_cols.pkl"))
        joblib.dump(label_enc,    os.path.join(model_dir, "label_encoder.pkl"))
        print(f"  Encoders saved to {model_dir}/")

    print("\nClass distribution (train):")
    unique, counts = np.unique(y_train, return_counts=True)
    labels = label_enc.inverse_transform(unique)
    for lbl, cnt in sorted(zip(labels, counts), key=lambda x: -x[1]):
        print(f"  {lbl:15s}: {cnt:>7,}  ({cnt/len(y_train)*100:.1f}%)")

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_cols": feature_cols,
        "encoders": encoders,
        "label_encoder": label_enc,
    }


if __name__ == "__main__":
    data = prepare_pipeline()
    print("\nPreprocessing complete.")
    print(f"X_train shape: {data['X_train'].shape}")
    print(f"X_test  shape: {data['X_test'].shape}")
