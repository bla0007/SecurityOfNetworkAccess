"""
predict.py — SONA (Security of Network Access)
===============================================
Run inference on a single network connection record.
Used by the SONA dashboard and for testing the saved model.
"""

import numpy as np
import pandas as pd
import joblib
import os

from preprocess import CATEGORICAL_COLS, COLUMNS, engineer_features


def load_artifacts(model_dir: str = "models") -> dict:
    """Load saved model, encoders, and label info."""
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    # Single-row predictions don't benefit from n_jobs=-1 parallelism and
    # it's slow to spin up on Windows — force single-threaded inference.
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    return {
        "model":         model,
        "encoders":      joblib.load(os.path.join(model_dir, "encoders.pkl")),
        "feature_cols":  joblib.load(os.path.join(model_dir, "feature_cols.pkl")),
        "label_names":   joblib.load(os.path.join(model_dir, "label_names.pkl")),
        "label_encoder": joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
        if os.path.exists(os.path.join(model_dir, "label_encoder.pkl")) else None,
    }


def predict_single(record: dict, artifacts: dict) -> dict:
    """
    Predict the attack category for a single connection record.

    Args:
        record: dict with network feature values (see COLUMNS in preprocess.py)
        artifacts: loaded model artifacts from load_artifacts()

    Returns:
        dict with 'predicted_class', 'confidence', 'all_probabilities'
    """
    encoders     = artifacts["encoders"]
    feature_cols = artifacts["feature_cols"]
    model        = artifacts["model"]
    label_names  = artifacts["label_names"]
    label_enc    = artifacts["label_encoder"]

    # Build a single-row DataFrame
    row = pd.DataFrame([record])

    # Add engineered features
    row = engineer_features(row)

    # Encode categoricals
    for col in CATEGORICAL_COLS:
        if col in feature_cols:
            le = encoders[col]
            val = str(row[col].iloc[0])
            row[col] = le.transform([val])[0] if val in le.classes_ else -1

    # Select and scale features
    X = row[feature_cols].values
    X_scaled = encoders["scaler"].transform(X)

    # Predict
    pred_idx = model.predict(X_scaled)[0]

    if label_enc is not None:
        pred_class = label_enc.inverse_transform([pred_idx])[0]
    else:
        pred_class = label_names[pred_idx]

    # Probabilities (if model supports it)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        all_proba = dict(zip(label_names, proba))
        confidence = float(proba[pred_idx])
    else:
        all_proba = {pred_class: 1.0}
        confidence = 1.0

    return {
        "predicted_class":  pred_class,
        "confidence":       confidence,
        "all_probabilities": all_proba,
        "is_attack":        pred_class != "Normal",
    }


# ── Example records for testing ──────────────────────────────────────────────

EXAMPLE_NORMAL = {
    "duration": 0, "protocol_type": "tcp", "service": "http",
    "flag": "SF", "src_bytes": 232, "dst_bytes": 8153,
    "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
    "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0,
    "root_shell": 0, "su_attempted": 0, "num_root": 0,
    "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
    "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
    "count": 8, "srv_count": 8, "serror_rate": 0.0,
    "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
    "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
    "dst_host_count": 9, "dst_host_srv_count": 9,
    "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
    "dst_host_same_src_port_rate": 0.11, "dst_host_srv_diff_host_rate": 0.0,
    "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
    "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0,
    "difficulty_level": 0,
}

EXAMPLE_DOS = {
    "duration": 0, "protocol_type": "tcp", "service": "http",
    "flag": "S0", "src_bytes": 0, "dst_bytes": 0,
    "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
    "num_failed_logins": 0, "logged_in": 0, "num_compromised": 0,
    "root_shell": 0, "su_attempted": 0, "num_root": 0,
    "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
    "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
    "count": 511, "srv_count": 511, "serror_rate": 1.0,
    "srv_serror_rate": 1.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
    "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
    "dst_host_count": 255, "dst_host_srv_count": 255,
    "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
    "dst_host_same_src_port_rate": 0.0, "dst_host_srv_diff_host_rate": 0.0,
    "dst_host_serror_rate": 1.0, "dst_host_srv_serror_rate": 1.0,
    "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0,
    "difficulty_level": 0,
}


if __name__ == "__main__":
    print("Loading model artifacts...")
    artifacts = load_artifacts()

    for name, record in [("Normal traffic", EXAMPLE_NORMAL), ("DoS attack", EXAMPLE_DOS)]:
        result = predict_single(record, artifacts)
        print(f"\n{name}:")
        print(f"  Predicted  : {result['predicted_class']}")
        print(f"  Is attack  : {result['is_attack']}")
        print(f"  Confidence : {result['confidence']:.1%}")
        print(f"  Probabilities:")
        for cls, prob in sorted(result["all_probabilities"].items(),
                                key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 20)
            print(f"    {cls:8s} {prob:.1%}  {bar}")
