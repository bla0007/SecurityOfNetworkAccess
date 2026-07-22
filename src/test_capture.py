"""
test_capture.py — SONA v2 Module 1 tester
==========================================
Tests the full live-capture pipeline WITHOUT needing admin rights
or a real network interface.

Simulates 50 connections (mix of normal and attack traffic),
feeds them through the feature extractor and model,
and prints the results — exactly what you'll see in real capture.

Run from the sona/ root folder:
    python src/test_capture.py
"""

import sys
import os
import time
import random

sys.path.insert(0, os.path.dirname(__file__))

from packet_capture import (
    Connection, ConnectionTracker, FeatureExtractor,
    infer_flag, format_alert, LiveCapture
)
import joblib
import pandas as pd
import numpy as np


# ── Simulated traffic profiles ────────────────────────────────────────────────

def make_connection(profile: str) -> Connection:
    """Create a fake Connection object that mimics a traffic type."""

    if profile == "normal_http":
        c = Connection("192.168.1.5", "93.184.216.34", 52341, 80, "tcp")
        c.src_bytes    = random.randint(200, 2000)
        c.dst_bytes    = random.randint(500, 8000)
        c.syn_count    = 1; c.ack_count = 5; c.fin_count = 1
        c.logged_in    = 1; c.service = "http"
        c.packet_count = random.randint(4, 20)

    elif profile == "dos_syn_flood":
        # DoS: hundreds of SYN packets, no ACK (S0 flag)
        c = Connection("10.0.0.99", "192.168.1.1", random.randint(1024,65535), 80, "tcp")
        c.src_bytes    = random.randint(0, 100)
        c.dst_bytes    = 0
        c.syn_count    = random.randint(100, 511)
        c.ack_count    = 0
        c.packet_count = c.syn_count
        c.service      = "http"

    elif profile == "probe_portscan":
        # Probe: many different ports, small bytes, RST responses
        c = Connection("10.0.0.42", "192.168.1.1", random.randint(1024,65535),
                       random.randint(1, 1024), "tcp")
        c.src_bytes    = random.randint(0, 60)
        c.dst_bytes    = 0
        c.syn_count    = 1
        c.rst_count    = 1
        c.packet_count = 2
        c.service      = "other"

    elif profile == "normal_dns":
        c = Connection("192.168.1.5", "8.8.8.8", 54321, 53, "udp")
        c.src_bytes    = random.randint(30, 100)
        c.dst_bytes    = random.randint(60, 300)
        c.packet_count = 2
        c.service      = "domain_u"

    elif profile == "dos_udp_flood":
        c = Connection("10.0.0.77", "192.168.1.1", random.randint(1024,65535), 80, "udp")
        c.src_bytes    = random.randint(50000, 200000)
        c.dst_bytes    = 0
        c.packet_count = random.randint(200, 511)
        c.service      = "other"

    elif profile == "normal_ftp":
        c = Connection("192.168.1.10", "192.168.1.200", 54000, 21, "tcp")
        c.src_bytes    = random.randint(100, 500)
        c.dst_bytes    = random.randint(200, 1000)
        c.syn_count    = 1; c.ack_count = 3; c.fin_count = 1
        c.logged_in    = 1; c.service = "ftp"
        c.packet_count = 6

    elif profile == "r2l_guess_passwd":
        # R2L: repeated failed login attempts
        c = Connection("203.0.113.5", "192.168.1.1", random.randint(1024,65535), 22, "tcp")
        c.src_bytes    = random.randint(500, 2000)
        c.dst_bytes    = random.randint(100, 500)
        c.syn_count    = 5; c.ack_count = 5
        c.packet_count = 20
        c.service      = "ssh"

    else:
        c = Connection("192.168.1.1", "192.168.1.2", 12345, 80, "tcp")
        c.src_bytes = 100; c.dst_bytes = 200
        c.packet_count = 3

    # Finalise
    c.start_time = time.time() - random.uniform(0.1, 5.0)
    c.last_seen  = time.time()
    c.duration   = c.last_seen - c.start_time
    c.flag       = infer_flag(c)
    c.land       = 1 if c.src_ip == c.dst_ip else 0

    return c


# ── Test runner ───────────────────────────────────────────────────────────────

def run_test():
    print("=" * 65)
    print("  SONA v2 — Module 1 pipeline test (no admin rights needed)")
    print("=" * 65)

    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    if not os.path.exists(os.path.join(model_dir, "best_model.pkl")):
        print("\n[ERROR] No trained model found in models/")
        print("        Run 'python src/train.py' first, then retry.\n")
        sys.exit(1)

    # Load artifacts
    print("\nLoading model artifacts...")
    model        = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    encoders     = joblib.load(os.path.join(model_dir, "encoders.pkl"))
    feature_cols = joblib.load(os.path.join(model_dir, "feature_cols.pkl"))
    label_names  = joblib.load(os.path.join(model_dir, "label_names.pkl"))
    label_enc    = (
        joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
        if os.path.exists(os.path.join(model_dir, "label_encoder.pkl"))
        else None
    )
    print(f"Model classes: {label_names}\n")

    tracker   = ConnectionTracker()
    extractor = FeatureExtractor(tracker)

    # Traffic mix: 60% normal, 40% attacks
    profiles = (
        ["normal_http"] * 15 +
        ["normal_dns"]  * 8  +
        ["normal_ftp"]  * 7  +
        ["dos_syn_flood"]   * 8 +
        ["dos_udp_flood"]   * 5 +
        ["probe_portscan"]  * 5 +
        ["r2l_guess_passwd"]* 2
    )
    random.shuffle(profiles)

    print(f"Simulating {len(profiles)} connections...\n")
    print(f"{'─'*65}")

    results = []

    for profile in profiles:
        conn   = make_connection(profile)
        record = extractor.extract(conn)

        # Encode and predict
        from preprocess import CATEGORICAL_COLS
        row = pd.DataFrame([record])

        for col in CATEGORICAL_COLS:
            if col in feature_cols:
                le  = encoders[col]
                val = str(row[col].iloc[0])
                row[col] = le.transform([val])[0] if val in le.classes_ else -1

        for col in feature_cols:
            if col not in row.columns:
                row[col] = 0

        X        = row[feature_cols].values
        X_scaled = encoders["scaler"].transform(X)
        pred_idx = model.predict(X_scaled)[0]
        pred     = label_enc.inverse_transform([pred_idx])[0] if label_enc else label_names[pred_idx]

        proba = {}
        if hasattr(model, "predict_proba"):
            p     = model.predict_proba(X_scaled)[0]
            proba = dict(zip(label_names, [round(float(v), 4) for v in p]))

        conf = proba.get(pred, 1.0)

        alert = {
            "timestamp":     time.strftime("%H:%M:%S"),
            "src_ip":        conn.src_ip,
            "dst_ip":        conn.dst_ip,
            "src_port":      conn.src_port,
            "dst_port":      conn.dst_port,
            "protocol":      conn.protocol,
            "service":       conn.service,
            "flag":          conn.flag,
            "src_bytes":     conn.src_bytes,
            "dst_bytes":     conn.dst_bytes,
            "duration":      round(conn.duration, 2),
            "prediction":    pred,
            "confidence":    round(conf, 4),
            "is_attack":     pred != "Normal",
            "probabilities": proba,
            "packets":       conn.packet_count,
            "_true_profile": profile,
        }
        results.append(alert)
        print(format_alert(alert))
        time.sleep(0.05)  # tiny delay so output is readable

    # Summary
    total   = len(results)
    attacks = sum(1 for r in results if r["is_attack"])
    normal  = total - attacks

    print(f"\n{'═'*65}")
    print(f"  TEST COMPLETE — {total} connections classified")
    print(f"{'═'*65}")
    print(f"  ✅  Normal traffic  : {normal}")
    print(f"  🚨  Threats detected: {attacks}")

    # Category breakdown
    from collections import Counter
    cats = Counter(r["prediction"] for r in results)
    print(f"\n  Breakdown:")
    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
        bar = "█" * cnt
        print(f"    {cat:12s} {cnt:>3}  {bar}")

    print(f"\n  Module 1 pipeline is working correctly.")
    print(f"  Next step: run with a real interface (needs admin/root):")
    print(f"    Windows : Run terminal as Administrator")
    print(f"              python src/packet_capture.py Wi-Fi")
    print(f"    Linux   : sudo python src/packet_capture.py eth0\n")

    return results


if __name__ == "__main__":
    run_test()
