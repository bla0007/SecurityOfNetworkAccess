"""
pcap_analysis.py — SONA Offline PCAP Analyzer
================================================
Analyzes a pre-captured .pcap/.pcapng file instead of live traffic.

Why this matters for SONA:
  Live capture over a bridged WiFi adapter is unreliable (802.11 doesn't
  support arbitrary MAC bridging the way Ethernet does — a well-documented
  VirtualBox limitation). Offline pcap analysis sidesteps that completely:
  you can capture traffic with Wireshark/tcpdump on ANY working setup
  (Host-Only VirtualBox network, a real Ethernet-connected lab, or even
  a public attack-traffic pcap from malware-traffic-analysis.net), then
  feed the file straight into SONA's exact same feature-extraction +
  ML classification pipeline used in live mode.

  This is also just a genuinely real capability — SOC analysts routinely
  analyze pcaps captured elsewhere rather than only watching live feeds.

Usage:
    python src/pcap_analysis.py --file capture.pcap
    python src/pcap_analysis.py --file capture.pcap --export report.csv
    python src/pcap_analysis.py --file capture.pcap --min-confidence 0.6
"""

import sys
import os
import argparse
import time
import json
import csv as csv_module
from collections import Counter, defaultdict
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(__file__))

import joblib
import pandas as pd

try:
    from scapy.utils import PcapReader
    from scapy.all import IP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False


def parse_args():
    p = argparse.ArgumentParser(description="SONA — Offline PCAP Analyzer")
    p.add_argument("--file", type=str, required=True,
                   help="Path to a .pcap or .pcapng file")
    p.add_argument("--model-version", choices=["unsw", "nsl-kdd"], default="unsw",
                   help="Which trained model to use (default: unsw)")
    p.add_argument("--model-dir", type=str, default=None,
                   help="Override model directory (default: models_unsw or models, "
                        "based on --model-version)")
    p.add_argument("--export", type=str, default=None,
                   help="Export full results to this CSV path")
    p.add_argument("--min-confidence", type=float, default=0.0,
                   help="Only show detections above this confidence (0.0-1.0)")
    p.add_argument("--flush-every", type=int, default=200,
                   help="Check for expired connections every N packets "
                        "(lower = more accurate windowing, slower)")
    p.add_argument("--feed-dashboard", action="store_true",
                   help="Write detected attacks into logs/threat_log.json + .csv "
                        "so the live dashboard (dashboard/live.py) displays them "
                        "immediately — no live capture needed for a demo.")
    return p.parse_args()


class PcapAnalyzer:
    """
    Replays a pcap file through SONA's exact live-capture feature
    pipeline, using each packet's own capture timestamp as the clock
    instead of wall-clock time — so 5-second connection windows behave
    correctly even though the whole file might replay in under a second.

    Works with either model via model_version — NSL-KDD's directional
    ConnectionTracker or UNSW-NB15's bidirectional FlowTracker.
    """

    def __init__(self, model_dir: str, model_version: str = "unsw"):
        self.model_version = model_version

        if model_version == "unsw":
            from flow_capture import FlowTracker, UNSWFeatureExtractor
            from preprocess_unsw import CATEGORICAL_COLS
            self.tracker   = FlowTracker()
            self.extractor = UNSWFeatureExtractor(self.tracker)
        else:
            from packet_capture import ConnectionTracker, FeatureExtractor
            from preprocess import CATEGORICAL_COLS
            self.tracker   = ConnectionTracker()
            self.extractor = FeatureExtractor(self.tracker)

        self.categorical_cols = CATEGORICAL_COLS
        self.model        = joblib.load(os.path.join(model_dir, "best_model.pkl"))
        # Predicting one connection at a time with n_jobs=-1 (set during
        # training) causes a multiprocessing worker pool to spin up for
        # EVERY single prediction — extremely slow/appears to hang on
        # Windows. Parallelism only helps for big batches, not single
        # rows, so force single-threaded inference here.
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1
        self.encoders     = joblib.load(os.path.join(model_dir, "encoders.pkl"))
        self.feature_cols = joblib.load(os.path.join(model_dir, "feature_cols.pkl"))
        self.label_names  = joblib.load(os.path.join(model_dir, "label_names.pkl"))
        self.label_enc    = (
            joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
            if os.path.exists(os.path.join(model_dir, "label_encoder.pkl"))
            else None
        )
        self.results: list[dict] = []

    def _classify(self, conn, now: float) -> dict:
        record = self.extractor.extract(conn)
        row = pd.DataFrame([record])

        for col in self.categorical_cols:
            if col in self.feature_cols:
                le  = self.encoders[col]
                val = str(row[col].iloc[0])
                row[col] = le.transform([val])[0] if val in le.classes_ else -1

        for col in self.feature_cols:
            if col not in row.columns:
                row[col] = 0

        X = row[self.feature_cols].values
        X_scaled = self.encoders["scaler"].transform(X)
        pred_idx = self.model.predict(X_scaled)[0]
        pred = (
            self.label_enc.inverse_transform([pred_idx])[0]
            if self.label_enc else self.label_names[pred_idx]
        )

        proba = {}
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(X_scaled)[0]
            proba = dict(zip(self.label_names, [round(float(v), 4) for v in p]))
        confidence = proba.get(pred, 1.0)

        if self.model_version == "unsw":
            return {
                "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                "src_ip":        conn.orig_src_ip,
                "dst_ip":        conn.orig_dst_ip,
                "src_port":      conn.orig_src_port,
                "dst_port":      conn.orig_dst_port,
                "protocol":      conn.protocol,
                "service":       conn.service,
                "flag":          conn.state,
                "src_bytes":     conn.sbytes,
                "dst_bytes":     conn.dbytes,
                "duration":      round(conn.duration, 2),
                "prediction":    pred,
                "confidence":    round(confidence, 4),
                "is_attack":     pred != "Normal",
                "probabilities": proba,
                "packets":       conn.spkts + conn.dpkts,
            }

        return {
            "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "src_ip":        conn.src_ip,
            "dst_ip":        conn.dst_ip,
            "src_port":      conn.src_port,
            "dst_port":      conn.dst_port,
            "protocol":      conn.protocol,
            "service":       conn.service,
            "flag":          conn.flag,
            "src_bytes":     conn.src_bytes,
            "dst_bytes":     conn.dst_bytes,
            "duration":      round(getattr(conn, "duration", 0), 2),
            "prediction":    pred,
            "confidence":    round(confidence, 4),
            "is_attack":     pred != "Normal",
            "probabilities": proba,
            "packets":       conn.packet_count,
        }

    def analyze(self, pcap_path: str, flush_every: int = 200):
        if not SCAPY_AVAILABLE:
            raise RuntimeError("Scapy not installed. Run: pip install scapy")
        if not os.path.exists(pcap_path):
            raise FileNotFoundError(f"pcap file not found: {pcap_path}")

        print(f"Reading: {pcap_path}")
        packet_count = 0
        last_ts = None

        with PcapReader(pcap_path) as reader:
            for pkt in reader:
                packet_count += 1
                if IP not in pkt:
                    continue

                ts = float(pkt.time)
                last_ts = ts
                self.tracker.process_packet(pkt, ts=ts)

                # Periodically flush expired connections using the pcap's
                # OWN timeline, not wall-clock time
                if packet_count % flush_every == 0:
                    expired = self.tracker.flush_expired(now=ts)
                    for conn in expired:
                        self.results.append(self._classify(conn, ts))

        # Finalise anything still open at end of file
        if last_ts is not None:
            remaining = self.tracker.flush_all(now=last_ts)
            for conn in remaining:
                self.results.append(self._classify(conn, last_ts))

        print(f"Packets read       : {packet_count:,}")
        print(f"Connections found  : {len(self.results):,}")
        return self.results

    def summary(self) -> dict:
        total = len(self.results)
        attacks = [r for r in self.results if r["is_attack"]]
        by_type = Counter(r["prediction"] for r in self.results)
        top_attackers = Counter(r["src_ip"] for r in attacks)
        return {
            "total_connections": total,
            "total_attacks":     len(attacks),
            "by_type":           dict(by_type),
            "top_attacker_ips":  dict(top_attackers.most_common(10)),
        }

    def export_csv(self, path: str):
        if not self.results:
            print("Nothing to export.")
            return
        fields = [
            "timestamp", "src_ip", "dst_ip", "src_port", "dst_port",
            "protocol", "service", "flag", "src_bytes", "dst_bytes",
            "duration", "prediction", "confidence", "is_attack", "packets",
        ]
        with open(path, "w", newline="") as f:
            writer = csv_module.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for r in self.results:
                writer.writerow(r)
        print(f"Exported CSV → {path}")

    def export_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Exported JSON → {path}")

    def feed_dashboard(self, log_dir: str = "logs"):
        """
        Write every detected attack into logs/threat_log.json + .csv using
        the SAME schema ResponseEngine's ThreatLogger writes — so the live
        dashboard (dashboard/live.py) picks these up immediately, with no
        live capture needed. Existing log entries are preserved (appended).
        """
        os.makedirs(log_dir, exist_ok=True)
        json_path = os.path.join(log_dir, "threat_log.json")
        csv_path  = os.path.join(log_dir, "threat_log.csv")

        csv_fields = [
            "timestamp", "src_ip", "dst_ip", "src_port", "dst_port",
            "protocol", "service", "attack_type", "confidence",
            "flag", "src_bytes", "action", "rule_name", "strike",
        ]

        # Load existing events so we APPEND, never overwrite live-session data
        existing = []
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    existing = json.load(f)
            except Exception:
                existing = []

        attacks = [r for r in self.results if r["is_attack"]]
        new_events = [
            {
                "timestamp":   r["timestamp"],
                "src_ip":      r["src_ip"],
                "dst_ip":      r["dst_ip"],
                "src_port":    r["src_port"],
                "dst_port":    r["dst_port"],
                "protocol":    r["protocol"],
                "service":     r["service"],
                "attack_type": r["prediction"],
                "confidence":  r["confidence"],
                "flag":        r["flag"],
                "src_bytes":   r["src_bytes"],
                "action":      "LOGGED(PCAP_ANALYSIS)",
                "rule_name":   "",
                "strike":      0,
            }
            for r in attacks
        ]

        all_events = existing + new_events
        with open(json_path, "w") as f:
            json.dump(all_events, f, indent=2)

        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv_module.DictWriter(f, fieldnames=csv_fields)
            if write_header:
                writer.writeheader()
            for e in new_events:
                writer.writerow(e)

        print(f"\nFed {len(new_events)} detections into the live dashboard's logs:")
        print(f"  {json_path}")
        print(f"  {csv_path}")
        print(f"  Open the dashboard (streamlit run dashboard/live.py) to see them.")


def main():
    args = parse_args()

    print("=" * 65)
    print("  SONA — Offline PCAP Analyzer")
    print("=" * 65)

    model_dir = args.model_dir or ("models_unsw" if args.model_version == "unsw" else "models")
    print(f"  Model: {args.model_version.upper()}  ({model_dir}/)")

    if not os.path.exists(os.path.join(model_dir, "best_model.pkl")):
        print(f"[ERROR] No trained model found in {model_dir}/")
        train_script = "train_unsw.py" if args.model_version == "unsw" else "train.py"
        print(f"        Run python src/{train_script} first.")
        sys.exit(1)

    # PcapAnalyzer._classify() normalises BOTH model versions to the same
    # field names (src_bytes/dst_bytes/flag) for a consistent CSV export
    # schema — so the same formatter works for both, unlike LiveCapture's
    # raw dicts which differ by dataset.
    from packet_capture import format_alert

    analyzer = PcapAnalyzer(model_dir=model_dir, model_version=args.model_version)
    results  = analyzer.analyze(args.file, flush_every=args.flush_every)

    print(f"\n{'─'*65}")
    print("  Detections")
    print(f"{'─'*65}\n")

    shown = 0
    for r in results:
        if r["is_attack"] and r["confidence"] >= args.min_confidence:
            print(format_alert(r))
            shown += 1
    if shown == 0:
        print("  No attacks matched your confidence threshold.")

    summary = analyzer.summary()
    print(f"\n{'═'*65}")
    print("  SUMMARY")
    print(f"{'═'*65}")
    print(f"  Total connections analysed : {summary['total_connections']:,}")
    print(f"  Attacks detected           : {summary['total_attacks']:,}")
    print(f"\n  Breakdown by class:")
    for cls, cnt in sorted(summary["by_type"].items(), key=lambda x: -x[1]):
        bar = "█" * min(cnt, 50)
        print(f"    {cls:10s} {cnt:>5}  {bar}")

    if summary["top_attacker_ips"]:
        print(f"\n  Top attacker IPs:")
        for ip, cnt in summary["top_attacker_ips"].items():
            print(f"    {ip:20s} {cnt} detections")

    if args.export:
        analyzer.export_csv(args.export)
        json_path = os.path.splitext(args.export)[0] + ".json"
        analyzer.export_json(json_path)

    if args.feed_dashboard:
        analyzer.feed_dashboard()

    print(f"\n{'═'*65}\n")


if __name__ == "__main__":
    main()
