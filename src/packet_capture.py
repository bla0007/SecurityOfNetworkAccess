"""
packet_capture.py — SONA v2 Module 1
=====================================
Live packet capture and NSL-KDD feature extraction engine.

What this does:
  1. Sniffs live packets off your network interface using Scapy
  2. Tracks connection state across packets (src_ip, dst_ip, port, protocol)
  3. Extracts the same 41 features the ML model was trained on
  4. Yields feature records ready for the model to classify

Run standalone to test:
    python src/packet_capture.py

Architecture:
  PacketSniffer  → raw packets off the wire
  ConnectionTracker  → groups packets into connections, computes stats
  FeatureExtractor   → converts a connection into the 41-feature vector
  LiveCapture        → ties it all together, yields classified records
"""

import time
import threading
import queue
from collections import defaultdict, deque
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import joblib
import os

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, get_if_list
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("[WARN] Scapy not installed. Run: pip install scapy")


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Connection:
    """
    Tracks state for a single network connection.
    A connection = unique (src_ip, dst_ip, dst_port, protocol) tuple.
    We window connections over TIME_WINDOW seconds.
    """
    src_ip:      str
    dst_ip:      str
    src_port:    int
    dst_port:    int
    protocol:    str          # tcp / udp / icmp

    # Timing
    start_time:  float = field(default_factory=time.time)
    last_seen:   float = field(default_factory=time.time)

    # Byte counts
    src_bytes:   int = 0
    dst_bytes:   int = 0

    # TCP flag counters
    syn_count:   int = 0
    fin_count:   int = 0
    rst_count:   int = 0
    ack_count:   int = 0
    urg_count:   int = 0
    psh_count:   int = 0

    # Misc
    packet_count:  int = 0
    wrong_fragment:int = 0
    urgent_count:  int = 0
    land:          int = 0     # src_ip == dst_ip AND src_port == dst_port

    # Service guess (from dst_port)
    service:     str = "other"
    flag:        str = "OTH"   # connection state flag
    logged_in:   int = 0


# Map common destination ports to NSL-KDD service names
PORT_SERVICE_MAP = {
    20: "ftp_data", 21: "ftp", 22: "ssh", 23: "telnet",
    25: "smtp", 53: "domain_u", 67: "dhcp", 68: "dhcp",
    80: "http", 110: "pop_3", 111: "sunrpc", 119: "nntp",
    123: "ntp_u", 143: "imap4", 161: "snmp", 162: "snmp",
    179: "bgp", 194: "IRC", 389: "ldap", 443: "http_443",
    445: "microsoft_ds", 512: "exec", 513: "login", 514: "shell",
    515: "printer", 520: "ef", 540: "uucp", 543: "klogin",
    544: "kshell", 587: "smtp", 993: "imap4", 995: "pop_3",
    1080: "proxy", 1433: "mssql", 3306: "sql_net",
    3389: "remote_job", 5432: "sql_net", 6000: "X11",
    6667: "IRC", 8080: "http_8001",
}

# NSL-KDD TCP connection flag meanings
# SF = normal established+closed, S0 = SYN only (no response), etc.
def infer_flag(conn: Connection) -> str:
    if conn.protocol != "tcp":
        return "SF" if conn.packet_count > 1 else "S0"
    if conn.syn_count > 0 and conn.fin_count > 0:
        return "SF"   # Normal: SYN → SYN-ACK → ... → FIN
    if conn.syn_count > 0 and conn.rst_count > 0:
        return "RSTO" # Reset after SYN
    if conn.syn_count > 0 and conn.ack_count == 0:
        return "S0"   # SYN with no response (classic DoS/scan)
    if conn.rst_count > 0 and conn.syn_count == 0:
        return "RSTR" # Reset without SYN
    if conn.syn_count > 1:
        return "S1"   # Multiple SYNs
    return "OTH"


# ── Connection Tracker ────────────────────────────────────────────────────────

class ConnectionTracker:
    """
    Maintains a sliding window of connections.
    For each new packet, updates the relevant connection's stats.
    Periodically flushes expired connections for classification.

    TIME_WINDOW: how long (seconds) to track a connection before finalising it.
    COUNT_WINDOW: the 2-second window used for count/srv_count stats in NSL-KDD.
    """

    TIME_WINDOW  = 5.0   # seconds — connection expires after this much silence
    COUNT_WINDOW = 2.0   # seconds — NSL-KDD uses 2-second windows for rate stats

    def __init__(self):
        self._connections: dict[tuple, Connection] = {}
        self._lock = threading.Lock()
        # Recent connections for rate-based features (count, srv_count, etc.)
        # Each entry: (timestamp, dst_ip, dst_port, protocol, flag, had_error)
        self._recent: deque = deque(maxlen=500)

    @staticmethod
    def _is_multicast_or_broadcast(ip: str) -> bool:
        """
        True for multicast (224.0.0.0–239.255.255.255) and broadcast addresses.
        This is normal LAN discovery traffic (mDNS, IGMP, SSDP), not a real
        connection worth classifying — Snort/Suricata filter this too.
        """
        try:
            first_octet = int(ip.split(".")[0])
            return (224 <= first_octet <= 239) or ip == "255.255.255.255" or ip.endswith(".255")
        except (ValueError, IndexError):
            return False

    def process_packet(self, packet, ts: float = None) -> None:
        """
        Update connection state from a single packet.
        ts: optional explicit timestamp (used for offline pcap replay,
            where we must use the packet's OWN capture time, not
            wall-clock "now"). Defaults to time.time() for live capture.
        """
        if not SCAPY_AVAILABLE:
            return
        if IP not in packet:
            return

        now = ts if ts is not None else time.time()

        src_ip  = packet[IP].src
        dst_ip  = packet[IP].dst

        # Skip multicast/broadcast destinations — this is background LAN chatter
        # (mDNS, IGMP, SSDP, device discovery), not a real connection.
        # NSL-KDD (1999) has no concept of this traffic, so the model
        # misclassifies it as an attack. Real NIDS tools filter this too.
        if self._is_multicast_or_broadcast(dst_ip):
            return

        proto   = "other"
        src_port = 0
        dst_port = 0

        if TCP in packet:
            proto    = "tcp"
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
        elif UDP in packet:
            proto    = "udp"
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
        elif ICMP in packet:
            proto    = "icmp"

        key = (src_ip, dst_ip, src_port, dst_port, proto)

        with self._lock:
            if key not in self._connections:
                svc = PORT_SERVICE_MAP.get(dst_port, "other")
                land = 1 if (src_ip == dst_ip and src_port == dst_port) else 0
                self._connections[key] = Connection(
                    src_ip=src_ip, dst_ip=dst_ip,
                    src_port=src_port, dst_port=dst_port,
                    protocol=proto, service=svc, land=land,
                    start_time=now, last_seen=now,
                )

            conn = self._connections[key]
            conn.last_seen = now
            conn.packet_count += 1

            pkt_len = len(packet)

            # Attribute bytes to src or dst direction
            conn.src_bytes += pkt_len

            # Wrong fragments
            if packet[IP].frag > 0:
                conn.wrong_fragment += 1

            # TCP flag parsing
            if TCP in packet:
                flags = packet[TCP].flags
                if flags & 0x02: conn.syn_count += 1   # SYN
                if flags & 0x01: conn.fin_count += 1   # FIN
                if flags & 0x04: conn.rst_count += 1   # RST
                if flags & 0x10: conn.ack_count += 1   # ACK
                if flags & 0x20: conn.urg_count += 1   # URG
                if flags & 0x08: conn.psh_count += 1   # PSH
                if flags & 0x20: conn.urgent_count += 1
                # Simple heuristic: if we see ACK after SYN, assume logged in
                if flags & 0x10 and conn.syn_count > 0:
                    conn.logged_in = 1

    def flush_expired(self, now: float = None) -> list[Connection]:
        """
        Return and remove connections that haven't seen a packet recently.
        now: optional explicit clock (for offline pcap replay — use the
             pcap's own timestamps, not wall-clock time.time()).
        """
        now = now if now is not None else time.time()
        expired = []
        with self._lock:
            for key in list(self._connections):
                conn = self._connections[key]
                if now - conn.last_seen > self.TIME_WINDOW:
                    conn.flag = infer_flag(conn)
                    conn.duration = conn.last_seen - conn.start_time
                    self._recent.append((
                        now, conn.dst_ip, conn.dst_port,
                        conn.protocol, conn.flag,
                        1 if conn.syn_count > 0 and conn.ack_count == 0 else 0,
                        1 if conn.rst_count > 0 else 0,
                    ))
                    expired.append(conn)
                    del self._connections[key]
        return expired

    def flush_all(self, now: float = None) -> list[Connection]:
        """
        Finalise EVERY remaining connection regardless of the time window.
        Used at the end of an offline pcap file, where there's no more
        traffic coming to naturally expire the last few connections.
        """
        now = now if now is not None else time.time()
        finished = []
        with self._lock:
            for key in list(self._connections):
                conn = self._connections[key]
                conn.flag = infer_flag(conn)
                conn.duration = conn.last_seen - conn.start_time
                self._recent.append((
                    now, conn.dst_ip, conn.dst_port,
                    conn.protocol, conn.flag,
                    1 if conn.syn_count > 0 and conn.ack_count == 0 else 0,
                    1 if conn.rst_count > 0 else 0,
                ))
                finished.append(conn)
                del self._connections[key]
        return finished

    def compute_rate_features(self, conn: Connection, now: float = None) -> dict:
        """
        Compute the NSL-KDD rate-based features using the 2-second window.
        These capture how many *recent* connections went to the same host/service.
        now: optional explicit clock (for offline pcap replay).
        """
        now = now if now is not None else time.time()
        cutoff = now - self.COUNT_WINDOW

        with self._lock:
            recent = [r for r in self._recent if r[0] >= cutoff]

        total = len(recent) or 1  # avoid div/0

        # Connections to same dst_ip
        same_host  = [r for r in recent if r[1] == conn.dst_ip]
        # Connections to same dst_port/service
        same_srv   = [r for r in recent if r[2] == conn.dst_port]

        count     = len(same_host) or 1
        srv_count = len(same_srv)  or 1

        def rate(lst, predicate):
            return sum(1 for r in lst if predicate(r)) / len(lst) if lst else 0.0

        serror_rate     = rate(same_host, lambda r: r[5] == 1)
        rerror_rate     = rate(same_host, lambda r: r[6] == 1)
        same_srv_rate   = rate(same_host, lambda r: r[2] == conn.dst_port)
        diff_srv_rate   = 1.0 - same_srv_rate
        srv_serror_rate = rate(same_srv,  lambda r: r[5] == 1)
        srv_rerror_rate = rate(same_srv,  lambda r: r[6] == 1)

        # dst_host_* features — last 100 connections to same dst_ip
        host_recent = list(self._recent)[-100:]
        host_conns  = [r for r in host_recent if r[1] == conn.dst_ip] or recent
        hc = len(host_conns) or 1

        def hrate(lst, predicate):
            return sum(1 for r in lst if predicate(r)) / len(lst) if lst else 0.0

        return {
            "count":                       min(count, 511),
            "srv_count":                   min(srv_count, 511),
            "serror_rate":                 round(serror_rate, 4),
            "srv_serror_rate":             round(srv_serror_rate, 4),
            "rerror_rate":                 round(rerror_rate, 4),
            "srv_rerror_rate":             round(srv_rerror_rate, 4),
            "same_srv_rate":               round(same_srv_rate, 4),
            "diff_srv_rate":               round(diff_srv_rate, 4),
            "srv_diff_host_rate":          round(hrate(host_conns, lambda r: r[2] != conn.dst_port), 4),
            "dst_host_count":              min(len(host_conns), 255),
            "dst_host_srv_count":          min(len([r for r in host_conns if r[2] == conn.dst_port]), 255),
            "dst_host_same_srv_rate":      round(hrate(host_conns, lambda r: r[2] == conn.dst_port), 4),
            "dst_host_diff_srv_rate":      round(hrate(host_conns, lambda r: r[2] != conn.dst_port), 4),
            "dst_host_same_src_port_rate": round(hrate(host_conns, lambda r: r[2] == conn.src_port), 4),
            "dst_host_srv_diff_host_rate": round(hrate(host_conns, lambda r: r[1] != conn.dst_ip), 4),
            "dst_host_serror_rate":        round(hrate(host_conns, lambda r: r[5] == 1), 4),
            "dst_host_srv_serror_rate":    round(hrate(host_conns, lambda r: r[5] == 1 and r[2] == conn.dst_port), 4),
            "dst_host_rerror_rate":        round(hrate(host_conns, lambda r: r[6] == 1), 4),
            "dst_host_srv_rerror_rate":    round(hrate(host_conns, lambda r: r[6] == 1 and r[2] == conn.dst_port), 4),
        }


# ── Feature Extractor ─────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Converts a finalised Connection object into a 46-feature dict
    matching what the SONA model was trained on (41 NSL-KDD + 5 engineered).
    """

    def __init__(self, tracker: ConnectionTracker):
        self.tracker = tracker

    def extract(self, conn: Connection, now: float = None) -> dict:
        rates = self.tracker.compute_rate_features(conn, now=now)
        duration = getattr(conn, "duration", 0)

        record = {
            # Basic features
            "duration":           round(duration, 2),
            "protocol_type":      conn.protocol,
            "service":            conn.service,
            "flag":               conn.flag,
            "src_bytes":          conn.src_bytes,
            "dst_bytes":          conn.dst_bytes,
            "land":               conn.land,
            "wrong_fragment":     conn.wrong_fragment,
            "urgent":             conn.urgent_count,

            # Content features (hard to extract without deep inspection — sensible defaults)
            "hot":                0,
            "num_failed_logins":  0,
            "logged_in":          conn.logged_in,
            "num_compromised":    0,
            "root_shell":         0,
            "su_attempted":       0,
            "num_root":           0,
            "num_file_creations": 0,
            "num_shells":         0,
            "num_access_files":   0,
            "num_outbound_cmds":  0,
            "is_host_login":      0,
            "is_guest_login":     0,

            # Rate-based features from tracker
            **rates,
        }

        # Engineered features (same ones added in preprocess.py)
        src_b = record["src_bytes"]
        dst_b = record["dst_bytes"]
        record["bytes_ratio"]     = round(src_b / (dst_b + 1), 4)
        record["error_rate_diff"] = round(rates["serror_rate"] - rates["rerror_rate"], 4)
        record["bidirectional"]   = 1 if (src_b > 0 and dst_b > 0) else 0
        record["high_count"]      = 1 if rates["count"] > 100 else 0
        record["zero_bytes"]      = 1 if (src_b == 0 and dst_b == 0) else 0

        return record


# ── Live Capture Engine ───────────────────────────────────────────────────────

class LiveCapture:
    """
    Main engine — ties PacketSniffer, ConnectionTracker, FeatureExtractor
    and the trained model together.

    Usage:
        engine = LiveCapture(model_dir="models")
        engine.start(interface="Wi-Fi")   # or "eth0" on Linux
        for alert in engine.alerts():
            print(alert)
        engine.stop()
    """

    def __init__(self, model_dir: str = "models", dataset: str = "nsl_kdd"):
        """
        dataset: "nsl_kdd" (original) or "unsw" (modern UNSW-NB15 model).
        Each uses its own tracker + feature extractor, since UNSW-NB15
        flows are bidirectional while NSL-KDD connections are directional.
        """
        self.dataset = dataset

        if dataset == "unsw":
            from flow_capture import FlowTracker, UNSWFeatureExtractor
            self.tracker   = FlowTracker()
            self.extractor = UNSWFeatureExtractor(self.tracker)
        else:
            self.tracker   = ConnectionTracker()
            self.extractor = FeatureExtractor(self.tracker)

        self._queue    = queue.Queue()
        self._running  = False
        self._threads  = []

        # Load trained model artifacts
        self.model        = joblib.load(os.path.join(model_dir, "best_model.pkl"))
        # Same fix as pcap_analysis.py — predicting one connection at a
        # time with n_jobs=-1 causes a multiprocessing pool spin-up per
        # call, which is very slow/can appear to hang on Windows.
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
        print(f"[SONA] Model loaded ({dataset}) — classes: {self.label_names}")

    def _packet_callback(self, packet):
        """Called by Scapy for every captured packet."""
        self.tracker.process_packet(packet)

    def _flush_loop(self):
        """Background thread: flush expired connections and classify them."""
        while self._running:
            time.sleep(1.0)  # check every second
            expired = self.tracker.flush_expired()
            for conn in expired:
                try:
                    record = self.extractor.extract(conn)
                    alert  = self._classify(conn, record)
                    self._queue.put(alert)
                except Exception as e:
                    print(f"[SONA] Classification error: {e}")

    def _classify(self, conn, record: dict) -> dict:
        """Run the ML model on a feature record and return an alert dict."""
        if self.dataset == "unsw":
            from preprocess_unsw import CATEGORICAL_COLS
        else:
            from preprocess import CATEGORICAL_COLS

        row = pd.DataFrame([record])

        # Encode categoricals using the saved LabelEncoders
        for col in CATEGORICAL_COLS:
            if col in self.feature_cols:
                le  = self.encoders[col]
                val = str(row[col].iloc[0])
                row[col] = le.transform([val])[0] if val in le.classes_ else -1

        # Select features and scale
        # Only keep columns the model knows about
        for col in self.feature_cols:
            if col not in row.columns:
                row[col] = 0
        X = row[self.feature_cols].values
        X_scaled = self.encoders["scaler"].transform(X)

        pred_idx = self.model.predict(X_scaled)[0]
        pred_class = (
            self.label_enc.inverse_transform([pred_idx])[0]
            if self.label_enc else self.label_names[pred_idx]
        )

        proba = {}
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(X_scaled)[0]
            proba = dict(zip(self.label_names, [round(float(v), 4) for v in p]))

        confidence = proba.get(pred_class, 1.0)

        if self.dataset == "unsw":
            return {
                "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
                "src_ip":       conn.orig_src_ip,
                "dst_ip":       conn.orig_dst_ip,
                "src_port":     conn.orig_src_port,
                "dst_port":     conn.orig_dst_port,
                "protocol":     conn.protocol,
                "service":      conn.service,
                "state":        conn.state,
                "sbytes":       conn.sbytes,
                "dbytes":       conn.dbytes,
                "duration":     round(conn.duration, 2),
                "prediction":   pred_class,
                "confidence":   round(confidence, 4),
                "is_attack":    pred_class != "Normal",
                "probabilities": proba,
                "packets":      conn.spkts + conn.dpkts,
                # Aliases so the shared format_alert() (written for NSL-KDD
                # field names) works unchanged for UNSW alerts too:
                "flag":         conn.state,
                "src_bytes":    conn.sbytes,
                "dst_bytes":    conn.dbytes,
            }

        return {
            "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
            "src_ip":       conn.src_ip,
            "dst_ip":       conn.dst_ip,
            "src_port":     conn.src_port,
            "dst_port":     conn.dst_port,
            "protocol":     conn.protocol,
            "service":      conn.service,
            "flag":         conn.flag,
            "src_bytes":    conn.src_bytes,
            "dst_bytes":    conn.dst_bytes,
            "duration":     round(getattr(conn, "duration", 0), 2),
            "prediction":   pred_class,
            "confidence":   round(confidence, 4),
            "is_attack":    pred_class != "Normal",
            "probabilities": proba,
            "packets":      conn.packet_count,
        }

    def start(self, interface: str = None, packet_count: int = 0):
        """
        Start capturing.
        interface: network interface name.
                   None = Scapy picks the default.
                   On Windows use names like 'Wi-Fi' or 'Ethernet'.
                   On Linux use 'eth0', 'wlan0', 'lo', etc.
        packet_count: 0 = capture forever (until stop() is called).
        """
        if not SCAPY_AVAILABLE:
            print("[ERROR] Scapy not installed. Run: pip install scapy")
            return

        self._running = True

        # Start the flush/classify thread
        flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        flush_thread.start()
        self._threads.append(flush_thread)

        # Start Scapy sniffing in its own thread
        def sniff_thread():
            print(f"[SONA] Sniffing on interface: {interface or 'default'}")
            print("[SONA] Capturing... press Ctrl+C to stop.\n")
            sniff(
                iface=interface,
                prn=self._packet_callback,
                store=False,
                count=packet_count,
                stop_filter=lambda _: not self._running,
            )

        t = threading.Thread(target=sniff_thread, daemon=True)
        t.start()
        self._threads.append(t)

    def stop(self):
        """Stop all threads."""
        self._running = False
        for t in self._threads:
            t.join(timeout=3.0)
        print("[SONA] Capture stopped.")

    def alerts(self):
        """
        Generator — yields alert dicts as they arrive.
        Blocks waiting for the next alert.
        """
        while self._running:
            try:
                yield self._queue.get(timeout=1.0)
            except queue.Empty:
                continue


# ── Utility helpers ───────────────────────────────────────────────────────────

def list_interfaces() -> list[str]:
    """Print available network interfaces (best-effort names)."""
    if not SCAPY_AVAILABLE:
        return []
    ifaces = get_if_list()
    print("\nAvailable interfaces (raw):")
    for i, iface in enumerate(ifaces):
        print(f"  [{i}] {iface}")
    return ifaces


def find_interface_by_ip(target_ip: str) -> str | None:
    """
    Find the exact interface Scapy should use to see traffic destined for
    target_ip. This is the RELIABLE way to pick an interface on Windows —
    typing a friendly name like 'Wi-Fi' often silently fails because
    Npcap/Scapy identify adapters by GUID internally, not the Control
    Panel name. Matching by IP address sidesteps that entirely.
    """
    if not SCAPY_AVAILABLE:
        return None
    try:
        from scapy.arch.windows import get_windows_if_list
        for iface in get_windows_if_list():
            if target_ip in iface.get("ips", []):
                return iface["name"]
    except ImportError:
        pass  # Not on Windows — get_windows_if_list only exists there
    return None


def print_interfaces_with_ips():
    """
    Windows-friendly interface listing: shows friendly name + IP addresses
    together, so you can visually match which one has your real LAN IP
    instead of guessing from a bare name list.
    """
    if not SCAPY_AVAILABLE:
        return
    try:
        from scapy.arch.windows import get_windows_if_list
        print("\nDetected interfaces:")
        for iface in get_windows_if_list():
            ips = ", ".join(iface.get("ips", [])) or "(no IP)"
            print(f"  name={iface['name']!r:30s} ips={ips}")
    except ImportError:
        list_interfaces()


def format_alert(alert: dict) -> str:
    """Pretty-print a single alert to terminal."""
    icon  = "🚨" if alert["is_attack"] else "✅"
    color = "\033[91m" if alert["is_attack"] else "\033[92m"
    reset = "\033[0m"
    return (
        f"{color}{icon} [{alert['timestamp']}] "
        f"{alert['src_ip']}:{alert['src_port']} → "
        f"{alert['dst_ip']}:{alert['dst_port']} "
        f"({alert['protocol'].upper()}/{alert['service']}) "
        f"| {alert['prediction'].upper()} "
        f"({alert['confidence']:.0%} confidence) "
        f"| {alert['src_bytes']}B sent | flag={alert['flag']}"
        f"{reset}"
    )


# ── Standalone test mode ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  SONA v2 — Module 1: Live Packet Capture")
    print("=" * 60)

    # Step 1: list interfaces so user can pick the right one
    ifaces = list_interfaces()

    # Step 2: pick interface
    # Change this to your interface name if needed
    # Windows examples: 'Wi-Fi', 'Ethernet', 'Local Area Connection'
    # Linux examples:   'eth0', 'wlan0', 'ens33'
    INTERFACE = None  # None = Scapy auto-picks

    if len(sys.argv) > 1:
        INTERFACE = sys.argv[1]
        print(f"\nUsing interface from argument: {INTERFACE}")
    else:
        print("\nTip: run with interface name as argument:")
        print("     python src/packet_capture.py Wi-Fi")
        print("     python src/packet_capture.py eth0\n")

    # Step 3: start the engine
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
    if not os.path.exists(os.path.join(MODEL_DIR, "best_model.pkl")):
        print("[ERROR] No trained model found.")
        print("        Run python src/train.py first, then come back here.")
        sys.exit(1)

    engine = LiveCapture(model_dir=MODEL_DIR)

    try:
        engine.start(interface=INTERFACE)
        print(f"\n{'─'*60}")
        print("  Live alerts (connections classified every ~5 seconds)")
        print(f"{'─'*60}\n")

        alert_count = 0
        attack_count = 0

        for alert in engine.alerts():
            print(format_alert(alert))
            alert_count  += 1
            attack_count += int(alert["is_attack"])

            # Print a summary every 20 alerts
            if alert_count % 20 == 0:
                print(f"\n  [{alert_count} connections analysed | "
                      f"{attack_count} threats detected]\n")

    except KeyboardInterrupt:
        print("\n\nStopping capture...")
        engine.stop()
        print(f"\nSession summary:")
        print(f"  Total connections analysed : {alert_count}")
        print(f"  Threats detected           : {attack_count}")
        print(f"  Clean connections          : {alert_count - attack_count}")
