"""
flow_capture.py — SONA v3: Live capture for the UNSW-NB15 model
==================================================================
NSL-KDD (packet_capture.py) treats each traffic DIRECTION as its own
"connection" — A→B and B→A are two separate records. UNSW-NB15 instead
describes one bidirectional FLOW per conversation (spkts/dpkts and
sbytes/dbytes are counted together, from both directions, under a
single record). This module implements that properly instead of
bolting it onto the NSL-KDD tracker.

Architecture (mirrors packet_capture.py's shape, different internals):
  FlowTracker          → groups packets into bidirectional flows
  UNSWFeatureExtractor → converts a flow into the UNSW-NB15 feature set
  (LiveCapture in packet_capture.py drives both, chosen via `dataset=`)
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field


try:
    from scapy.all import IP, TCP, UDP, ICMP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class FlowConnection:
    """
    A single bidirectional flow. 'orig' = whichever side sent the FIRST
    packet we saw — that's what UNSW-NB15 calls "source" throughout.
    """
    orig_src_ip:   str
    orig_dst_ip:   str
    orig_src_port: int
    orig_dst_port: int
    protocol:      str    # tcp / udp / other

    start_time: float = field(default_factory=time.time)
    last_seen:  float = field(default_factory=time.time)

    spkts: int = 0   # packets sent BY the originator
    dpkts: int = 0   # packets sent back TO the originator
    sbytes: int = 0
    dbytes: int = 0

    sttl: int = 0    # last-seen TTL from originator
    dttl: int = 0    # last-seen TTL from the other side
    swin: int = 0    # last-seen TCP window from originator
    dwin: int = 0

    syn_count: int = 0
    fin_count: int = 0
    rst_count: int = 0
    ack_count: int = 0

    service:  str = "other"
    state:    str = "INT"
    duration: float = 0.0


PORT_SERVICE_MAP = {
    20: "ftp-data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
    53: "dns", 67: "dhcp", 68: "dhcp", 80: "http", 110: "pop3",
    119: "nntp", 123: "ntp", 143: "imap", 161: "snmp", 179: "bgp",
    194: "irc", 389: "ldap", 443: "ssl", 445: "smb", 465: "smtp",
    587: "smtp", 993: "imap", 995: "pop3", 1433: "mssql",
    3306: "mysql", 3389: "rdp", 3478: "stun", 5432: "postgresql", 6667: "irc",
    8080: "http",
}


def infer_state(conn: FlowConnection) -> str:
    """
    Best-effort approximation of Argus's connection "state" field.
    Exact reproduction isn't possible without Argus itself, but this
    captures the same broad categories UNSW-NB15 uses most often.
    """
    if conn.protocol != "tcp":
        return "CON" if conn.dpkts > 0 else "INT"
    if conn.syn_count > 0 and conn.fin_count > 0:
        return "FIN"          # Completed handshake + close
    if conn.syn_count > 0 and conn.rst_count > 0:
        return "RST"          # Reset after SYN
    if conn.syn_count > 0 and conn.ack_count > 0 and conn.dpkts > 0:
        return "CON"          # Established, still going
    if conn.syn_count > 0 and conn.dpkts == 0:
        return "REQ"          # SYN sent, no reply at all
    return "INT"               # Incomplete / inactive


class FlowTracker:
    """
    Bidirectional equivalent of ConnectionTracker (packet_capture.py).
    Both directions of one conversation map to the SAME flow record.
    """

    TIME_WINDOW = 5.0    # seconds of silence before a flow is finalised
    RECENT_MAXLEN = 100  # UNSW-NB15's ct_* features use a 100-connection
                         # rolling window (not a time window) per the
                         # original Moustafa & Slay feature definitions

    def __init__(self):
        self._flows: dict[tuple, FlowConnection] = {}
        self._lock = threading.Lock()
        self._recent: deque = deque(maxlen=self.RECENT_MAXLEN)

    @staticmethod
    def _is_multicast_or_broadcast(ip: str) -> bool:
        try:
            first_octet = int(ip.split(".")[0])
            return (224 <= first_octet <= 239) or ip == "255.255.255.255" or ip.endswith(".255")
        except (ValueError, IndexError):
            return False

    @staticmethod
    def _flow_key(ip_a, port_a, ip_b, port_b, proto) -> tuple:
        """Order-independent key — A→B and B→A hash to the same flow."""
        endpoints = sorted([(ip_a, port_a), (ip_b, port_b)])
        return (endpoints[0], endpoints[1], proto)

    def process_packet(self, packet, ts: float = None) -> None:
        if not SCAPY_AVAILABLE or IP not in packet:
            return

        now = ts if ts is not None else time.time()
        src_ip, dst_ip = packet[IP].src, packet[IP].dst

        if self._is_multicast_or_broadcast(dst_ip):
            return

        proto = "other"
        src_port = dst_port = 0
        ttl = packet[IP].ttl
        window = 0

        if TCP in packet:
            proto, src_port, dst_port = "tcp", packet[TCP].sport, packet[TCP].dport
            window = int(packet[TCP].window)
        elif UDP in packet:
            proto, src_port, dst_port = "udp", packet[UDP].sport, packet[UDP].dport
        elif ICMP in packet:
            proto = "icmp"

        key = self._flow_key(src_ip, src_port, dst_ip, dst_port, proto)
        pkt_len = len(packet)

        with self._lock:
            if key not in self._flows:
                svc = PORT_SERVICE_MAP.get(dst_port, PORT_SERVICE_MAP.get(src_port, "-"))
                self._flows[key] = FlowConnection(
                    orig_src_ip=src_ip, orig_dst_ip=dst_ip,
                    orig_src_port=src_port, orig_dst_port=dst_port,
                    protocol=proto, service=svc,
                    start_time=now, last_seen=now,
                )

            conn = self._flows[key]
            conn.last_seen = now
            is_forward = (src_ip == conn.orig_src_ip and src_port == conn.orig_src_port)

            if is_forward:
                conn.spkts += 1
                conn.sbytes += pkt_len
                conn.sttl = ttl
                if TCP in packet:
                    conn.swin = window
            else:
                conn.dpkts += 1
                conn.dbytes += pkt_len
                conn.dttl = ttl
                if TCP in packet:
                    conn.dwin = window

            if TCP in packet:
                flags = packet[TCP].flags
                if flags & 0x02: conn.syn_count += 1
                if flags & 0x01: conn.fin_count += 1
                if flags & 0x04: conn.rst_count += 1
                if flags & 0x10: conn.ack_count += 1

    def _finalise(self, conn: FlowConnection, now: float):
        conn.duration = max(conn.last_seen - conn.start_time, 0.0)
        conn.state = infer_state(conn)
        self._recent.append({
            "src_ip": conn.orig_src_ip, "dst_ip": conn.orig_dst_ip,
            "src_port": conn.orig_src_port, "dst_port": conn.orig_dst_port,
            "service": conn.service, "state": conn.state, "sttl": conn.sttl,
        })

    def flush_expired(self, now: float = None) -> list[FlowConnection]:
        now = now if now is not None else time.time()
        expired = []
        with self._lock:
            for key in list(self._flows):
                conn = self._flows[key]
                if now - conn.last_seen > self.TIME_WINDOW:
                    self._finalise(conn, now)
                    expired.append(conn)
                    del self._flows[key]
        return expired

    def flush_all(self, now: float = None) -> list[FlowConnection]:
        now = now if now is not None else time.time()
        finished = []
        with self._lock:
            for key in list(self._flows):
                conn = self._flows[key]
                self._finalise(conn, now)
                finished.append(conn)
                del self._flows[key]
        return finished

    def compute_ct_features(self, conn: FlowConnection) -> dict:
        """
        The 8 'ct_*' rolling-window count features UNSW-NB15 uses,
        approximated over the last 100 finalised connections — matching
        the original dataset's count-based (not time-based) windowing.
        """
        with self._lock:
            recent = list(self._recent)

        def count(pred):
            return sum(1 for r in recent if pred(r))

        return {
            "ct_srv_src":       count(lambda r: r["service"] == conn.service and r["src_ip"] == conn.orig_src_ip),
            "ct_srv_dst":       count(lambda r: r["service"] == conn.service and r["dst_ip"] == conn.orig_dst_ip),
            "ct_dst_ltm":       count(lambda r: r["dst_ip"] == conn.orig_dst_ip),
            "ct_src_ltm":       count(lambda r: r["src_ip"] == conn.orig_src_ip),
            "ct_src_dport_ltm": count(lambda r: r["src_ip"] == conn.orig_src_ip and r["dst_port"] == conn.orig_dst_port),
            "ct_dst_sport_ltm": count(lambda r: r["dst_ip"] == conn.orig_dst_ip and r["src_port"] == conn.orig_src_port),
            "ct_dst_src_ltm":   count(lambda r: r["src_ip"] == conn.orig_src_ip and r["dst_ip"] == conn.orig_dst_ip),
            "ct_state_ttl":     count(lambda r: r["state"] == conn.state and r["sttl"] == conn.sttl),
        }


class UNSWFeatureExtractor:
    """Converts a finalised FlowConnection into the UNSW-NB15 feature dict."""

    def __init__(self, tracker: FlowTracker):
        self.tracker = tracker

    def extract(self, conn: FlowConnection) -> dict:
        ct = self.tracker.compute_ct_features(conn)
        dur = max(conn.duration, 1e-6)  # avoid div/0 for near-instant flows

        rate  = (conn.spkts + conn.dpkts) / dur
        sload = (conn.sbytes * 8) / dur   # bits/sec
        dload = (conn.dbytes * 8) / dur
        smean = conn.sbytes / conn.spkts if conn.spkts > 0 else 0
        dmean = conn.dbytes / conn.dpkts if conn.dpkts > 0 else 0
        is_sm = 1 if (conn.orig_src_ip == conn.orig_dst_ip and
                      conn.orig_src_port == conn.orig_dst_port) else 0

        record = {
            "dur":    round(conn.duration, 4),
            "spkts":  conn.spkts,
            "dpkts":  conn.dpkts,
            "sbytes": conn.sbytes,
            "dbytes": conn.dbytes,
            "rate":   round(rate, 4),
            "sttl":   conn.sttl,
            "dttl":   conn.dttl,
            "sload":  round(sload, 2),
            "dload":  round(dload, 2),
            "swin":   conn.swin,
            "dwin":   conn.dwin,
            "smean":  round(smean, 2),
            "dmean":  round(dmean, 2),
            "is_sm_ips_ports": is_sm,
            "proto":   conn.protocol,
            "service": conn.service,
            "state":   conn.state,
            **ct,
        }

        # Engineered features — must match preprocess_unsw.py exactly
        record["bytes_ratio"]   = round(conn.sbytes / (conn.dbytes + 1), 4)
        record["pkts_ratio"]    = round(conn.spkts / (conn.dpkts + 1), 4)
        record["load_diff"]     = round(sload - dload, 2)
        record["bidirectional"] = 1 if (conn.sbytes > 0 and conn.dbytes > 0) else 0
        record["zero_bytes"]    = 1 if (conn.sbytes == 0 and conn.dbytes == 0) else 0

        return record


def format_flow_alert(alert: dict) -> str:
    """Pretty-print a UNSW-NB15 classification result, matching the
    NSL-KDD format_alert() style used elsewhere in SONA."""
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
        f"| {alert['sbytes']}B↑ {alert['dbytes']}B↓ | state={alert['state']}"
        f"{reset}"
    )
