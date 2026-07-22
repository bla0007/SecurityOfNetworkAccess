"""
response_engine.py — SONA v2 Module 2
=======================================
Automated threat response engine.

What this does when an attack is detected:
  1. Checks the IP against a safe allowlist (never block your own network)
  2. Applies a confidence threshold — only act on high-confidence detections
  3. Applies a strike system — block only after N confirmed attacks from same IP
  4. Blocks the attacker's IP via Windows Firewall (netsh) or Linux iptables
  5. Logs every action to threat_log.json and threat_log.csv
  6. Can automatically unblock IPs after a cooldown period
  7. Exposes a full audit trail of every decision made

Run standalone to test without live capture:
    python src/response_engine.py

Architecture:
    ResponseEngine
        ├── AllowlistManager   — never block safe IPs
        ├── ThreatScorer       — strike counting + confidence gating
        ├── FirewallManager    — OS-level block/unblock
        └── ThreatLogger       — persistent JSON + CSV audit log
"""

import os
import json
import csv
import time
import platform
import subprocess
import threading
import ipaddress
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional


# ── Configuration ─────────────────────────────────────────────────────────────

class Config:
    # Only act on detections above this confidence
    CONFIDENCE_THRESHOLD = 0.75

    # Number of confirmed attacks from same IP before blocking
    STRIKE_THRESHOLD = 2

    # Window (seconds) in which strikes are counted
    STRIKE_WINDOW = 60

    # Automatically unblock IPs after this many seconds (0 = never auto-unblock)
    AUTO_UNBLOCK_AFTER = 300   # 5 minutes

    # Log file paths (relative to project root)
    LOG_JSON = "logs/threat_log.json"
    LOG_CSV  = "logs/threat_log.csv"

    # Attack types that trigger an immediate block (even on first strike)
    INSTANT_BLOCK_TYPES = {"DoS", "U2R"}

    # Attack types that need STRIKE_THRESHOLD strikes before blocking
    STRIKE_BLOCK_TYPES = {"Probe", "R2L"}

    # Attack types that are logged but never auto-blocked
    LOG_ONLY_TYPES = set()


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ThreatEvent:
    """A single classified alert that triggered a response decision."""
    timestamp:   str
    src_ip:      str
    dst_ip:      str
    src_port:    int
    dst_port:    int
    protocol:    str
    service:     str
    attack_type: str
    confidence:  float
    flag:        str
    src_bytes:   int
    action:      str        # "BLOCKED", "LOGGED", "ALLOWLISTED", "BELOW_THRESHOLD", "STRIKE_N"
    rule_name:   str = ""   # Windows Firewall rule name if blocked
    strike:      int = 0    # Which strike number this was


@dataclass
class BlockedIP:
    """Tracks a currently-blocked IP."""
    ip:           str
    blocked_at:   str
    attack_type:  str
    confidence:   float
    rule_name:    str
    auto_unblock: bool
    unblock_at:   Optional[str] = None


# ── Allowlist Manager ─────────────────────────────────────────────────────────

class AllowlistManager:
    """
    Never block IPs on this list.
    Pre-populated with:
      - RFC1918 private ranges (your LAN — blocking these would lock you out)
      - Loopback
      - Common DNS servers
    You can add your own IPs via add().
    """

    _SAFE_RANGES = [
        "10.0.0.0/8",       # Private LAN
        "172.16.0.0/12",    # Private LAN
        "192.168.0.0/16",   # Private LAN (your home/office network)
        "127.0.0.0/8",      # Loopback
        "169.254.0.0/16",   # Link-local
        "::1/128",           # IPv6 loopback
        "fe80::/10",         # IPv6 link-local
    ]

    _SAFE_IPS = {
        "8.8.8.8",          # Google DNS
        "8.8.4.4",          # Google DNS
        "1.1.1.1",          # Cloudflare DNS
        "1.0.0.1",          # Cloudflare DNS
    }

    def __init__(self, extra_ips: list[str] = None, test_target_ips: list[str] = None):
        self._networks = []
        for cidr in self._SAFE_RANGES:
            try:
                self._networks.append(ipaddress.ip_network(cidr, strict=False))
            except ValueError:
                pass
        self._ips = set(self._SAFE_IPS)
        if extra_ips:
            for ip in extra_ips:
                self.add(ip)

        # IPs that should NEVER be treated as safe, even if they're on a
        # private range — used to demo real blocking against a VM/test host
        # on your own LAN without weakening protection for the rest of it.
        self._test_targets = set(test_target_ips or [])
        for ip in self._test_targets:
            print(f"[ALLOWLIST] Test target (blockable despite private range): {ip}")

    def add(self, ip: str):
        """Add an IP or CIDR to the allowlist."""
        try:
            if "/" in ip:
                self._networks.append(ipaddress.ip_network(ip, strict=False))
            else:
                self._ips.add(ip)
            print(f"[ALLOWLIST] Added: {ip}")
        except ValueError:
            print(f"[ALLOWLIST] Invalid IP/CIDR: {ip}")

    def is_safe(self, ip: str) -> bool:
        """Return True if this IP must never be blocked."""
        if ip in self._test_targets:
            return False   # explicit override — always treat as blockable
        if ip in self._ips:
            return True
        try:
            addr = ipaddress.ip_address(ip)
            return any(addr in net for net in self._networks)
        except ValueError:
            return False


# ── Threat Scorer (Strike System) ─────────────────────────────────────────────

class ThreatScorer:
    """
    Tracks strike counts per source IP within a rolling time window.
    Prevents false-positive blocks from single misclassifications.
    """

    def __init__(self):
        # ip → list of (timestamp, attack_type, confidence)
        self._strikes: dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()

    def record(self, ip: str, attack_type: str, confidence: float) -> int:
        """
        Record a strike for this IP. Returns current strike count
        within the window.
        """
        now = time.time()
        cutoff = now - Config.STRIKE_WINDOW

        with self._lock:
            # Remove old strikes outside the window
            self._strikes[ip] = [
                s for s in self._strikes[ip] if s[0] >= cutoff
            ]
            self._strikes[ip].append((now, attack_type, confidence))
            return len(self._strikes[ip])

    def get_strike_count(self, ip: str) -> int:
        now = time.time()
        cutoff = now - Config.STRIKE_WINDOW
        with self._lock:
            return sum(1 for s in self._strikes[ip] if s[0] >= cutoff)

    def clear(self, ip: str):
        with self._lock:
            self._strikes.pop(ip, None)


# ── Firewall Manager ──────────────────────────────────────────────────────────

class FirewallManager:
    """
    Blocks/unblocks IPs using the OS firewall.
    Windows: netsh advfirewall (no extra tools needed)
    Linux:   iptables (needs root)
    """

    OS = platform.system()   # "Windows" or "Linux"

    def __init__(self, state_file: str = "logs/blocked_ips.json"):
        self._blocked: dict[str, BlockedIP] = {}
        self._lock = threading.Lock()
        self._rule_prefix = "SONA_BLOCK_"
        self._state_file = state_file
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        self._load_state()

    def _load_state(self):
        """Load blocked IP state from disk (survives restarts, visible to dashboard)."""
        if os.path.exists(self._state_file):
            try:
                with open(self._state_file) as f:
                    data = json.load(f)
                for ip, d in data.items():
                    self._blocked[ip] = BlockedIP(**d)
            except Exception:
                pass

    def _save_state(self):
        """Persist blocked IP state to disk after every change."""
        with self._lock:
            data = {ip: asdict(b) for ip, b in self._blocked.items()}
        with open(self._state_file, "w") as f:
            json.dump(data, f, indent=2)

    def block(self, ip: str, attack_type: str, confidence: float) -> tuple[bool, str]:
        """
        Add a firewall rule blocking all inbound traffic from ip.
        Returns (success, rule_name).
        """
        rule_name = f"{self._rule_prefix}{ip.replace('.', '_')}"

        with self._lock:
            if ip in self._blocked:
                return True, rule_name  # Already blocked

        try:
            if self.OS == "Windows":
                success = self._block_windows(ip, rule_name)
            else:
                success = self._block_linux(ip, rule_name)

            if success:
                unblock_at = None
                if Config.AUTO_UNBLOCK_AFTER > 0:
                    unblock_at = (
                        datetime.now() + timedelta(seconds=Config.AUTO_UNBLOCK_AFTER)
                    ).strftime("%Y-%m-%d %H:%M:%S")

                with self._lock:
                    self._blocked[ip] = BlockedIP(
                        ip=ip,
                        blocked_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        attack_type=attack_type,
                        confidence=confidence,
                        rule_name=rule_name,
                        auto_unblock=Config.AUTO_UNBLOCK_AFTER > 0,
                        unblock_at=unblock_at,
                    )
                self._save_state()
                return True, rule_name
            return False, ""

        except Exception as e:
            print(f"[FIREWALL] Error blocking {ip}: {e}")
            return False, ""

    def unblock(self, ip: str) -> bool:
        """Remove the firewall rule for this IP."""
        with self._lock:
            blocked = self._blocked.get(ip)
            if not blocked:
                return False
            rule_name = blocked.rule_name

        try:
            if self.OS == "Windows":
                success = self._unblock_windows(rule_name)
            else:
                success = self._unblock_linux(ip)

            if success:
                with self._lock:
                    self._blocked.pop(ip, None)
                self._save_state()
                print(f"[FIREWALL] Unblocked: {ip}")
            return success
        except Exception as e:
            print(f"[FIREWALL] Error unblocking {ip}: {e}")
            return False

    def _block_windows(self, ip: str, rule_name: str) -> bool:
        """Block IP using Windows Firewall via netsh."""
        cmd = [
            "netsh", "advfirewall", "firewall", "add", "rule",
            f"name={rule_name}",
            "dir=in",
            "action=block",
            f"remoteip={ip}",
            "enable=yes",
            "description=SONA automated block",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0

    def _unblock_windows(self, rule_name: str) -> bool:
        cmd = [
            "netsh", "advfirewall", "firewall", "delete", "rule",
            f"name={rule_name}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0

    def _block_linux(self, ip: str, rule_name: str) -> bool:
        """Block IP using iptables on Linux."""
        cmd = ["iptables", "-I", "INPUT", "-s", ip, "-j", "DROP"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0

    def _unblock_linux(self, ip: str) -> bool:
        cmd = ["iptables", "-D", "INPUT", "-s", ip, "-j", "DROP"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0

    def is_blocked(self, ip: str) -> bool:
        with self._lock:
            return ip in self._blocked

    def blocked_ips(self) -> list[BlockedIP]:
        with self._lock:
            return list(self._blocked.values())

    def check_auto_unblock(self):
        """Called periodically — unblock IPs whose cooldown has expired."""
        now = datetime.now()
        with self._lock:
            to_unblock = [
                b for b in self._blocked.values()
                if b.auto_unblock and b.unblock_at and
                datetime.strptime(b.unblock_at, "%Y-%m-%d %H:%M:%S") <= now
            ]
        for b in to_unblock:
            print(f"[FIREWALL] Auto-unblocking {b.ip} (cooldown expired)")
            self.unblock(b.ip)


# ── Threat Logger ─────────────────────────────────────────────────────────────

class ThreatLogger:
    """
    Writes every threat event to:
      logs/threat_log.json  — full detail, machine readable
      logs/threat_log.csv   — spreadsheet-friendly for reporting
    """

    CSV_FIELDS = [
        "timestamp", "src_ip", "dst_ip", "src_port", "dst_port",
        "protocol", "service", "attack_type", "confidence",
        "flag", "src_bytes", "action", "rule_name", "strike",
    ]

    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        self._json_path = os.path.join(log_dir, "threat_log.json")
        self._csv_path  = os.path.join(log_dir, "threat_log.csv")
        self._lock = threading.Lock()
        self._events: list[dict] = []

        # Load existing events
        if os.path.exists(self._json_path):
            try:
                with open(self._json_path) as f:
                    self._events = json.load(f)
            except Exception:
                self._events = []

        # Ensure CSV has headers
        if not os.path.exists(self._csv_path):
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
                writer.writeheader()

    def log(self, event: ThreatEvent):
        """Write a ThreatEvent to both log files."""
        row = asdict(event)
        with self._lock:
            self._events.append(row)
            # JSON — rewrite full file (simple, fine for demo scale)
            with open(self._json_path, "w") as f:
                json.dump(self._events, f, indent=2)
            # CSV — append row
            with open(self._csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
                writer.writerow({k: row.get(k, "") for k in self.CSV_FIELDS})

    def get_events(self, since_minutes: int = 60) -> list[dict]:
        """Return events from the last N minutes."""
        cutoff = datetime.now() - timedelta(minutes=since_minutes)
        with self._lock:
            return [
                e for e in self._events
                if datetime.strptime(e["timestamp"], "%Y-%m-%d %H:%M:%S") >= cutoff
            ]

    def get_stats(self) -> dict:
        """Summary stats for the dashboard."""
        with self._lock:
            events = list(self._events)
        total   = len(events)
        blocked = sum(1 for e in events if e["action"] == "BLOCKED")
        by_type = defaultdict(int)
        for e in events:
            by_type[e["attack_type"]] += 1
        top_ips = defaultdict(int)
        for e in events:
            if e["action"] == "BLOCKED":
                top_ips[e["src_ip"]] += 1
        return {
            "total_events":  total,
            "total_blocked": blocked,
            "by_type":       dict(by_type),
            "top_attacker_ips": dict(sorted(top_ips.items(), key=lambda x: -x[1])[:10]),
        }


# ── Response Engine (ties it all together) ────────────────────────────────────

class ResponseEngine:
    """
    The main decision-maker.

    Call process(alert) with every alert from LiveCapture.
    It will:
      1. Skip normal traffic silently
      2. Skip allowlisted IPs with a log entry
      3. Skip low-confidence detections
      4. Apply strike counting for Probe/R2L
      5. Block immediately for DoS/U2R
      6. Call FirewallManager to enforce the block
      7. Log everything to ThreatLogger
    """

    def __init__(
        self,
        log_dir:        str = "logs",
        allowlist_ips:  list[str] = None,
        test_target_ips: list[str] = None,
        dry_run:        bool = False,
    ):
        """
        dry_run=True: log decisions but don't actually change firewall rules.
        Use dry_run=True for testing without admin rights.

        test_target_ips: IPs to treat as blockable even if they're on a
        private LAN range (e.g. your Kali VM's bridged IP). Use this ONLY
        for controlled demos against a machine you own — never add your
        router or other real devices here.
        """
        self.allowlist = AllowlistManager(extra_ips=allowlist_ips, test_target_ips=test_target_ips)
        self.scorer    = ThreatScorer()
        self.firewall  = FirewallManager()
        self.logger    = ThreatLogger(log_dir=log_dir)
        self.dry_run   = dry_run
        self._lock     = threading.Lock()

        # Start auto-unblock background thread
        self._running = True
        t = threading.Thread(target=self._auto_unblock_loop, daemon=True)
        t.start()

        mode = "DRY RUN (no real firewall changes)" if dry_run else "LIVE (firewall active)"
        print(f"[RESPONSE] Engine started — mode: {mode}")
        print(f"[RESPONSE] Confidence threshold : {Config.CONFIDENCE_THRESHOLD:.0%}")
        print(f"[RESPONSE] Strike threshold     : {Config.STRIKE_THRESHOLD} per {Config.STRIKE_WINDOW}s")
        print(f"[RESPONSE] Auto-unblock after   : {Config.AUTO_UNBLOCK_AFTER}s")
        print(f"[RESPONSE] Instant block types  : {Config.INSTANT_BLOCK_TYPES}")

    def process(self, alert: dict) -> Optional[ThreatEvent]:
        """
        Process one alert from LiveCapture.
        Returns a ThreatEvent if action was taken, None for normal traffic.
        """
        # 1. Skip normal traffic — no action needed
        if not alert.get("is_attack", False):
            return None

        src_ip      = alert["src_ip"]
        attack_type = alert["prediction"]
        confidence  = alert["confidence"]

        # Base event (will be updated with action before logging)
        def make_event(action, rule_name="", strike=0) -> ThreatEvent:
            return ThreatEvent(
                timestamp   = alert["timestamp"],
                src_ip      = src_ip,
                dst_ip      = alert["dst_ip"],
                src_port    = alert["src_port"],
                dst_port    = alert["dst_port"],
                protocol    = alert["protocol"],
                service     = alert["service"],
                attack_type = attack_type,
                confidence  = confidence,
                flag        = alert["flag"],
                src_bytes   = alert["src_bytes"],
                action      = action,
                rule_name   = rule_name,
                strike      = strike,
            )

        # 2. Allowlist check — never block safe IPs
        if self.allowlist.is_safe(src_ip):
            event = make_event("ALLOWLISTED")
            self.logger.log(event)
            self._print_decision(event)
            return event

        # 3. Confidence gate — don't act on uncertain predictions
        if confidence < Config.CONFIDENCE_THRESHOLD:
            event = make_event("BELOW_THRESHOLD")
            self.logger.log(event)
            self._print_decision(event)
            return event

        # 4. Strike counting for softer attack types
        if attack_type in Config.STRIKE_BLOCK_TYPES:
            strike = self.scorer.record(src_ip, attack_type, confidence)
            if strike < Config.STRIKE_THRESHOLD:
                event = make_event(f"STRIKE_{strike}", strike=strike)
                self.logger.log(event)
                self._print_decision(event)
                return event

        # 5. Block decision
        if attack_type in Config.LOG_ONLY_TYPES:
            event = make_event("LOGGED")
            self.logger.log(event)
            self._print_decision(event)
            return event

        # 6. Execute firewall block
        strike = self.scorer.get_strike_count(src_ip)
        rule_name = ""

        if not self.dry_run and not self.firewall.is_blocked(src_ip):
            success, rule_name = self.firewall.block(src_ip, attack_type, confidence)
            action = "BLOCKED" if success else "BLOCK_FAILED"
        elif self.dry_run:
            action    = "BLOCKED(DRY_RUN)"
            rule_name = f"SONA_BLOCK_{src_ip.replace('.', '_')}"
        else:
            action    = "ALREADY_BLOCKED"
            rule_name = f"SONA_BLOCK_{src_ip.replace('.', '_')}"

        event = make_event(action, rule_name=rule_name, strike=strike)
        self.logger.log(event)
        self._print_decision(event)
        return event

    def _print_decision(self, event: ThreatEvent):
        """Colour-coded terminal output for every response decision."""
        icons = {
            "BLOCKED":       ("🔴", "\033[91m"),
            "BLOCKED(DRY_RUN)": ("🔴", "\033[91m"),
            "ALREADY_BLOCKED":  ("🔒", "\033[91m"),
            "BLOCK_FAILED":  ("❌", "\033[93m"),
            "ALLOWLISTED":   ("🟢", "\033[92m"),
            "BELOW_THRESHOLD":  ("⚪", "\033[37m"),
            "LOGGED":        ("📋", "\033[94m"),
        }
        strike_icon = lambda a: ("⚡", "\033[93m") if a.startswith("STRIKE") else icons.get(a, ("❓", "\033[0m"))
        icon, color = strike_icon(event.action)
        reset = "\033[0m"

        print(
            f"{color}{icon} [{event.timestamp}] "
            f"{event.src_ip} → {event.dst_ip}:{event.dst_port} "
            f"| {event.attack_type.upper()} ({event.confidence:.0%}) "
            f"| Action: {event.action}"
            + (f" | Rule: {event.rule_name}" if event.rule_name else "")
            + f"{reset}"
        )

    def _auto_unblock_loop(self):
        """Background thread — check for expired blocks every 30 seconds."""
        while self._running:
            time.sleep(30)
            self.firewall.check_auto_unblock()

    def stop(self):
        self._running = False

    def status(self) -> dict:
        """Return current engine status for the dashboard."""
        return {
            "blocked_ips": [asdict(b) for b in self.firewall.blocked_ips()],
            "stats":       self.logger.get_stats(),
            "config": {
                "confidence_threshold": Config.CONFIDENCE_THRESHOLD,
                "strike_threshold":     Config.STRIKE_THRESHOLD,
                "auto_unblock_after":   Config.AUTO_UNBLOCK_AFTER,
                "dry_run":              self.dry_run,
            }
        }

    def manual_unblock(self, ip: str) -> bool:
        """Manually unblock an IP (called from dashboard)."""
        self.scorer.clear(ip)
        return self.firewall.unblock(ip)


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    print("=" * 65)
    print("  SONA v2 — Module 2: Automated Response Engine Test")
    print("=" * 65)
    print()

    # Use dry_run=True so we test logic without touching real firewall rules
    engine = ResponseEngine(
        log_dir="logs",
        dry_run=True,   # ← change to False when running as Administrator
    )

    print(f"\n{'─'*65}")
    print("  Simulating attack scenarios...\n")

    # Simulated alerts that Module 1 would produce
    test_alerts = [
        # Normal — should be silently ignored
        {"is_attack": False,  "src_ip": "192.168.1.5",  "prediction": "Normal",
         "confidence": 0.99, "timestamp": "2026-07-16 15:00:01",
         "dst_ip": "8.8.8.8", "src_port": 52341, "dst_port": 443,
         "protocol": "tcp", "service": "http_443", "flag": "SF", "src_bytes": 450},

        # DoS with high confidence — instant block
        {"is_attack": True,   "src_ip": "203.0.113.10", "prediction": "DoS",
         "confidence": 0.97, "timestamp": "2026-07-16 15:00:03",
         "dst_ip": "192.168.1.1", "src_port": 44123, "dst_port": 80,
         "protocol": "tcp", "service": "http", "flag": "S0", "src_bytes": 0},

        # Probe — strike 1 (not blocked yet)
        {"is_attack": True,   "src_ip": "198.51.100.7",  "prediction": "Probe",
         "confidence": 0.88, "timestamp": "2026-07-16 15:00:05",
         "dst_ip": "192.168.1.1", "src_port": 11000, "dst_port": 22,
         "protocol": "tcp", "service": "ssh", "flag": "RSTR", "src_bytes": 44},

        # Low confidence — should be skipped
        {"is_attack": True,   "src_ip": "198.51.100.99", "prediction": "DoS",
         "confidence": 0.55, "timestamp": "2026-07-16 15:00:07",
         "dst_ip": "192.168.1.1", "src_port": 55000, "dst_port": 80,
         "protocol": "tcp", "service": "http", "flag": "S0", "src_bytes": 0},

        # Private IP — should be allowlisted (never blocked)
        {"is_attack": True,   "src_ip": "192.168.1.99",  "prediction": "Probe",
         "confidence": 0.92, "timestamp": "2026-07-16 15:00:09",
         "dst_ip": "192.168.1.1", "src_port": 33000, "dst_port": 445,
         "protocol": "tcp", "service": "microsoft_ds", "flag": "S1", "src_bytes": 60},

        # Probe — strike 2 — now blocked
        {"is_attack": True,   "src_ip": "198.51.100.7",  "prediction": "Probe",
         "confidence": 0.91, "timestamp": "2026-07-16 15:00:11",
         "dst_ip": "192.168.1.1", "src_port": 11001, "dst_port": 23,
         "protocol": "tcp", "service": "telnet", "flag": "RSTR", "src_bytes": 40},

        # R2L — strike 1
        {"is_attack": True,   "src_ip": "203.0.113.55",  "prediction": "R2L",
         "confidence": 0.83, "timestamp": "2026-07-16 15:00:13",
         "dst_ip": "192.168.1.1", "src_port": 60000, "dst_port": 22,
         "protocol": "tcp", "service": "ssh", "flag": "SF", "src_bytes": 1200},

        # U2R — instant block (privilege escalation)
        {"is_attack": True,   "src_ip": "203.0.113.77",  "prediction": "U2R",
         "confidence": 0.79, "timestamp": "2026-07-16 15:00:15",
         "dst_ip": "192.168.1.1", "src_port": 48000, "dst_port": 80,
         "protocol": "tcp", "service": "http", "flag": "SF", "src_bytes": 5000},

        # DoS again from same IP — already blocked
        {"is_attack": True,   "src_ip": "203.0.113.10",  "prediction": "DoS",
         "confidence": 0.95, "timestamp": "2026-07-16 15:00:17",
         "dst_ip": "192.168.1.1", "src_port": 44200, "dst_port": 80,
         "protocol": "tcp", "service": "http", "flag": "S0", "src_bytes": 0},
    ]

    for alert in test_alerts:
        engine.process(alert)
        time.sleep(0.3)

    # Summary
    status = engine.status()
    stats  = status["stats"]
    print(f"\n{'═'*65}")
    print("  RESPONSE ENGINE TEST COMPLETE")
    print(f"{'═'*65}")
    print(f"  Total threat events logged : {stats['total_events']}")
    print(f"  IPs blocked                : {stats['total_blocked']}")
    print(f"\n  Breakdown by attack type:")
    for atype, cnt in stats["by_type"].items():
        print(f"    {atype:12s}: {cnt}")
    print(f"\n  Currently blocked IPs:")
    for b in status["blocked_ips"]:
        print(f"    {b['ip']:20s} | {b['attack_type']:8s} | "
              f"blocked at {b['blocked_at']}"
              + (f" | unblock at {b['unblock_at']}" if b['unblock_at'] else ""))
    print(f"\n  Logs written to:")
    print(f"    logs/threat_log.json")
    print(f"    logs/threat_log.csv")
    print(f"\n  To run with real firewall blocking:")
    print(f"    1. Run terminal as Administrator")
    print(f"    2. Change dry_run=False in this file")
    print(f"    3. python src/response_engine.py\n")

    engine.stop()
