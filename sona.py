"""
sona.py — SONA v2/v3 Main Runner
==================================
Connects Module 1 (LiveCapture) + Module 2 (ResponseEngine) together.
This is what you run to have SONA actually watching and defending your network.

Supports two models via --model-version:
  unsw    (default) — retrained on UNSW-NB15, 9 modern attack families,
                       bidirectional flow tracking. Recommended.
  nsl-kdd            — original 1999 dataset, kept for comparison.

Usage:
    # Dry run — no real firewall changes (safe for testing)
    python sona.py

    # Live mode — actually blocks IPs via Windows Firewall
    # Must run terminal as Administrator
    python sona.py --live --host-ip 192.168.31.211

    # Use the original NSL-KDD model instead
    python sona.py --live --host-ip 192.168.31.211 --model-version nsl-kdd
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from packet_capture import (
    list_interfaces, format_alert,
    find_interface_by_ip, print_interfaces_with_ips,
)
from response_engine import ResponseEngine


def parse_args():
    p = argparse.ArgumentParser(description="SONA — Security of Network Access")
    p.add_argument("--live",      action="store_true",
                   help="Enable real firewall blocking (requires admin)")
    p.add_argument("--model-version", choices=["unsw", "nsl-kdd"], default="unsw",
                   help="Which trained model to run live (default: unsw — "
                        "the modern, retrained model)")
    p.add_argument("--interface", type=str, default=None,
                   help="Network interface to capture on (e.g. Wi-Fi, eth0). "
                        "On Windows, prefer --host-ip instead — it's more reliable.")
    p.add_argument("--host-ip", type=str, default=None,
                   help="Your machine's LAN IP (e.g. 192.168.31.211). SONA will "
                        "auto-resolve the exact interface Scapy needs — this avoids "
                        "the common Windows bug where a friendly name like 'Wi-Fi' "
                        "silently attaches to the wrong adapter.")
    p.add_argument("--allowlist", nargs="*", default=[],
                   help="Extra IPs to never block (e.g. 192.168.1.1)")
    p.add_argument("--test-target", nargs="*", default=[],
                   help="IPs to treat as blockable even on your private LAN "
                        "(e.g. your Kali VM's bridged IP) — for controlled demos only")
    p.add_argument("--verbose", action="store_true",
                   help="Print EVERY classified connection (including Normal) — "
                        "use this to debug why expected attacks aren't showing up")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print("  SONA — Security of Network Access")
    print(f"  Model: {args.model_version.upper()}")
    print("  Module 1: Live Packet Capture")
    print("  Module 2: Automated Response Engine")
    print("=" * 65)

    dry_run = not args.live
    if dry_run:
        print("\n  [DRY RUN] Firewall rules will NOT be applied.")
        print("  Run with --live flag (as Administrator) for real blocking.\n")
    else:
        print("\n  [LIVE MODE] Firewall rules WILL be applied.\n")

    # Resolve the correct interface — IP-based matching is far more
    # reliable on Windows than typing a friendly name like "Wi-Fi"
    interface = args.interface
    if args.host_ip:
        resolved = find_interface_by_ip(args.host_ip)
        if resolved:
            print(f"  Resolved --host-ip {args.host_ip} → interface: {resolved!r}\n")
            interface = resolved
        else:
            print(f"  [WARN] Could not find an interface with IP {args.host_ip}.")
            print(f"         Showing all detected interfaces + IPs below:\n")
            print_interfaces_with_ips()
            sys.exit(1)
    elif interface is None:
        print("Available interfaces:")
        print_interfaces_with_ips()
        print()

    # Pick the right model + capture engine
    if args.model_version == "unsw":
        from packet_capture_unsw import UNSWLiveCapture
        model_dir = os.path.join(os.path.dirname(__file__), "models_unsw")
        CaptureEngine = UNSWLiveCapture
    else:
        from packet_capture import LiveCapture
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        CaptureEngine = LiveCapture

    if not os.path.exists(os.path.join(model_dir, "best_model.pkl")):
        train_script = "train_unsw.py" if args.model_version == "unsw" else "train.py"
        print(f"[ERROR] No trained model found in {model_dir}/. Run python src/{train_script} first.")
        sys.exit(1)

    # Initialise Module 1
    capture = CaptureEngine(model_dir=model_dir)

    # Initialise Module 2
    response = ResponseEngine(
        log_dir="logs",
        allowlist_ips=args.allowlist or [],
        test_target_ips=args.test_target or [],
        dry_run=dry_run,
    )

    print(f"\n{'─'*65}")
    print(f"  Starting capture on: {interface or 'default interface'}")
    print(f"  Press Ctrl+C to stop and see session summary")
    print(f"{'─'*65}\n")

    capture.start(interface=interface)

    # Counters for session summary
    total = 0
    threats = 0
    blocked = 0

    try:
        for alert in capture.alerts():
            total += 1
            if alert["is_attack"]:
                threats += 1
                event = response.process(alert)
                if event and event.action in ("BLOCKED", "BLOCKED(DRY_RUN)"):
                    blocked += 1
                if args.verbose:
                    print(format_alert(alert))
            elif args.verbose:
                # Print Normal classifications too, so you can see exactly
                # what SONA is predicting for every connection — useful for
                # debugging why an expected attack isn't showing up.
                print(format_alert(alert))

    except KeyboardInterrupt:
        print("\n\nStopping SONA...")
        capture.stop()
        response.stop()

        # Final session summary
        stats = response.logger.get_stats()
        print(f"\n{'═'*65}")
        print(f"  SONA SESSION SUMMARY")
        print(f"{'═'*65}")
        print(f"  Connections analysed  : {total}")
        print(f"  Threats detected      : {threats}")
        print(f"  IPs blocked           : {blocked}")
        print(f"\n  Attack breakdown:")
        for atype, cnt in stats.get("by_type", {}).items():
            if atype != "Normal":
                print(f"    {atype:12s}: {cnt}")
        bips = response.firewall.blocked_ips()
        if bips:
            print(f"\n  Blocked IPs ({len(bips)}):")
            for b in bips:
                print(f"    {b.ip:20s} | {b.attack_type}")
        print(f"\n  Full logs: logs/threat_log.json | logs/threat_log.csv")
        print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()
