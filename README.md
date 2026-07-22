# SONA — Security of Network Access

An ML-powered Network Intrusion Detection & Prevention System that watches
live network traffic, classifies it against real attack patterns, and
automatically responds — blocking malicious IPs, logging every decision,
and surfacing it all on a real-time SOC-style dashboard.

Built from scratch: no Snort/Suricata under the hood — the packet capture,
flow reconstruction, feature engineering, ML classification, automated
response, and dashboard are all custom-built.

---

## Why this exists

Most "ML intrusion detection" student projects stop at: load a CSV, train
a model, report accuracy. SONA goes further — it's a full pipeline that
watches **real live packets**, extracts network-flow features on the fly,
classifies them with a trained model, and **takes action**: blocking
attacker IPs at the OS firewall level, the same category of behavior as a
real IPS (Intrusion Prevention System).

## Architecture

```
 Live packets (Scapy)
        │
        ▼
 Connection/Flow Tracker  ──► groups packets into connections (NSL-KDD)
                               or bidirectional flows (UNSW-NB15)
        │
        ▼
 Feature Extractor  ──► reconstructs the same features the model trained on
        │
        ▼
 ML Classifier (XGBoost/Random Forest)  ──► Normal or an attack category
        │
        ▼
 Response Engine  ──► allowlist check → confidence gate → strike system
        │              → firewall block (netsh/iptables) → audit log
        ▼
 Live SOC Dashboard (Streamlit)  ──► reads the logs in real time
```

## Two trained models — an honest evolution story

SONA started on **NSL-KDD** (1999) — the classic intrusion-detection
benchmark. Testing it against real live traffic and a real downloaded
attack pcap exposed a genuine problem: a 1999-era dataset doesn't
generalize to modern network patterns (it correctly classified attacks
in synthetic tests, but called real attack traffic "Normal").

That finding motivated a full retraining on **UNSW-NB15** (2015) — a
modern dataset covering 9 real attack families (Fuzzers, Analysis,
Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms).
This required more than swapping a CSV: UNSW-NB15 describes traffic as
**bidirectional flows**, so the live packet-capture engine was rebuilt
around a proper flow tracker (`src/flow_capture.py`) instead of the
original directional connection tracker.

| | NSL-KDD (v1) | UNSW-NB15 (v2) |
|---|---|---|
| Year | 1999 | 2015 |
| Classes | 5 (Normal, DoS, Probe, R2L, U2R) | 10 (Normal + 9 modern families) |
| Best model | XGBoost | Random Forest (SMOTE-balanced) |
| Weighted F1 | — | 0.750 |
| Macro F1 | — | 0.502 |
| Live traffic model | Directional connections | Bidirectional flows |

Class imbalance (Generic/Normal dominate; Worms/Analysis are tiny) was
addressed with SMOTE oversampling, and model selection uses macro-F1
specifically so a model isn't rewarded just for being right about the
common classes.

**Known limitation:** the "Analysis" attack category remains hard to
detect (near-zero precision) — this matches published UNSW-NB15
benchmarks, where Analysis overlaps heavily with Reconnaissance/Fuzzers
in feature space. Documented, not hidden.

## What each module does

**Module 1 — Live packet capture** (`src/packet_capture.py`,
`src/flow_capture.py`) — sniffs real traffic via Scapy, reconstructs
connections/flows, computes NSL-KDD or UNSW-NB15 features on the fly.

**Module 2 — Automated response engine** (`src/response_engine.py`) —
on a detected attack: checks an allowlist (never blocks your own LAN),
gates on confidence, applies a "strike" system (soft attack types need
repeated hits before blocking), then blocks the IP via Windows Firewall
(`netsh`) or iptables, with automatic timed unblocking and a full JSON/CSV
audit trail.

**Module 3 — Live SOC dashboard** (`dashboard/live.py`) — a Streamlit app
that reads the response engine's logs in real time: KPIs, a live threat
feed, attack-category breakdowns, a blocked-IP table with one-click
unblock, and a GeoIP map of public attacker IPs. Runs as a separate
process from the sensor — the same architecture real SIEM tools use.

**Offline PCAP analysis** (`src/pcap_analysis.py`) — replays a `.pcap`
file through the exact same feature-extraction + classification pipeline
as live capture, using each packet's own timestamp. Useful for analyzing
captures from anywhere (Wireshark, a public attack-traffic archive) —
and can feed results straight into the live dashboard with
`--feed-dashboard`.

## Known limitations (documented deliberately)

- **VirtualBox Bridged networking over WiFi is unreliable** for live-attack
  demos — 802.11 doesn't support arbitrary MAC bridging the way Ethernet
  does. Use a VirtualBox Host-Only network instead, or use the offline
  PCAP analyzer, which sidesteps this entirely.
- **Live capture, the response engine, and firewall control are local-only
  by design.** They are not (and should not be) deployed to a public
  multi-tenant host — that would let strangers trigger firewall changes on
  someone's machine. Only the dashboard's read-only demo mode is safe to
  deploy publicly.
- The `state` feature for UNSW-NB15 live capture is a best-effort
  approximation of what Argus (the tool the dataset was built with)
  computes — exact reproduction isn't possible from raw packets alone.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download a dataset:
- NSL-KDD: place `KDDTrain+.txt` / `KDDTest+.txt` in `data/`
- UNSW-NB15: place `UNSW_NB15_training-set.csv` / `-testing-set.csv` in `data/unsw/`

Train:
```bash
python src/train.py          # NSL-KDD
python src/train_unsw.py     # UNSW-NB15 (recommended)
```

Run live (requires Administrator/root):
```bash
python sona.py --live --host-ip <your-LAN-IP> --model-version unsw
```

Or analyze a pcap file (no admin rights, no live networking needed):
```bash
python src/pcap_analysis.py --file capture.pcap --feed-dashboard
```

View the dashboard:
```bash
streamlit run dashboard/live.py
```

## Tech stack

Python · Scapy · Scikit-learn · XGBoost · imbalanced-learn (SMOTE) ·
Streamlit · Plotly · Windows Firewall / iptables

## What I'd build next

- MITRE ATT&CK framework mapping for each detected attack category
- A proper flow-pairing rewrite for the NSL-KDD tracker (currently
  directional; UNSW-NB15's tracker already does this correctly)
- Deep-packet-inspection features (TCP jitter, precise RTT) for a closer
  match to UNSW-NB15's original feature set
