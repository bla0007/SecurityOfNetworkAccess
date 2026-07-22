"""
live.py — SONA v2 Module 3: Live SOC Dashboard
================================================
A real-time Security Operations Center style dashboard.

Unlike the original app.py (which used a static dataset for one-off
predictions), this dashboard reads the LIVE threat log that sona.py
writes while it's running, and auto-refreshes every few seconds.

How it works:
  - sona.py (Module 1 + 2) runs in one terminal, watching real traffic
    and writing every decision to logs/threat_log.json/csv and
    logs/blocked_ips.json
  - This dashboard runs in a separate terminal/browser tab and reads
    those files every 2 seconds — exactly how a real SIEM frontend
    reads from a shared log store, decoupled from the sensor process.

Run in a SEPARATE terminal from sona.py:
    streamlit run dashboard/live.py

Prerequisite:
    pip install streamlit-autorefresh requests
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import time
from datetime import datetime, timedelta
from collections import Counter, defaultdict

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ── Paths ──────────────────────────────────────────────────────────────────

PROJECT_ROOT   = os.path.join(os.path.dirname(__file__), "..")
LOG_JSON_PATH  = os.path.join(PROJECT_ROOT, "logs", "threat_log.json")
BLOCKED_PATH   = os.path.join(PROJECT_ROOT, "logs", "blocked_ips.json")

# Attack severity → colour mapping used throughout
SEVERITY_COLORS = {
    "Normal": "#10B981",
    "DoS":    "#EF4444",
    "Probe":  "#F59E0B",
    "R2L":    "#8B5CF6",
    "U2R":    "#DC2626",
}

ACTION_COLORS = {
    "BLOCKED":            "#EF4444",
    "BLOCKED(DRY_RUN)":   "#F87171",
    "ALREADY_BLOCKED":    "#DC2626",
    "BLOCK_FAILED":       "#F59E0B",
    "ALLOWLISTED":        "#10B981",
    "BELOW_THRESHOLD":    "#94A3B8",
    "LOGGED":             "#3B82F6",
}


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SONA — Live SOC Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .kpi-card { background: var(--background-color, #0E1117);
                border: 1px solid #2D3340; border-radius: 10px;
                padding: 16px 18px; }
    .kpi-value { font-size: 28px; font-weight: 700; line-height: 1.1; }
    .kpi-label { font-size: 12px; color: #94A3B8; text-transform: uppercase;
                 letter-spacing: 0.04em; margin-top: 4px; }
    .live-dot { height: 10px; width: 10px; border-radius: 50%;
                background: #10B981; display: inline-block;
                animation: pulse 1.5s infinite; margin-right: 6px; }
    @keyframes pulse { 0%{opacity:1;} 50%{opacity:0.3;} 100%{opacity:1;} }
    .attack-row-DoS   { border-left: 4px solid #EF4444; }
    .attack-row-Probe { border-left: 4px solid #F59E0B; }
    .attack-row-R2L   { border-left: 4px solid #8B5CF6; }
    .attack-row-U2R   { border-left: 4px solid #DC2626; }
</style>
""", unsafe_allow_html=True)


# ── Data loading (re-read every refresh — no caching, we WANT fresh data) ────

def load_events() -> list[dict]:
    if not os.path.exists(LOG_JSON_PATH):
        return []
    try:
        with open(LOG_JSON_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def load_blocked() -> dict:
    if not os.path.exists(BLOCKED_PATH):
        return {}
    try:
        with open(BLOCKED_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def geolocate_ip(ip: str) -> dict:
    """
    Free GeoIP lookup via ip-api.com (45 req/min limit, no key needed).
    Cached for 1 hour per IP so we don't hammer the API.
    Private/reserved IPs are skipped.
    """
    import ipaddress
    try:
        addr = ipaddress.ip_address(ip)
        if addr.is_private or addr.is_loopback or addr.is_link_local:
            return {"lat": None, "lon": None, "country": "Private network", "city": ""}
    except ValueError:
        return {"lat": None, "lon": None, "country": "Unknown", "city": ""}

    if not REQUESTS_AVAILABLE:
        return {"lat": None, "lon": None, "country": "Unknown", "city": ""}

    try:
        r = requests.get(f"http://ip-api.com/json/{ip}", timeout=1.5)
        d = r.json()
        if d.get("status") == "success":
            return {"lat": d.get("lat"), "lon": d.get("lon"),
                     "country": d.get("country", ""), "city": d.get("city", "")}
    except Exception:
        pass
    return {"lat": None, "lon": None, "country": "Unknown", "city": ""}


def unblock_ip_action(ip: str, rule_name: str):
    """Remove firewall rule directly from the dashboard process."""
    import platform, subprocess
    system = platform.system()
    try:
        if system == "Windows":
            cmd = ["netsh", "advfirewall", "firewall", "delete", "rule", f"name={rule_name}"]
        else:
            cmd = ["iptables", "-D", "INPUT", "-s", ip, "-j", "DROP"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        success = result.returncode == 0
    except Exception:
        success = False

    if success:
        # Update persisted state so it disappears from the dashboard
        blocked = load_blocked()
        blocked.pop(ip, None)
        os.makedirs(os.path.dirname(BLOCKED_PATH), exist_ok=True)
        with open(BLOCKED_PATH, "w") as f:
            json.dump(blocked, f, indent=2)
    return success


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ SONA SOC")
    st.markdown("Security of Network Access")
    st.caption("Live monitoring dashboard")
    st.divider()

    if AUTOREFRESH_AVAILABLE:
        refresh_sec = st.slider("Auto-refresh (seconds)", 2, 30, 3)
        st_autorefresh(interval=refresh_sec * 1000, key="soc_refresh")
        st.success(f"🔴 Live — refreshing every {refresh_sec}s")
    else:
        st.warning("Install `streamlit-autorefresh` for live auto-refresh:\n\n"
                    "`pip install streamlit-autorefresh`")
        if st.button("🔄 Refresh now"):
            st.rerun()

    st.divider()
    window = st.selectbox(
        "Time window",
        ["Last 15 min", "Last hour", "Last 6 hours", "All time"],
        index=1,
    )
    st.divider()
    st.caption(
        "Run SONA in another terminal:\n\n"
        "`python sona.py --live`\n\n"
        "This dashboard reads its logs live."
    )


# ── Load and filter data ──────────────────────────────────────────────────────

all_events = load_events()
blocked    = load_blocked()

window_minutes = {
    "Last 15 min": 15, "Last hour": 60,
    "Last 6 hours": 360, "All time": None,
}[window]

if window_minutes:
    cutoff = datetime.now() - timedelta(minutes=window_minutes)
    events = [
        e for e in all_events
        if datetime.strptime(e["timestamp"], "%Y-%m-%d %H:%M:%S") >= cutoff
    ]
else:
    events = all_events

df = pd.DataFrame(events) if events else pd.DataFrame(
    columns=["timestamp", "src_ip", "dst_ip", "attack_type", "confidence", "action"]
)


# ── Header ─────────────────────────────────────────────────────────────────

h1, h2 = st.columns([3, 1])
with h1:
    st.markdown("# 🛡️ SONA — Live Security Operations Center")
with h2:
    st.markdown(
        f"<div style='text-align:right;padding-top:20px'>"
        f"<span class='live-dot'></span>{datetime.now().strftime('%H:%M:%S')}"
        f"</div>", unsafe_allow_html=True
    )

if not all_events:
    st.info(
        "No events yet. Start SONA in another terminal to begin monitoring:\n\n"
        "```\npython sona.py --live --interface Wi-Fi\n```\n\n"
        "This dashboard will populate automatically as traffic is analysed."
    )
    st.stop()


# ── KPI row ────────────────────────────────────────────────────────────────

total_events   = len(df)
attack_events  = df[df["attack_type"] != "Normal"] if "attack_type" in df else pd.DataFrame()
blocked_count  = len(blocked)
threat_count   = len(attack_events)
unique_attackers = attack_events["src_ip"].nunique() if not attack_events.empty else 0
block_actions  = df[df["action"].isin(["BLOCKED", "BLOCKED(DRY_RUN)"])] if "action" in df else pd.DataFrame()

k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    (k1, "Events analysed",   total_events,   "#3B82F6"),
    (k2, "Threats detected",  threat_count,   "#F59E0B"),
    (k3, "IPs blocked (live)",blocked_count,  "#EF4444"),
    (k4, "Unique attacker IPs",unique_attackers,"#8B5CF6"),
    (k5, "Block actions taken",len(block_actions),"#DC2626"),
]
for col, label, val, color in kpis:
    col.markdown(
        f"<div class='kpi-card'><div class='kpi-value' style='color:{color}'>{val}</div>"
        f"<div class='kpi-label'>{label}</div></div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ── Main layout: live feed + charts ───────────────────────────────────────────

col_feed, col_charts = st.columns([1.3, 1])

with col_feed:
    st.markdown("### 📡 Live threat feed")
    if attack_events.empty:
        st.success("No threats detected in this time window. Network looks clean.")
    else:
        recent = attack_events.sort_values("timestamp", ascending=False).head(25)
        for _, row in recent.iterrows():
            atype = row.get("attack_type", "Unknown")
            color = SEVERITY_COLORS.get(atype, "#94A3B8")
            action = row.get("action", "")
            action_color = ACTION_COLORS.get(action, "#94A3B8")
            st.markdown(
                f"""
                <div style="border-left:4px solid {color}; padding:8px 12px;
                            margin-bottom:6px; background:rgba(255,255,255,0.03);
                            border-radius:4px; font-size:13px;">
                    <b>{row['timestamp']}</b> &nbsp;
                    <code>{row['src_ip']}</code> → <code>{row.get('dst_ip','')}</code>:{row.get('dst_port','')}
                    &nbsp;|&nbsp; <span style="color:{color};font-weight:600">{atype}</span>
                    ({row.get('confidence',0):.0%})
                    &nbsp;|&nbsp; <span style="color:{action_color}">{action}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

with col_charts:
    st.markdown("### 📊 Attack breakdown")
    if not attack_events.empty:
        cat_counts = attack_events["attack_type"].value_counts().reset_index()
        cat_counts.columns = ["Attack type", "Count"]
        fig = px.bar(
            cat_counts, x="Count", y="Attack type", orientation="h",
            color="Attack type",
            color_discrete_map=SEVERITY_COLORS,
        )
        fig.update_layout(showlegend=False, height=220, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No attacks to break down yet.")

    st.markdown("### ⏱️ Events over time")
    if not df.empty:
        df["ts"] = pd.to_datetime(df["timestamp"])
        df["minute"] = df["ts"].dt.floor("min")
        timeline = df.groupby(["minute", "attack_type"]).size().reset_index(name="count")
        fig2 = px.bar(
            timeline, x="minute", y="count", color="attack_type",
            color_discrete_map=SEVERITY_COLORS,
        )
        fig2.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=0),
                            legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig2, use_container_width=True)


st.divider()


# ── Blocked IPs table with unblock action ─────────────────────────────────────

st.markdown("### 🔒 Currently blocked IPs")

if not blocked:
    st.caption("No IPs currently blocked.")
else:
    for ip, info in blocked.items():
        c1, c2, c3, c4, c5 = st.columns([2, 1.5, 1.5, 2, 1])
        c1.markdown(f"**{ip}**")
        c2.markdown(f"`{info.get('attack_type','?')}`")
        c3.markdown(f"{info.get('confidence',0):.0%} conf.")
        c4.caption(f"Blocked: {info.get('blocked_at','')}")
        if c5.button("Unblock", key=f"unblock_{ip}"):
            ok = unblock_ip_action(ip, info.get("rule_name", ""))
            if ok:
                st.success(f"Unblocked {ip}")
                st.rerun()
            else:
                st.error(f"Failed to unblock {ip} — try running as Administrator.")


st.divider()


# ── Attacker geography ─────────────────────────────────────────────────────────

st.markdown("### 🌍 Attacker geography")

if attack_events.empty:
    st.caption("No attack traffic to map yet.")
else:
    unique_ips = attack_events["src_ip"].unique()[:20]  # cap lookups
    geo_rows = []
    for ip in unique_ips:
        geo = geolocate_ip(ip)
        if geo["lat"] is not None:
            attack_types_for_ip = attack_events[attack_events["src_ip"] == ip]["attack_type"].mode()
            geo_rows.append({
                "ip": ip, "lat": geo["lat"], "lon": geo["lon"],
                "location": f"{geo['city']}, {geo['country']}",
                "attack_type": attack_types_for_ip.iloc[0] if not attack_types_for_ip.empty else "Unknown",
            })

    if geo_rows:
        geo_df = pd.DataFrame(geo_rows)
        fig3 = px.scatter_geo(
            geo_df, lat="lat", lon="lon", color="attack_type",
            hover_name="ip", hover_data=["location"],
            color_discrete_map=SEVERITY_COLORS,
            projection="natural earth",
        )
        fig3.update_layout(height=350, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.caption("All attacking IPs are private/local network addresses — nothing to map. "
                   "This is expected when testing on your own LAN.")


st.divider()
st.caption(
    "SONA SOC Dashboard reads live logs from `logs/threat_log.json` and "
    "`logs/blocked_ips.json`, written by `sona.py` running in another terminal. "
    "This decoupled design mirrors how real SIEM tools separate sensors from viewers."
)
