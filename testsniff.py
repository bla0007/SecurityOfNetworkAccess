from scapy.all import sniff

print("Starting sniff...")

sniff(
    iface="20",
    filter="ip",
    prn=lambda p: print(p.summary()),
    store=False
)