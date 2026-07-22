"""
packet_capture_unsw.py — SONA v3: UNSW-NB15 live capture entry point
========================================================================
Thin wrapper so sona.py can import a clean `UNSWLiveCapture` class,
matching the same shape as the original `LiveCapture`. All the real
logic (bidirectional flow tracking + UNSW-NB15 feature extraction)
lives in flow_capture.py — this just wires it into the same engine
LiveCapture already provides via its `dataset=` parameter.
"""

from packet_capture import LiveCapture


class UNSWLiveCapture(LiveCapture):
    """Same interface as LiveCapture (start/stop/alerts), pre-configured
    to use the bidirectional UNSW-NB15 pipeline instead of NSL-KDD's."""

    def __init__(self, model_dir: str = "models_unsw"):
        super().__init__(model_dir=model_dir, dataset="unsw")
