"""
event_logger.py — state change logger + shift report generator
Logs every state transition with timestamp and risk.
Generates per-shift summary reports.
Persists to JSON, keeps last 2000 events in memory.
"""

import json
import os
import time
import threading
import logging
from collections import defaultdict, deque

log = logging.getLogger(__name__)

MAX_EVENTS   = 2000
PERSIST_SECS = 60


class EventLogger:

    def __init__(self, data_dir: str):
        self._events    = deque(maxlen=MAX_EVENTS)
        self._lock      = threading.Lock()
        self._data_file = os.path.join(data_dir, "event_log.json")
        self._shift_start = time.time()

        self._load()

        t = threading.Thread(target=self._persist_loop, daemon=True)
        t.start()

    def log(self, state: str, risk: int, alerts: list):
        entry = {
            "ts":     round(time.time(), 2),
            "state":  state,
            "risk":   risk,
            "alerts": [a.get("type", "") for a in alerts] if alerts else [],
        }
        with self._lock:
            self._events.append(entry)

    def recent(self, limit: int = 200) -> list:
        with self._lock:
            events = list(self._events)
        return list(reversed(events[-limit:]))

    def shift_report(self) -> dict:
        with self._lock:
            events = list(self._events)

        # Filter to this shift
        shift_events = [e for e in events if e["ts"] >= self._shift_start]

        # Time per state (approximate — use transitions)
        state_durations = defaultdict(float)
        for i in range(len(shift_events) - 1):
            s    = shift_events[i]["state"]
            dt   = shift_events[i+1]["ts"] - shift_events[i]["ts"]
            dt   = min(dt, 300)   # cap at 5 min (handles gaps)
            state_durations[s] += dt

        # Count alert types
        all_alerts = []
        for e in shift_events:
            all_alerts.extend(e.get("alerts", []))
        alert_counts = defaultdict(int)
        for a in all_alerts:
            alert_counts[a] += 1

        # Peak risk event
        peak = max(shift_events, key=lambda e: e["risk"], default=None)

        return {
            "shift_start":     self._shift_start,
            "shift_mins":      round((time.time() - self._shift_start) / 60, 1),
            "total_events":    len(shift_events),
            "state_durations": {k: round(v, 1) for k, v in state_durations.items()},
            "alert_counts":    dict(alert_counts),
            "peak_risk":       peak,
            "edge_events":     alert_counts.get("EDGE_ALERT", 0),
            "fall_suspects":   alert_counts.get("FALL_SUSPECTED", 0),
            "fall_confirmed":  alert_counts.get("FALL_CONFIRMED", 0),
        }

    def new_shift(self):
        self._shift_start = time.time()
        log.info("New shift started")

    def _save(self):
        try:
            with self._lock:
                data = {
                    "events":      list(self._events),
                    "shift_start": self._shift_start,
                }
            with open(self._data_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            log.warning("Event log save error: %s", e)

    def _load(self):
        if not os.path.exists(self._data_file):
            return
        try:
            with open(self._data_file) as f:
                data = json.load(f)
            for e in data.get("events", []):
                self._events.append(e)
            self._shift_start = data.get("shift_start", time.time())
            log.info("Event log loaded (%d events)", len(self._events))
        except Exception as e:
            log.warning("Event log load error: %s", e)

    def _persist_loop(self):
        while True:
            time.sleep(PERSIST_SECS)
            self._save()
