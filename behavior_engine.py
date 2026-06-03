"""
behavior_engine.py — 12-state patient behavior state machine
Input:  zone classifications + posture + motion + fall signals
Output: state string + risk score 0-100 + alert list
"""

import time
import logging

log = logging.getLogger(__name__)

# ── States ──────────────────────────────────────────────────────────────────
NO_PERSON       = "NO_PERSON"
LYING_IN_BED    = "LYING_IN_BED"
ACTIVE_IN_BED   = "ACTIVE_IN_BED"
SITTING_IN_BED  = "SITTING_IN_BED"
EDGE_SITTING    = "EDGE_SITTING"      # ← most clinically important pre-fall state
EXITING_BED     = "EXITING_BED"
STANDING        = "STANDING"
WALKING         = "WALKING"
OFF_BED         = "OFF_BED"
ON_FLOOR        = "ON_FLOOR"
FALL_SUSPECTED  = "FALL_SUSPECTED"
FALL_CONFIRMED  = "FALL_CONFIRMED"

# ── Risk scores ──────────────────────────────────────────────────────────────
RISK = {
    NO_PERSON:      0,
    LYING_IN_BED:   5,
    ACTIVE_IN_BED:  20,
    SITTING_IN_BED: 25,
    EDGE_SITTING:   55,
    EXITING_BED:    75,
    STANDING:       40,
    WALKING:        45,
    OFF_BED:        60,
    ON_FLOOR:       85,
    FALL_SUSPECTED: 95,
    FALL_CONFIRMED: 100,
}

# ── Edge alert threshold ──────────────────────────────────────────────────────
EDGE_ALERT_SECONDS = 8.0


class BehaviorEngine:

    def __init__(self):
        self.state             = NO_PERSON
        self.risk_score        = 0
        self.alerts            = []
        self._edge_start       = None
        self._state_start      = time.time()
        self._prev_state       = NO_PERSON

    def update(self, zones, posture, motion_score, fall_result):
        """
        zones        = {head_zone, torso_zone, feet_zone}
        posture      = LYING / RECLINING / SITTING / STANDING
        motion_score = 0-100
        fall_result  = {suspected: bool, confirmed: bool}
        Returns dict with state, risk_score, alerts, time_at_edge
        """
        hz = zones.get("head_zone",  "UNKNOWN")
        tz = zones.get("torso_zone", "UNKNOWN")
        fz = zones.get("feet_zone",  "UNKNOWN")

        self.alerts = []
        new_state   = self._resolve_state(hz, tz, fz, posture,
                                           motion_score, fall_result)

        # ── Edge timer ────────────────────────────────────────────────────────
        if new_state == EDGE_SITTING:
            if self._edge_start is None:
                self._edge_start = time.time()
            time_at_edge = time.time() - self._edge_start
            if time_at_edge >= EDGE_ALERT_SECONDS:
                self.alerts.append({
                    "type":    "EDGE_ALERT",
                    "message": f"Patient at bed edge for {time_at_edge:.0f}s",
                    "seconds": round(time_at_edge, 1),
                })
        else:
            self._edge_start = None
            time_at_edge     = 0.0

        # ── State change ──────────────────────────────────────────────────────
        if new_state != self.state:
            log.info("State: %s → %s  (risk %d)",
                     self.state, new_state, RISK.get(new_state, 0))
            self._prev_state  = self.state
            self.state        = new_state
            self._state_start = time.time()

        # ── Fall alerts ───────────────────────────────────────────────────────
        if new_state == FALL_SUSPECTED:
            self.alerts.append({"type": "FALL_SUSPECTED",
                                 "message": "Possible fall — monitoring"})
        if new_state == FALL_CONFIRMED:
            self.alerts.append({"type": "FALL_CONFIRMED",
                                 "message": "FALL CONFIRMED — immediate response needed"})

        self.risk_score = RISK.get(new_state, 0)

        return {
            "state":         self.state,
            "risk_score":    self.risk_score,
            "alerts":        self.alerts,
            "time_at_edge":  round(time_at_edge, 1),
            "state_duration": round(time.time() - self._state_start, 1),
        }

    def _resolve_state(self, hz, tz, fz, posture, motion, fall):
        # ── Falls override everything ─────────────────────────────────────────
        if fall.get("confirmed"):
            return FALL_CONFIRMED
        if fall.get("suspected"):
            return FALL_SUSPECTED

        # ── No person ─────────────────────────────────────────────────────────
        if hz == "UNKNOWN" and tz == "UNKNOWN" and fz == "UNKNOWN":
            return NO_PERSON

        # ── On floor / low in frame ───────────────────────────────────────────
        if fall.get("on_floor") and posture in ("LYING", "RECLINING"):
            return ON_FLOOR

        # ── Fully in bed ──────────────────────────────────────────────────────
        if hz == "BED" and tz == "BED" and fz == "BED":
            if posture in ("LYING", "RECLINING") and motion < 25:
                return LYING_IN_BED
            if motion >= 25:
                return ACTIVE_IN_BED
            if posture in ("SITTING",):
                return SITTING_IN_BED
            return LYING_IN_BED

        # ── Edge sitting — most important pre-fall state ──────────────────────
        if hz == "BED" and tz == "BED" and fz in ("EDGE", "FLOOR"):
            if posture in ("SITTING", "RECLINING"):
                return EDGE_SITTING

        # ── Exiting bed ───────────────────────────────────────────────────────
        if tz == "BED" and fz == "FLOOR":
            return EXITING_BED
        if hz == "BED" and tz in ("EDGE", "FLOOR"):
            return EXITING_BED

        # ── Standing / walking ────────────────────────────────────────────────
        if posture == "STANDING":
            if motion > 35:
                return WALKING
            return STANDING

        # ── Off bed but not fallen ─────────────────────────────────────────────
        if hz in ("FLOOR", "EDGE") and tz in ("FLOOR", "EDGE") and fz in ("FLOOR", "EDGE"):
            return OFF_BED

        # ── Default — active in bed ────────────────────────────────────────────
        return ACTIVE_IN_BED
