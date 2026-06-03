"""
fall_engine.py — V2 fall detection
Requires ALL THREE conditions to confirm a fall:
  1. Rapid descent  (hip Y increases > threshold in short time)
  2. Ground contact (majority of visible keypoints in bottom 35% of frame)
  3. Persistence    (stays down >= CONFIRM_SECONDS without recovery)

V1 velocity-only approach dropped — too many false alarms.
"""

import time
import logging
from collections import deque

log = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
VELOCITY_THRESHOLD   = 0.30   # hip Y change per second (normalised coords)
FLOOR_FRACTION       = 0.35   # bottom 35% of frame = "ground"
FLOOR_KP_RATIO       = 0.55   # >55% of visible keypoints must be in floor zone
CONFIRM_SECONDS      = 2.5    # must stay down this long to confirm
RECOVERY_WINDOW      = 2.5    # if stands up within this, cancel fall
HISTORY_FRAMES       = 8      # frames of hip position history for velocity calc


class FallEngine:

    def __init__(self):
        self._hip_history    = deque(maxlen=HISTORY_FRAMES)   # (timestamp, y)
        self._fall_start     = None   # when rapid descent was first detected
        self._suspected_at   = None
        self._on_floor       = False

    def update(self, pose_result):
        """
        pose_result: dict from pose_engine.process() or None
        Returns: {suspected, confirmed, on_floor, velocity}
        """
        if pose_result is None:
            return {"suspected": False, "confirmed": False,
                    "on_floor": False, "velocity": 0.0}

        landmarks = pose_result.get("landmarks", [])
        torso     = pose_result.get("torso", [None, None])

        # ── Hip Y tracking (torso center Y) ──────────────────────────────────
        ty = torso[1] if torso and torso[1] is not None else None
        now = time.time()

        if ty is not None:
            self._hip_history.append((now, ty))

        # ── Velocity: change in hip Y over recent frames ──────────────────────
        velocity = 0.0
        if len(self._hip_history) >= 3:
            oldest_t, oldest_y = self._hip_history[0]
            dt = now - oldest_t
            if dt > 0.05:
                velocity = (ty - oldest_y) / dt if ty is not None else 0.0

        # ── Floor contact: are most keypoints near the bottom? ────────────────
        visible_kps = [lm for lm in landmarks if lm.get("visibility", 0) >= 0.4]
        floor_kps   = [lm for lm in visible_kps if lm.get("y", 0) >= (1.0 - FLOOR_FRACTION)]
        on_floor = (
            len(visible_kps) > 0 and
            len(floor_kps) / len(visible_kps) >= FLOOR_KP_RATIO
        )
        self._on_floor = on_floor

        # ── Rapid descent detected? ───────────────────────────────────────────
        rapid_descent = velocity > VELOCITY_THRESHOLD

        # ── State machine ─────────────────────────────────────────────────────
        suspected  = False
        confirmed  = False

        if rapid_descent and on_floor:
            if self._fall_start is None:
                self._fall_start  = now
                self._suspected_at = now
                log.warning("Fall SUSPECTED — rapid descent + floor contact")
            suspected = True

            # Recovery: if hip Y is rising (person getting up), cancel
            if len(self._hip_history) >= 3:
                recent_ys = [y for _, y in list(self._hip_history)[-3:]]
                if recent_ys[-1] < recent_ys[0] - 0.05:  # moving upward
                    if now - self._fall_start < RECOVERY_WINDOW:
                        log.info("Fall cancelled — person recovering")
                        self._fall_start  = None
                        self._suspected_at = None
                        suspected = False

            # Confirm after staying down long enough
            if self._fall_start and (now - self._fall_start) >= CONFIRM_SECONDS:
                confirmed = True
                log.error("Fall CONFIRMED at %.1fs", now - self._fall_start)

        else:
            # No fall conditions — reset if we had a suspected fall
            if self._fall_start is not None:
                log.info("Fall cleared — no longer on floor")
            self._fall_start   = None
            self._suspected_at = None

        return {
            "suspected":  suspected,
            "confirmed":  confirmed,
            "on_floor":   on_floor,
            "velocity":   round(velocity, 3),
        }
