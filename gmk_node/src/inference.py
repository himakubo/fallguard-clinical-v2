"""
inference.py — main AI loop on GMK Tek
YOLOv8-Pose gives us both detection AND keypoints in one call.
No separate pose_engine.py needed anymore.

Flow per frame:
  frame → YOLOv8-Pose+ByteTrack → patient keypoints
        → body-part engine → zone engine
        → behavior engine → fall engine
        → shared state dict
"""

import threading
import time
import logging

import frame_grabber
import detector as det
import zone_manager
import behavior_engine as be
import fall_engine as fe

log = logging.getLogger(__name__)

_lock    = threading.Lock()
_state   = {
    "state":         "NO_PERSON",
    "risk_score":    0,
    "alerts":        [],
    "head":          None,
    "torso":         None,
    "feet":          None,
    "head_zone":     "UNKNOWN",
    "torso_zone":    "UNKNOWN",
    "feet_zone":     "UNKNOWN",
    "posture":       "UNKNOWN",
    "body_angle":    0.0,
    "motion_score":  0.0,
    "fall_velocity": 0.0,
    "on_floor":      False,
    "landmarks":     [],
    "fps":           0.0,
    "time_at_edge":  0.0,
    "calibrated":    False,
    "patient_id":    None,
    "persons_count": 0,
    "ts":            0.0,
    "body_parts":    {"head": None, "torso": None, "feet": None},
    "zones":         {"head_zone":"UNKNOWN","torso_zone":"UNKNOWN","feet_zone":"UNKNOWN"},
    "hospital":      {"room_number":"4B","patient_id":"P-001"},
}
_running = False


def get_state():
    with _lock:
        return dict(_state)


def _loop():
    global _running

    detector  = det.PersonDetector()
    behaviour = be.BehaviorEngine()
    fall      = fe.FallEngine()

    t0, fc, fps = time.time(), 0, 0.0
    log.info("GMK Tek inference loop started")

    while _running:
        frame = frame_grabber.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        # ── Detection + pose in one call ──────────────────────────────
        bed_zone = zone_manager.get_bed_zone()
        persons  = detector.detect(frame, bed_zone)
        patient  = detector.get_patient(persons)

        fc += 1
        elapsed = time.time() - t0
        if elapsed >= 2.0:
            fps = round(fc/elapsed, 1); fc = 0; t0 = time.time()

        if patient is None:
            behav_res = behaviour.update(
                {"head_zone":"UNKNOWN","torso_zone":"UNKNOWN","feet_zone":"UNKNOWN"},
                "UNKNOWN", 0,
                {"suspected":False,"confirmed":False,"on_floor":False,"velocity":0.0}
            )
            with _lock:
                _state.update({
                    "state":"NO_PERSON","risk_score":0,"alerts":[],
                    "head":None,"torso":None,"feet":None,
                    "head_zone":"UNKNOWN","torso_zone":"UNKNOWN","feet_zone":"UNKNOWN",
                    "posture":"UNKNOWN","body_angle":0.0,"motion_score":0.0,
                    "fall_velocity":0.0,"on_floor":False,"landmarks":[],
                    "fps":fps,"time_at_edge":0.0,
                    "calibrated":zone_manager.is_calibrated(),
                    "patient_id":None,"persons_count":len(persons),
                    "ts":time.time(),
                    "body_parts":{"head":None,"torso":None,"feet":None},
                    "zones":{"head_zone":"UNKNOWN","torso_zone":"UNKNOWN","feet_zone":"UNKNOWN"},
                })
            continue

        # ── Zone classification ────────────────────────────────────────
        zones = zone_manager.classify_parts(patient["head"], patient["torso"], patient["feet"])

        # ── Fall + behavior ────────────────────────────────────────────
        fall_res  = fall.update(patient)
        behav_res = behaviour.update(
            zones,
            patient["posture"],
            patient["motion_score"],
            fall_res,
        )

        with _lock:
            _state.update({
                "state":         behav_res["state"],
                "risk_score":    behav_res["risk_score"],
                "alerts":        behav_res["alerts"],
                "head":          patient["head"],
                "torso":         patient["torso"],
                "feet":          patient["feet"],
                "head_zone":     zones["head_zone"],
                "torso_zone":    zones["torso_zone"],
                "feet_zone":     zones["feet_zone"],
                "posture":       patient["posture"],
                "body_angle":    patient["body_angle"],
                "motion_score":  patient["motion_score"],
                "fall_velocity": fall_res["velocity"],
                "on_floor":      fall_res["on_floor"],
                "landmarks":     patient["keypoints"],
                "fps":           fps,
                "time_at_edge":  behav_res["time_at_edge"],
                "calibrated":    zone_manager.is_calibrated(),
                "patient_id":    patient["id"],
                "persons_count": len(persons),
                "ts":            time.time(),
                "body_parts":    {"head":patient["head"],"torso":patient["torso"],"feet":patient["feet"]},
                "zones":         zones,
            })


def start():
    global _running
    if _running:
        return
    _running = True
    t = threading.Thread(target=_loop, daemon=True, name="inference")
    t.start()


def stop():
    global _running
    _running = False
