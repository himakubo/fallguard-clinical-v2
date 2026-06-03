"""
detector.py — YOLOv8n-Pose + ByteTrack
Single model replaces old YOLO + MediaPipe two-model approach.

Why YOLOv8-Pose instead of YOLOv8n + MediaPipe:
  - YOLOv8n trained on COCO eye-level photos → fails on overhead/angled cameras
  - YOLOv8-Pose trained on diverse datasets including CCTV/surveillance footage
  - One model call gives both bounding box AND 17 keypoints
  - MediaPipe was designed for front-facing selfie cameras, poor on overhead views
  - Tested: YOLOv8x at conf=0.15 returned 0 detections; YOLOv8n-pose detects correctly

COCO 17 keypoints (0-indexed):
  0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
  5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow
  9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip
  13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle

ByteTrack patient ID logic:
  - First detection: person in bed zone = patient, else largest bbox
  - Patient ID locked across all frames
  - Nurse walks in → gets different ID → patient never switches
  - Patient lost > 5s → reassign on next bed-zone entry
"""

import time
import logging
import numpy as np
from ultralytics import YOLO

log = logging.getLogger(__name__)

MODEL      = "yolov8n-pose.pt"
CONF       = 0.20    # lower than standard — overhead angles reduce confidence scores
LOST_SECS  = 5.0

# COCO keypoint indices used for body-part calculation
KP_HEAD   = [0, 1, 2, 3, 4]          # nose, eyes, ears
KP_TORSO  = [5, 6, 11, 12]           # shoulders, hips
KP_FEET   = [15, 16]                  # ankles
KP_KNEES  = [13, 14]
VIS_THRESH = 0.3


class PersonDetector:

    def __init__(self):
        log.info("Loading YOLOv8n-Pose...")
        self._model      = YOLO(MODEL)
        self._patient_id = None
        self._lost_since = None
        log.info("YOLOv8n-Pose loaded")

    def detect(self, rgb_frame, bed_zone_pts=None):
        """
        Run YOLOv8-Pose + ByteTrack on frame.
        Returns list of person dicts with pose keypoints.

        Each person:
        {
          id, bbox:[nx1,ny1,nx2,ny2], conf,
          cx, cy,                      # normalised bbox center
          is_patient,
          keypoints: [{name,x,y,visibility}, ...],  # normalised 0-1
          head:[x,y], torso:[x,y], feet:[x,y],
          body_angle, motion_score, posture
        }
        """
        h, w = rgb_frame.shape[:2]

        results = self._model.track(
            rgb_frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONF,
            verbose=False,
            imgsz=640,
        )

        persons = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            kpts_all = results[0].keypoints

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                tid  = int(box.id[0]) if box.id is not None else i

                nx1, ny1 = x1/w, y1/h
                nx2, ny2 = x2/w, y2/h
                cx  = (nx1+nx2)/2
                cy  = (ny1+ny2)/2

                # Extract keypoints for this person
                kpts = []
                head, torso, feet = None, None, None
                body_angle  = 0.0
                posture     = "UNKNOWN"
                motion_score = 0.0

                if kpts_all is not None and i < len(kpts_all.xy):
                    kp_xy  = kpts_all.xy[i].cpu().numpy()    # (17, 2) pixel coords
                    kp_vis = kpts_all.conf[i].cpu().numpy() if kpts_all.conf is not None else np.ones(17)

                    NAMES = ["nose","left_eye","right_eye","left_ear","right_ear",
                             "left_shoulder","right_shoulder","left_elbow","right_elbow",
                             "left_wrist","right_wrist","left_hip","right_hip",
                             "left_knee","right_knee","left_ankle","right_ankle"]

                    kpts = [
                        {
                            "name":       NAMES[j],
                            "x":          round(float(kp_xy[j,0]) / w, 4),
                            "y":          round(float(kp_xy[j,1]) / h, 4),
                            "visibility": round(float(kp_vis[j]), 3),
                        }
                        for j in range(17)
                    ]

                    def avg_kp(indices):
                        xs, ys = [], []
                        for idx in indices:
                            if kp_vis[idx] >= VIS_THRESH and kp_xy[idx,0] > 0:
                                xs.append(kp_xy[idx,0]/w)
                                ys.append(kp_xy[idx,1]/h)
                        if not xs: return None
                        return [round(float(np.mean(xs)),4), round(float(np.mean(ys)),4)]

                    head  = avg_kp(KP_HEAD)
                    torso = avg_kp(KP_TORSO)
                    feet  = avg_kp(KP_FEET)

                    # Body angle: nose Y vs ankle Y (0=lying, 90=standing)
                    if kp_vis[0] >= VIS_THRESH and kp_vis[15] >= VIS_THRESH:
                        spread = abs(kp_xy[0,1] - kp_xy[15,1]) / h
                        body_angle = min(90.0, float(spread * 180))
                    elif torso and feet:
                        spread = abs(torso[1] - feet[1])
                        body_angle = min(90.0, float(spread * 180))

                    if   body_angle < 20:  posture = "LYING"
                    elif body_angle < 40:  posture = "RECLINING"
                    elif body_angle < 65:  posture = "SITTING"
                    else:                  posture = "STANDING"

                    # Motion score vs previous frame
                    raw = [(kp_xy[j,0]/w, kp_xy[j,1]/h) for j in range(17)]
                    attr = f"_prev_{tid}"
                    if hasattr(self, attr):
                        prev = np.array(getattr(self, attr))
                        curr = np.array(raw)
                        motion_score = float(min(100.0, np.mean(np.abs(curr-prev))*2000))
                    setattr(self, attr, raw)

                persons.append({
                    "id":           tid,
                    "bbox":         [round(nx1,4), round(ny1,4), round(nx2,4), round(ny2,4)],
                    "conf":         round(conf, 3),
                    "cx":           round(cx, 4),
                    "cy":           round(cy, 4),
                    "is_patient":   False,
                    "keypoints":    kpts,
                    "head":         head,
                    "torso":        torso,
                    "feet":         feet,
                    "body_angle":   round(body_angle, 1),
                    "posture":      posture,
                    "motion_score": round(motion_score, 1),
                })

        # ── Patient ID assignment ─────────────────────────────────────
        if not persons:
            if self._patient_id is not None and self._lost_since is None:
                self._lost_since = time.time()
            if self._lost_since and (time.time()-self._lost_since) > LOST_SECS:
                log.info("Patient ID %s lost — will reassign", self._patient_id)
                self._patient_id = None
                self._lost_since = None
            return persons

        self._lost_since = None
        ids = [p["id"] for p in persons]

        # Current patient still visible
        if self._patient_id in ids:
            for p in persons:
                if p["id"] == self._patient_id:
                    p["is_patient"] = True
            return persons

        # Reassign — prefer person inside bed zone
        if bed_zone_pts and len(bed_zone_pts) >= 3:
            for p in persons:
                if _pip(p["cx"], p["cy"], bed_zone_pts):
                    self._patient_id = p["id"]
                    p["is_patient"]  = True
                    log.info("Patient assigned (bed zone): ID %d", p["id"])
                    return persons

        # Fallback: largest bounding box
        largest = max(persons, key=lambda p: (
            (p["bbox"][2]-p["bbox"][0]) * (p["bbox"][3]-p["bbox"][1])
        ))
        self._patient_id     = largest["id"]
        largest["is_patient"] = True
        log.info("Patient assigned (largest bbox): ID %d", largest["id"])
        return persons

    def get_patient(self, persons):
        for p in persons:
            if p["is_patient"]:
                return p
        return None

    def reset_patient(self):
        self._patient_id = None
        self._lost_since = None


def _pip(px, py, polygon):
    """Ray-casting point-in-polygon."""
    n, inside, j = len(polygon), False, len(polygon)-1
    for i in range(n):
        xi, yi = polygon[i]["x"], polygon[i]["y"]
        xj, yj = polygon[j]["x"], polygon[j]["y"]
        if ((yi>py) != (yj>py)) and (px < (xj-xi)*(py-yi)/(yj-yi+1e-9)+xi):
            inside = not inside
        j = i
    return inside
