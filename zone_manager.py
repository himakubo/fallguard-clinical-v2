"""
zone_manager.py — zone classification on GMK Tek
Loads calibration from Pi via API and hot-reloads every 10s.
"""

import json, os, time, threading, logging, requests

log = logging.getLogger(__name__)

PI_IP   = os.environ.get("PI_IP",   "100.83.213.24")
PI_PORT = os.environ.get("PI_PORT", "5000")
LOCAL_CALIB = os.path.join(os.path.dirname(__file__), "..", "data", "calibration.json")

_bed_zone  = []
_edge_zone = []
_lock      = threading.Lock()


def _pip(px, py, poly):
    n, inside, j = len(poly), False, len(poly)-1
    for i in range(n):
        xi,yi = poly[i]["x"],poly[i]["y"]
        xj,yj = poly[j]["x"],poly[j]["y"]
        if ((yi>py)!=(yj>py)) and (px<(xj-xi)*(py-yi)/(yj-yi+1e-9)+xi):
            inside = not inside
        j = i
    return inside


def _load_from_pi():
    try:
        r = requests.get(f"http://{PI_IP}:{PI_PORT}/api/calibrate", timeout=2)
        # Pi doesn't have GET calibrate — load from local file synced via save
    except:
        pass
    # Load from local copy
    if os.path.exists(LOCAL_CALIB):
        try:
            with open(LOCAL_CALIB) as f:
                data = json.load(f)
            with _lock:
                global _bed_zone, _edge_zone
                _bed_zone  = data.get("bedZone",  [])
                _edge_zone = data.get("edgeZone", [])
        except Exception as e:
            log.warning("Calib load error: %s", e)


def init():
    _load_from_pi()
    def loop():
        while True:
            time.sleep(10)
            _load_from_pi()
    threading.Thread(target=loop, daemon=True).start()


def save(bed, edge):
    os.makedirs(os.path.dirname(LOCAL_CALIB), exist_ok=True)
    with open(LOCAL_CALIB, "w") as f:
        json.dump({"bedZone": bed, "edgeZone": edge}, f, indent=2)
    with _lock:
        global _bed_zone, _edge_zone
        _bed_zone, _edge_zone = bed, edge


def get_bed_zone():
    with _lock:
        return list(_bed_zone)


def classify(x, y):
    if x is None or y is None:
        return "UNKNOWN"
    with _lock:
        bed, edge = list(_bed_zone), list(_edge_zone)
    if _pip(x, y, bed):   return "BED"
    if _pip(x, y, edge):  return "EDGE"
    if bed:               return "FLOOR"
    return "UNKNOWN"


def classify_parts(head, torso, feet):
    hx,hy = (head  or [None,None])
    tx,ty = (torso or [None,None])
    fx,fy = (feet  or [None,None])
    return {
        "head_zone":  classify(hx,hy),
        "torso_zone": classify(tx,ty),
        "feet_zone":  classify(fx,fy),
    }


def is_calibrated():
    with _lock:
        return len(_bed_zone) >= 3
