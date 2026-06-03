"""
frame_grabber.py — pulls frames from Pi MJPEG stream
Runs in background thread, fills a shared frame buffer.
GMK Tek AI pipeline reads from this buffer.
"""

import cv2
import threading
import time
import logging
import os
import requests
import numpy as np

log = logging.getLogger(__name__)

PI_IP   = os.environ.get("PI_IP",   "100.83.213.24")
PI_PORT = os.environ.get("PI_PORT", "5000")
PI_PASS = os.environ.get("PI_PASS", "fallguard2024")

_lock        = threading.Lock()
_latest_frame = None
_fps          = 0.0
_running      = False
_token        = None


def _get_token():
    global _token
    try:
        r = requests.post(
            f"http://{PI_IP}:{PI_PORT}/api/rawfeed/auth",
            json={"password": PI_PASS}, timeout=5
        )
        if r.ok:
            _token = r.json()["token"]
            log.info("Got Pi stream token")
    except Exception as e:
        log.warning("Token fetch failed: %s", e)


def _grab_loop():
    global _latest_frame, _fps, _running

    _get_token()

    url = f"http://{PI_IP}:{PI_PORT}/api/rawfeed/stream?token={_token or ''}"
    log.info("Connecting to Pi stream: %s", url)

    t0 = time.time()
    fc = 0

    while _running:
        try:
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                log.warning("Cannot open Pi stream, retrying in 3s...")
                time.sleep(3)
                continue

            log.info("Pi stream connected")
            while _running:
                ret, frame = cap.read()
                if not ret:
                    log.warning("Frame read failed, reconnecting...")
                    break
                # frame is BGR — convert to RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with _lock:
                    _latest_frame = rgb

                fc += 1
                elapsed = time.time() - t0
                if elapsed >= 2.0:
                    _fps = round(fc / elapsed, 1)
                    fc = 0
                    t0 = time.time()

            cap.release()
        except Exception as e:
            log.warning("Stream error: %s", e)
            time.sleep(3)


def get_frame():
    with _lock:
        return _latest_frame.copy() if _latest_frame is not None else None


def get_fps():
    return _fps


def start():
    global _running
    if _running:
        return
    _running = True
    t = threading.Thread(target=_grab_loop, daemon=True, name="grabber")
    t.start()
    # Wait for first frame
    for _ in range(50):
        time.sleep(0.1)
        if get_frame() is not None:
            log.info("First frame from Pi received")
            return
    log.warning("No frame from Pi yet — continuing anyway")


def stop():
    global _running
    _running = False
