"""
heatmap_engine.py — dwell-time position heatmap
Accumulates patient torso position over a shift.
Every frame increments the grid cell at (x,y) weighted by risk_score.
High-density cells = where the patient spent the most time at risk.
Grid is 64×64, normalised to match the 0-1 floor plan coordinates.
Persisted to disk every 5 minutes.
"""

import json
import os
import time
import threading
import logging
import numpy as np

log = logging.getLogger(__name__)

GRID_SIZE    = 64
PERSIST_SECS = 300   # save every 5 minutes


class HeatmapEngine:

    def __init__(self, data_dir: str):
        self._grid      = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self._lock      = threading.Lock()
        self._data_file = os.path.join(data_dir, "heatmap_grid.json")
        self._max_val   = 0.0
        self._shift_start = time.time()
        self._last_save   = time.time()
        self._frame_count = 0

        self._load()

        # Background persist thread
        t = threading.Thread(target=self._persist_loop, daemon=True)
        t.start()

    def accumulate(self, nx: float, ny: float, risk_score: int):
        """
        nx, ny: normalised position 0-1
        risk_score: 0-100 weight
        """
        gx = int(np.clip(nx * GRID_SIZE, 0, GRID_SIZE - 1))
        gy = int(np.clip(ny * GRID_SIZE, 0, GRID_SIZE - 1))
        weight = max(1, risk_score)

        with self._lock:
            self._grid[gy, gx] += weight
            self._max_val = float(self._grid.max())
            self._frame_count += 1

    def to_dict(self) -> dict:
        with self._lock:
            grid_list = self._grid.tolist()
            max_val   = self._max_val
        return {
            "grid":        grid_list,
            "size":        GRID_SIZE,
            "max_val":     max_val,
            "frames":      self._frame_count,
            "shift_start": self._shift_start,
            "shift_mins":  round((time.time() - self._shift_start) / 60, 1),
        }

    def reset(self):
        with self._lock:
            self._grid[:] = 0
            self._max_val = 0.0
            self._frame_count = 0
            self._shift_start = time.time()
        self._save()
        log.info("Heatmap reset")

    def _save(self):
        try:
            data = {
                "grid":        self._grid.tolist(),
                "shift_start": self._shift_start,
                "frames":      self._frame_count,
            }
            with open(self._data_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            log.warning("Heatmap save error: %s", e)

    def _load(self):
        if not os.path.exists(self._data_file):
            return
        try:
            with open(self._data_file) as f:
                data = json.load(f)
            grid = np.array(data["grid"], dtype=np.float32)
            if grid.shape == (GRID_SIZE, GRID_SIZE):
                with self._lock:
                    self._grid        = grid
                    self._max_val     = float(grid.max())
                    self._shift_start = data.get("shift_start", time.time())
                    self._frame_count = data.get("frames", 0)
            log.info("Heatmap loaded (%d frames)", self._frame_count)
        except Exception as e:
            log.warning("Heatmap load error: %s", e)

    def _persist_loop(self):
        while True:
            time.sleep(PERSIST_SECS)
            with self._lock:
                pass   # just to flush — save outside lock
            self._save()
            log.debug("Heatmap persisted")
