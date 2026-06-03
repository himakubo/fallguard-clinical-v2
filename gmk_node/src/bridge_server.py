"""
bridge_server.py — GMK Tek main server
All AI runs here now (YOLO + ByteTrack + MediaPipe + behavior + fall)
Pi only streams MJPEG frames.

Endpoints:
  GET  /api/health
  GET  /api/pose          — full detection state
  GET  /api/heatmap       — dwell heatmap grid
  POST /api/heatmap/reset
  GET  /api/events        — event log
  GET  /api/report        — shift summary
  GET  /api/alerts/stream — SSE push alerts
  POST /api/calibrate     — save zones (also syncs to Pi)
  GET  /api/rawfeed/token — get Pi MJPEG token
  GET  /api/rawfeed/stream — proxy Pi MJPEG
"""

import asyncio, json, logging, os, time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

import frame_grabber
import inference
import zone_manager
from heatmap_engine import HeatmapEngine
from event_logger   import EventLogger

PI_IP    = os.environ.get("PI_IP",   "100.83.213.24")
PI_PORT  = os.environ.get("PI_PORT", "5000")
PI_BASE  = f"http://{PI_IP}:{PI_PORT}"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
STATIC   = os.path.join(os.path.dirname(__file__), "..", "static")

os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

heatmap = HeatmapEngine(data_dir=DATA_DIR)
events  = EventLogger(data_dir=DATA_DIR)

_alert_queues: list[asyncio.Queue] = []
_prev_state = None


async def _alert_watcher():
    """Watch inference state and push alerts via SSE."""
    global _prev_state
    while True:
        state = inference.get_state()
        s     = state.get("state")
        alerts = state.get("alerts", [])

        # Heatmap accumulation
        torso = state.get("torso")
        if torso and torso[0] is not None:
            heatmap.accumulate(torso[0], torso[1], state.get("risk_score", 0))

        # Event log on state change
        if s and s != _prev_state:
            events.log(s, state.get("risk_score", 0), alerts)
            _prev_state = s

        # Push alerts
        if alerts:
            msg = json.dumps({
                "state":  s,
                "risk":   state.get("risk_score", 0),
                "alerts": alerts,
                "ts":     time.time(),
            })
            dead = []
            for q in _alert_queues:
                try:    q.put_nowait(msg)
                except: dead.append(q)
            for q in dead:
                _alert_queues.remove(q)

        await asyncio.sleep(0.1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting GMK Tek AI pipeline")
    frame_grabber.start()
    zone_manager.init()
    inference.start()
    task = asyncio.create_task(_alert_watcher())
    log.info("All systems running")
    yield
    task.cancel()
    inference.stop()
    frame_grabber.stop()


app = FastAPI(title="FallGuard GMK", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
async def health():
    s = inference.get_state()
    return {
        "status":      "ok",
        "fps":         s.get("fps", 0),
        "state":       s.get("state"),
        "calibrated":  s.get("calibrated"),
        "patient_id":  s.get("patient_id"),
        "persons":     s.get("persons_count", 0),
        "pi_ip":       PI_IP,
    }


@app.get("/api/pose")
async def pose():
    s = inference.get_state()
    return {
        "state":        s.get("state"),
        "risk_score":   s.get("risk_score"),
        "alerts":       s.get("alerts"),
        "landmarks":    s.get("landmarks"),
        "body_parts":   s.get("body_parts"),
        "zones":        s.get("zones"),
        "posture":      s.get("posture"),
        "body_angle":   s.get("body_angle"),
        "motion_score": s.get("motion_score"),
        "fall_velocity":s.get("fall_velocity"),
        "on_floor":     s.get("on_floor"),
        "time_at_edge": s.get("time_at_edge"),
        "fps":          s.get("fps"),
        "calibrated":   s.get("calibrated"),
        "patient_id":   s.get("patient_id"),
        "persons_count":s.get("persons_count"),
        "ts":           s.get("ts"),
        "hospital":     {"room_number":"4B","patient_id":"P-001"},
    }


@app.get("/api/heatmap")
async def get_heatmap():
    return heatmap.to_dict()


@app.post("/api/heatmap/reset")
async def reset_heatmap():
    heatmap.reset()
    return {"ok": True}


@app.get("/api/events")
async def get_events(limit: int = 200):
    return {"events": events.recent(limit)}


@app.get("/api/report")
async def get_report():
    return events.shift_report()


@app.get("/api/alerts/stream")
async def alert_stream(request: Request):
    q: asyncio.Queue = asyncio.Queue(maxsize=50)
    _alert_queues.append(q)

    async def generate() -> AsyncGenerator[str, None]:
        yield 'data: {"type":"connected"}\n\n'
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        finally:
            if q in _alert_queues:
                _alert_queues.remove(q)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


@app.post("/api/calibrate")
async def calibrate(request: Request):
    data = await request.json()
    bed  = data.get("bedZone",  [])
    edge = data.get("edgeZone", [])
    if len(bed) < 3 or len(edge) < 3:
        return Response('{"error":"need 3+ points"}', status_code=400)
    # Save locally on GMK Tek
    zone_manager.save(bed, edge)
    # Also forward to Pi so it has a copy
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            await c.post(f"{PI_BASE}/api/calibrate", json=data)
    except:
        pass
    return {"ok": True, "bed_pts": len(bed), "edge_pts": len(edge)}


@app.get("/api/rawfeed/token")
async def rawfeed_token(password: str = ""):
    async with httpx.AsyncClient(timeout=5.0) as c:
        r = await c.post(f"{PI_BASE}/api/rawfeed/auth", json={"password": password})
    return Response(content=r.content, status_code=r.status_code,
                    media_type="application/json")


@app.get("/api/rawfeed/stream")
async def rawfeed_proxy(token: str = ""):
    url = f"{PI_BASE}/api/rawfeed/stream?token={token}"
    async def stream():
        async with httpx.AsyncClient(timeout=None) as c:
            async with c.stream("GET", url) as resp:
                async for chunk in resp.aiter_bytes(4096):
                    yield chunk
    return StreamingResponse(stream(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


if os.path.exists(STATIC):
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("bridge_server:app", host="0.0.0.0", port=8080,
                reload=False, log_level="info")
