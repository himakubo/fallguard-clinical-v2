# FallGuard Clinical V2

Privacy-first AI patient monitoring for senior care and hospital wards.

## What it does
Detects patient state in real time using a ceiling-mounted camera and edge AI.
No cloud. No raw video exposed to caregivers by default.

## States detected
LYING_IN_BED → ACTIVE_IN_BED → EDGE_SITTING → EXITING_BED → STANDING → WALKING → FALL_CONFIRMED

## Architecture
- **Raspberry Pi 4** — camera capture + MJPEG stream only
- **GMK Tek N97** — YOLOv8n-Pose + ByteTrack + behavior engine + fall engine + dashboard

## Why AI moved from Pi to GMK Tek
Running YOLO + pose estimation on Pi 4 ARM CPU drops to 3-4 FPS and freezes the camera.
GMK Tek N97 runs the full pipeline at 8-10 FPS. Pi does camera-only — never freezes.

## AI Models
- **YOLOv8n-Pose** — person detection + 17-point skeleton in one model
- **ByteTrack** — persistent patient ID, survives nurse walking into frame
- **Behavior engine** — 12-state deterministic state machine
- **Fall engine V2** — 3-condition confirmation (no false alarms)

## Setup
See pi_node/setup_pi.sh and gmk_node/setup_gmk.sh
