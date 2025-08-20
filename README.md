# Real-time WebRTC VLM — Multi-Object Detection (Phone → Browser)

This repo provides a reproducible demo that streams live video from a phone browser via WebRTC to a laptop browser, performs multi-object detection (in two modes), and overlays bounding boxes in near real-time.

## One-command start

Option A (recommended):

```bash
./start.sh
```

Option B:

```bash
docker-compose up --build
```

By default, MODE=wasm. To run server mode:

```bash
MODE=server ./start.sh
```

## Modes

- wasm: on-device inference in the browser using a web worker (low-resource friendly). Defaults to 320×240 at ~10 FPS with frame-thinning.
- server: the backend can be extended to run ONNX inference and push detections over DataChannel (scaffold present).

## How to run

1. git clone and `cd` into this repo
2. `./start.sh` (or `docker-compose up --build`)
3. Open `http://localhost:3000` on your laptop. A QR code will display.
4. Scan the QR with your phone (Chrome on Android, Safari on iOS). Allow camera access.
5. You should see the phone video on the laptop with overlays.

If your phone cannot reach your laptop directly, set up a tunnel (e.g., ngrok) and open the public URL on the phone.

## Benchmarks

Run a short bench that writes `metrics.json`:

```bash
./bench/run_bench.sh --duration=30 --mode=wasm
```

Fields: median_e2e_latency_ms, p95_e2e_latency_ms, processed_fps, uplink_kbps, downlink_kbps.

## Troubleshooting

- If phone won’t connect: ensure both devices are on the same Wi‑Fi or use a tunnel.
- If overlays are misaligned: verify timestamps are in ms.
- If CPU is high: reduce resolution to 320×240; wasm mode already frame-thins to ~10 FPS.
- Use Chrome webrtc-internals for RTP stats.

## Notes

- This demo includes scaffolding for server-side inference; a simple dummy detection is returned. Replace with ONNX inference if desired.
- See `report.md` for design choices, low-resource mode, and backpressure policy.


