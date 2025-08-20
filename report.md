# Design summary

- Signaling and relay: a minimal Python (aiohttp + aiortc) server accepts two roles — `publisher` (phone) and `viewer` (laptop). The phone publishes its camera as a WebRTC video track; the viewer receives that track. A shared frame queue with length 1 applies backpressure (dropping old frames).
- Overlay contract: results follow the JSON schema with normalized coordinates. Browser aligns by `frame_id`/`capture_ts` and draws on a canvas over the video element.
- Low-resource mode: `wasm` processes 320×240 frames at ~10 FPS using a web worker; only the latest frame is processed and older frames are dropped.
- Server mode: scaffolded fan-out path sends detections via DataChannel. Replace the dummy detections with ONNX inference to enable CPU-based server inference.

## Backpressure policy

- Fixed queue size (1) for frames; newest frame replaces the old one when overloaded, ensuring freshness and stable latency.
- Viewer only paints the most recent detection result.

## Tradeoffs

- Using a shared queue simplifies the design but couples multiple viewers; per-peer queues would isolate backpressure.
- Wasm mode uses dummy inference in this example; swapping in onnxruntime-web is straightforward but increases bundle cost.
- Server mode reduces phone CPU but increases downlink bandwidth for detection results and may raise latency.

## Low-resource path

- 320×240 input scaling, ~10 FPS sampling, single-frame queue, canvas overlay only for latest frame.
- Intended to run smoothly on modest laptops (no GPU).

## Next improvement

- Add real onnxruntime-web (wasm) and an actual quantized YOLO/SSD model; implement server ONNX path for `MODE=server`.


