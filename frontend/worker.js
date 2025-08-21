// This is the Web Worker for WASM-mode inference. It runs in the background.

importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/ort.min.js');

let session = null;

// Initialize ONNX Runtime session
self.addEventListener('message', async (event) => {
    const { type, payload } = event.data;

    if (type === 'init') {
        try {
            if (payload.modelUrl) {
                session = await ort.InferenceSession.create(payload.modelUrl);
                console.log("WASM model loaded successfully in worker.");
            } else {
                session = null; // run in dummy mode
            }
            self.postMessage({ type: 'init_complete', status: 'success' });
        } catch (e) {
            console.error("Failed to load WASM model:", e);
            // still allow dummy mode
            session = null;
            self.postMessage({ type: 'init_complete', status: 'error', error: e.toString() });
        }
    } else if (type === 'infer') {
        const { frameData, capture_ts } = payload;

        // Placeholder for actual inference. When no model is provided,
        // return dummy detections immediately.
        const inference_ts = performance.now();
        const recv_ts = performance.now(); // In WASM mode, this is very close to capture_ts

        const detections = [{
            "label": "dummy_object",
            "score": 0.99,
            "xmin": 0.2, "ymin": 0.2, "xmax": 0.8, "ymax": 0.8
        }];

        self.postMessage({
            type: 'result',
            payload: {
                frame_id: capture_ts,
                capture_ts: capture_ts,
                recv_ts: recv_ts,
                inference_ts: inference_ts,
                detections: detections
            }
        });
    }
});
