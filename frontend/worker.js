// This is the Web Worker for WASM-mode inference. It runs in the background.

importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/ort.min.js');

let session = null;

// Initialize ONNX Runtime session
self.addEventListener('message', async (event) => {
    const { type, payload } = event.data;

    if (type === 'init') {
        try {
            session = await ort.InferenceSession.create(payload.modelUrl);
            self.postMessage({ type: 'init_complete', status: 'success' });
            console.log("WASM model loaded successfully in worker.");
        } catch (e) {
            console.error("Failed to load WASM model:", e);
            self.postMessage({ type: 'init_complete', status: 'error', error: e.toString() });
        }
    } else if (type === 'infer' && session) {
        const { frameData, capture_ts } = payload;

        // This is a placeholder for actual inference logic.
        // It would involve converting the image data to a tensor,
        // running the session, and post-processing the output.
        // For simplicity, we'll return dummy data instantly.
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
