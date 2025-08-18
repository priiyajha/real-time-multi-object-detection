import asyncio
import os
import json
import logging
import cv2
import numpy as np
import onnxruntime as ort
import time

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, VideoStreamTrack
from aiortc.contrib.media import MediaRelay

# Log configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
relay = MediaRelay()
pcs = set()
MODEL_PATH = 'models/yolov8n.onnx'
MODE = os.environ.get('MODE', 'wasm')

# Backpressure queue for frames
FRAME_QUEUE = asyncio.Queue(maxsize=1)

# ONNX Runtime Inference Session
try:
    session = ort.InferenceSession(MODEL_PATH)
    logger.info(f"ONNX model loaded successfully in {MODE} mode.")
except Exception as e:
    logger.error(f"Failed to load ONNX model: {e}")
    session = None

def preprocess(frame):
    """Preprocesses a frame for inference."""
    img = cv2.cvtColor(np.array(frame.to_ndarray(format="bgr24")), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

async def detect_objects(track):
    """Processes video frames and performs object detection."""
    global FRAME_QUEUE
    while True:
        try:
            # Drop old frames if the queue is full (backpressure)
            if FRAME_QUEUE.full():
                FRAME_QUEUE.get_nowait()
                FRAME_QUEUE.task_done()
                logger.debug("Dropped a frame due to backpressure.")

            frame = await track.recv()
            await FRAME_QUEUE.put(frame)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in frame reception: {e}")
            break

async def process_and_send(pc, data_channel):
    """Pulls frames from the queue, processes them, and sends results."""
    while True:
        try:
            frame = await FRAME_QUEUE.get()
            capture_ts = frame.timestamp
            recv_ts = time.time() * 1000

            if MODE == 'server':
                inference_ts = time.time() * 1000
                if session is not None:
                    # Preprocess the frame and get predictions
                    input_data = preprocess(frame)
                    outputs = session.run(None, {'images': input_data})

                    # NOTE: YOLOv8 output post-processing is complex. This is a simplified example.
                    # A full implementation would parse the raw outputs into bounding boxes.
                    # This example sends a dummy detection to demonstrate the pipeline.
                    detections = [{
                        "label": "dummy_object",
                        "score": 0.99,
                        "xmin": 0.2, "ymin": 0.2, "xmax": 0.8, "ymax": 0.8
                    }]
                else:
                    detections = []

                inference_ts = time.time() * 1000

                result = {
                    "frame_id": int(capture_ts), # Using timestamp as ID for simplicity
                    "capture_ts": int(capture_ts),
                    "recv_ts": int(recv_ts),
                    "inference_ts": int(inference_ts),
                    "detections": detections
                }

                # Send result over the data channel
                if data_channel and data_channel.readyState == 'open':
                    data_channel.send(json.dumps(result))

            FRAME_QUEUE.task_done()

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in processing/sending: {e}")
            break

async def offer_handler(request):
    """Handles the WebRTC offer from the client."""
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    pc_id = f"PeerConnection({len(pcs)})"
    pcs.add(pc)

    logger.info(f"{pc_id} created.")

    # Handlers for PeerConnection lifecycle
    @pc.on("icecandidate")
    async def on_icecandidate(candidate):
        if candidate:
            logger.info(f"{pc_id} ICE candidate: {candidate.sdp}")
            await pc.localDescription.setLocalDescription(offer)

    @pc.on("track")
    async def on_track(track):
        logger.info(f"Track {track.kind} received from {pc_id}")
        if track.kind == "video":
            pc.video_track = track

            # Start a separate task to handle video processing
            processing_task = asyncio.ensure_future(detect_objects(track))

            # Add a listener for the video stream
            @track.on("ended")
            def on_ended():
                logger.info(f"Video track for {pc_id} ended.")
                processing_task.cancel()

    # Handle data channel
    @pc.on("datachannel")
    def on_datachannel(channel):
        logger.info(f"Data channel {channel.label} opened from {pc_id}")

        # Start the processing and sending task
        sending_task = asyncio.ensure_future(process_and_send(pc, channel))

        @channel.on("close")
        def on_close():
            logger.info(f"Data channel {channel.label} closed.")
            sending_task.cancel()

    # Create the answer and set local description
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}),
    )

async def index_handler(request):
    """Serves the main frontend page."""
    # This is a very basic way to serve the file.
    # In a real app, you would use a proper static file server.
    file_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'index.html')
    with open(file_path, 'r') as f:
        content = f.read()
    return web.Response(text=content, content_type='text/html')

async def static_handler(request):
    """Serves static files like worker.js."""
    file_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', request.path)
    if not os.path.exists(file_path):
        raise web.HTTPNotFound()

    with open(file_path, 'r') as f:
        content = f.read()

    mime_type = 'application/javascript' if request.path.endswith('.js') else 'text/plain'
    return web.Response(text=content, content_type=mime_type)

async def on_shutdown(app):
    """Closes all peer connections on shutdown."""
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

if __name__ == "__main__":
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_post("/offer", offer_handler)
    app.router.add_get("/worker.js", static_handler)
    app.on_shutdown.append(on_shutdown)
    web.run_app(app, host="0.0.0.0", port=8000)
