# import asyncio
# import os
# import json
# import logging
# import cv2
# import numpy as np
# import onnxruntime as ort
# import time
#
# from aiohttp import web
# from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, VideoStreamTrack
# from aiortc.contrib.media import MediaRelay
#
# # Log configuration
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # Global variables
# relay = MediaRelay()
# pcs = set()
# MODEL_PATH = 'models/yolov8n.onnx'
# MODE = os.environ.get('MODE', 'wasm')
#
# # Backpressure queue for frames
# FRAME_QUEUE = asyncio.Queue(maxsize=1)
#
# # ONNX Runtime Inference Session
# try:
#     session = ort.InferenceSession(MODEL_PATH)
#     logger.info(f"ONNX model loaded successfully in {MODE} mode.")
# except Exception as e:
#     logger.error(f"Failed to load ONNX model: {e}")
#     session = None
#
# def preprocess(frame):
#     """Preprocesses a frame for inference."""
#     img = cv2.cvtColor(np.array(frame.to_ndarray(format="bgr24")), cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
#     img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
#     return np.expand_dims(img, axis=0)
#
# async def detect_objects(track):
#     """Processes video frames and performs object detection."""
#     global FRAME_QUEUE
#     while True:
#         try:
#             # Drop old frames if the queue is full (backpressure)
#             if FRAME_QUEUE.full():
#                 FRAME_QUEUE.get_nowait()
#                 FRAME_QUEUE.task_done()
#                 logger.debug("Dropped a frame due to backpressure.")
#
#             frame = await track.recv()
#             await FRAME_QUEUE.put(frame)
#
#         except asyncio.CancelledError:
#             break
#         except Exception as e:
#             logger.error(f"Error in frame reception: {e}")
#             break
#
# async def process_and_send(pc, data_channel):
#     """Pulls frames from the queue, processes them, and sends results."""
#     while True:
#         try:
#             frame = await FRAME_QUEUE.get()
#             capture_ts = frame.timestamp
#             recv_ts = time.time() * 1000
#
#             if MODE == 'server':
#                 inference_ts = time.time() * 1000
#                 if session is not None:
#                     # Preprocess the frame and get predictions
#                     input_data = preprocess(frame)
#                     outputs = session.run(None, {'images': input_data})
#
#                     # NOTE: YOLOv8 output post-processing is complex. This is a simplified example.
#                     # A full implementation would parse the raw outputs into bounding boxes.
#                     # This example sends a dummy detection to demonstrate the pipeline.
#                     detections = [{
#                         "label": "dummy_object",
#                         "score": 0.99,
#                         "xmin": 0.2, "ymin": 0.2, "xmax": 0.8, "ymax": 0.8
#                     }]
#                 else:
#                     detections = []
#
#                 inference_ts = time.time() * 1000
#
#                 result = {
#                     "frame_id": int(capture_ts), # Using timestamp as ID for simplicity
#                     "capture_ts": int(capture_ts),
#                     "recv_ts": int(recv_ts),
#                     "inference_ts": int(inference_ts),
#                     "detections": detections
#                 }
#
#                 # Send result over the data channel
#                 if data_channel and data_channel.readyState == 'open':
#                     data_channel.send(json.dumps(result))
#
#             FRAME_QUEUE.task_done()
#
#         except asyncio.CancelledError:
#             break
#         except Exception as e:
#             logger.error(f"Error in processing/sending: {e}")
#             break
#
# async def offer_handler(request):
#     """Handles the WebRTC offer from the client."""
#     params = await request.json()
#     offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
#
#     pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
#     pc_id = f"PeerConnection({len(pcs)})"
#     pcs.add(pc)
#
#     logger.info(f"{pc_id} created.")
#
#     # Handlers for PeerConnection lifecycle
#     @pc.on("icecandidate")
#     async def on_icecandidate(candidate):
#         if candidate:
#             logger.info(f"{pc_id} ICE candidate: {candidate.sdp}")
#             await pc.localDescription.setLocalDescription(offer)
#
#     @pc.on("track")
#     async def on_track(track):
#         logger.info(f"Track {track.kind} received from {pc_id}")
#         if track.kind == "video":
#             pc.video_track = track
#
#             # Start a separate task to handle video processing
#             processing_task = asyncio.ensure_future(detect_objects(track))
#
#             # Add a listener for the video stream
#             @track.on("ended")
#             def on_ended():
#                 logger.info(f"Video track for {pc_id} ended.")
#                 processing_task.cancel()
#
#     # Handle data channel
#     @pc.on("datachannel")
#     def on_datachannel(channel):
#         logger.info(f"Data channel {channel.label} opened from {pc_id}")
#
#         # Start the processing and sending task
#         sending_task = asyncio.ensure_future(process_and_send(pc, channel))
#
#         @channel.on("close")
#         def on_close():
#             logger.info(f"Data channel {channel.label} closed.")
#             sending_task.cancel()
#
#     # Create the answer and set local description
#     await pc.setRemoteDescription(offer)
#     answer = await pc.createAnswer()
#     await pc.setLocalDescription(answer)
#
#     return web.Response(
#         content_type="application/json",
#         text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}),
#     )
#
# async def index_handler(request):
#     """Serves the main frontend page."""
#     # This is a very basic way to serve the file.
#     # In a real app, you would use a proper static file server.
#     file_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'index.html')
#     with open(file_path, 'r') as f:
#         content = f.read()
#     return web.Response(text=content, content_type='text/html')
#
# async def static_handler(request):
#     """Serves static files like worker.js."""
#     file_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', request.path)
#     if not os.path.exists(file_path):
#         raise web.HTTPNotFound()
#
#     with open(file_path, 'r') as f:
#         content = f.read()
#
#     mime_type = 'application/javascript' if request.path.endswith('.js') else 'text/plain'
#     return web.Response(text=content, content_type=mime_type)
#
# async def on_shutdown(app):
#     """Closes all peer connections on shutdown."""
#     coros = [pc.close() for pc in pcs]
#     await asyncio.gather(*coros)
#     pcs.clear()
#
# if __name__ == "__main__":
#     app = web.Application()
#     app.router.add_get("/", index_handler)
#     app.router.add_post("/offer", offer_handler)
#     app.router.add_get("/worker.js", static_handler)
#     app.on_shutdown.append(on_shutdown)
#     web.run_app(app, host="0.0.0.0", port=8000)


#!/usr/bin/env python3
import asyncio
import os
import json
import logging
import cv2
import numpy as np
import onnxruntime as ort
import time
import socket
import signal

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

# --- Constants ---
MODEL_PATH = 'models/yolov8n.onnx'
MODE = os.environ.get('MODE', 'wasm')  # set to 'server' to run inference here
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# COCO class names list
CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webrtc-server")

# --- Global State ---
pcs = set()
# single frame queue (shared). If you expect multiple simultaneous peers, consider per-pc queue.
FRAME_QUEUE = asyncio.Queue(maxsize=2)

# --- ONNX Runtime Setup ---
session = None
if MODE == 'server':
    try:
        session = ort.InferenceSession(MODEL_PATH)
        logger.info(f"ONNX model loaded successfully from {MODEL_PATH}.")
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")
        session = None
else:
    logger.info("Running in wasm mode (server will not run inference). Set MODE=server to enable server-side inference")

# --- Function to get Local IP ---
def get_local_ip():
    """Finds the local IP address of the machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

LOCAL_IP = get_local_ip()

# --- Image Processing and Inference ---
def preprocess(frame):
    """
    Expects an aiortc VideoFrame. Convert to ndarray (BGR), resize and normalize.
    Returns input tensor and original width/height.
    """
    # frame.to_ndarray returns RGB by default; explicitly request BGR if available.
    img = frame.to_ndarray(format="bgr24")
    original_height, original_width = img.shape[:2]
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, (INPUT_WIDTH, INPUT_HEIGHT))
    input_img = input_img.transpose(2, 0, 1)
    input_img = input_img[np.newaxis, :, :, :].astype(np.float32) / 255.0
    return input_img, original_width, original_height

def postprocess(outputs, original_width, original_height):
    """
    Postprocess outputs from YOLOv8 ONNX export.
    This function expects outputs to be an array-like of shape (1, N, 85) depending on export.
    Adjust indexing if your model uses different output layout.
    """
    # Normalize to shape (N, 85) then transpose if necessary. Many YOLOv8 exports give (1, N, 85).
    out = np.squeeze(outputs[0])
    if out.ndim == 1:
        # nothing detected
        return []
    # out shape -> (N, 85) where [x, y, w, h, conf, cls0, cls1, ...]
    boxes, scores, class_ids = [], [], []
    x_factor = original_width / INPUT_WIDTH
    y_factor = original_height / INPUT_HEIGHT

    for row in out:
        # row: [x, y, w, h, conf, class_scores...]
        conf = float(row[4])
        if conf < CONF_THRESHOLD:
            continue
        class_scores = row[5:]
        class_id = int(np.argmax(class_scores))
        class_score = float(class_scores[class_id]) * conf  # combined score
        if class_score < SCORE_THRESHOLD:
            continue

        x, y, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        left = int((x - w / 2) * x_factor)
        top = int((y - h / 2) * y_factor)
        width = int(w * x_factor)
        height = int(h * y_factor)

        boxes.append([left, top, width, height])
        scores.append(class_score)
        class_ids.append(class_id)

    detections = []
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, scores, SCORE_THRESHOLD, NMS_THRESHOLD)
        if len(indices) > 0:
            # indices can be array of shape (K,1) depending on OpenCV version
            if hasattr(indices, "flatten"):
                idxs = indices.flatten()
            else:
                idxs = [int(i[0]) for i in indices]
            for i in idxs:
                box = boxes[i]
                xmin = max(0.0, box[0] / original_width)
                ymin = max(0.0, box[1] / original_height)
                xmax = min(1.0, (box[0] + box[2]) / original_width)
                ymax = min(1.0, (box[1] + box[3]) / original_height)
                detections.append({
                    "label": CLASS_NAMES[class_ids[i]] if class_ids[i] < len(CLASS_NAMES) else str(class_ids[i]),
                    "score": float(scores[i]),
                    "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax
                })
    return detections

# --- Worker tasks ---
async def frame_processor_task(pc_id, data_channel_container, stop_event):
    """
    Pop frames from the global FRAME_QUEUE, run inference (when MODE == 'server'),
    and send JSON results over data channel (if available).
    """
    logger.info(f"[{pc_id}] Frame processor started")
    try:
        while not stop_event.is_set():
            try:
                frame = await asyncio.wait_for(FRAME_QUEUE.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            capture_ts = int(time.time() * 1000)
            detections = []
            if MODE == 'server' and session is not None:
                try:
                    recv_ts = int(time.time() * 1000)
                    input_data, original_width, original_height = preprocess(frame)
                    outputs = session.run(None, {'images': input_data})  # adjust input name if necessary
                    detections = postprocess(outputs, original_width, original_height)
                    inference_ts = int(time.time() * 1000)
                except Exception as e:
                    logger.exception(f"[{pc_id}] Inference error: {e}")
                    detections = []
            result = {
                "frame_id": capture_ts,
                "capture_ts": capture_ts,
                "detections": detections
            }
            # send via data channel if available and open
            dc = data_channel_container.get('dc')
            if dc is not None and dc.readyState == "open":
                try:
                    dc.send(json.dumps(result))
                except Exception as e:
                    logger.exception(f"[{pc_id}] Failed to send on data channel: {e}")

            FRAME_QUEUE.task_done()
    except asyncio.CancelledError:
        logger.info(f"[{pc_id}] Frame processor cancelled")
    finally:
        logger.info(f"[{pc_id}] Frame processor stopped")

async def track_receiver_task(pc_id, track, stop_event):
    """
    Receive frames from incoming track and push them into FRAME_QUEUE (dropping old frames if queue full).
    """
    logger.info(f"[{pc_id}] Track receiver started for kind={track.kind}")
    try:
        while not stop_event.is_set():
            frame = await track.recv()
            # drop the oldest if queue full
            if FRAME_QUEUE.full():
                try:
                    _ = FRAME_QUEUE.get_nowait()
                    FRAME_QUEUE.task_done()
                except Exception:
                    pass
            await FRAME_QUEUE.put(frame)
    except asyncio.CancelledError:
        logger.info(f"[{pc_id}] Track receiver cancelled")
    except Exception as e:
        logger.exception(f"[{pc_id}] Error receiving track: {e}")
    finally:
        logger.info(f"[{pc_id}] Track receiver stopped")

# --- WebRTC and HTTP Server Setup ---
async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)
    pc_id = f"PC-{id(pc)}"
    logger.info(f"[{pc_id}] Created peer connection")

    # container to hold data channel (set when ondatachannel fires)
    data_channel_container = {"dc": None}

    # stop event for tasks
    stop_event = asyncio.Event()
    tasks = []

    @pc.on("datachannel")
    def on_datachannel(channel):
        logger.info(f"[{pc_id}] DataChannel created: {channel.label}")

        @channel.on("message")
        def on_message(message):
            # If client sends control messages, handle here
            logger.debug(f"[{pc_id}] DataChannel message: {message}")

        data_channel_container["dc"] = channel

    @pc.on("track")
    def on_track(track):
        logger.info(f"[{pc_id}] Track received: kind={track.kind}")
        if track.kind == "video":
            # start receiver and processor tasks
            receiver_task = asyncio.create_task(track_receiver_task(pc_id, track, stop_event))
            processor_task = asyncio.create_task(frame_processor_task(pc_id, data_channel_container, stop_event))
            tasks.extend([receiver_task, processor_task])

            @track.on("ended")
            async def on_ended():
                logger.info(f"[{pc_id}] Track ended")
                stop_event.set()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"[{pc_id}] Connection state: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed", "disconnected"):
            stop_event.set()
            # cancel tasks
            for t in tasks:
                try:
                    t.cancel()
                except Exception:
                    pass
            await pc.close()
            pcs.discard(pc)
            logger.info(f"[{pc_id}] Peer connection closed and cleaned up")

    # set remote description and create answer
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    res = {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }
    logger.info(f"[{pc_id}] Sending answer")
    return web.json_response(res)

# Root index (optional)
async def index(request):
    return web.Response(content_type="text/plain", text="WebRTC VLM server is running.\n")

# Shutdown handler to close all peer connections
async def on_shutdown(app):
    logger.info("Shutting down, closing peer connections...")
    coros = []
    for pc in list(pcs):
        try:
            coros.append(pc.close())
        except Exception:
            pass
    if coros:
        await asyncio.gather(*coros)

def main():
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.on_shutdown.append(on_shutdown)

    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting server at http://{LOCAL_IP}:{port}")
    web.run_app(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted -- exiting")
