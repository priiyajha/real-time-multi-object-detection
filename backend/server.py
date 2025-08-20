import asyncio
import json
import logging
import os
import socket
import time
from typing import List

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay

# --- Config ---
MODE = os.environ.get("MODE", "wasm")

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webrtc-bridge")

# --- Global state ---
pcs = set()
relay = MediaRelay()
publisher_video_track = None  # will hold the latest publisher track
frame_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
viewer_datachannels: List = []


async def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


async def track_receiver_task(track, stop_event: asyncio.Event):
    logger.info("Track receiver started")
    try:
        while not stop_event.is_set():
            frame = await track.recv()
            if frame_queue.full():
                try:
                    _ = frame_queue.get_nowait()
                    frame_queue.task_done()
                except Exception:
                    pass
            await frame_queue.put(frame)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.exception(f"receiver error: {e}")
    finally:
        logger.info("Track receiver stopped")


async def processor_task(stop_event: asyncio.Event):
    logger.info("Processor started")
    try:
        while not stop_event.is_set():
            try:
                frame = await asyncio.wait_for(frame_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            capture_ts = int(time.time() * 1000)
            recv_ts = capture_ts
            inference_ts = capture_ts
            # Dummy detections; in server mode you would run ONNX here
            detections = [
                {"label": "person", "score": 0.9, "xmin": 0.1, "ymin": 0.1, "xmax": 0.5, "ymax": 0.6}
            ]
            result = {
                "frame_id": capture_ts,
                "capture_ts": capture_ts,
                "recv_ts": recv_ts,
                "inference_ts": inference_ts,
                "detections": detections
            }
            # fanout to all viewer datachannels
            for dc in list(viewer_datachannels):
                try:
                    if dc.readyState == "open":
                        dc.send(json.dumps(result))
                except Exception as e:
                    logger.warning(f"send failed: {e}")
            frame_queue.task_done()
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Processor stopped")


async def offer(request):
    params = await request.json()
    role = params.get("role", "viewer")
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)
    stop_event = asyncio.Event()
    tasks: List[asyncio.Task] = []

    @pc.on("connectionstatechange")
    async def on_state():
        logger.info(f"PC state: {pc.connectionState} ({role})")
        if pc.connectionState in ("failed", "closed", "disconnected"):
            stop_event.set()
            for t in tasks:
                try:
                    t.cancel()
                except Exception:
                    pass
            await pc.close()
            pcs.discard(pc)

    @pc.on("datachannel")
    def on_dc(channel):
        logger.info(f"DataChannel: {channel.label} ({role})")
        if role == "viewer":
            viewer_datachannels.append(channel)
            @channel.on("close")
            def _():
                try:
                    viewer_datachannels.remove(channel)
                except ValueError:
                    pass

    if role == "publisher":
        @pc.on("track")
        def on_track(track):
            global publisher_video_track
            logger.info(f"Publisher track: {track.kind}")
            if track.kind == "video":
                publisher_video_track = relay.subscribe(track)
                tasks.append(asyncio.create_task(track_receiver_task(track, stop_event)))
                # ensure processor is running
                if not any(t.get_name() == "processor" for t in tasks):
                    pt = asyncio.create_task(processor_task(stop_event))
                    pt.set_name("processor")
                    tasks.append(pt)
    else:
        # viewer: attach publisher track if present
        if publisher_video_track is not None:
            pc.addTrack(publisher_video_track)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


async def env(request):
    ip = await get_local_ip()
    return web.json_response({"local_ip": ip, "mode": MODE})


async def main():
    @web.middleware
    async def cors_middleware(request, handler):
        if request.method == 'OPTIONS':
            resp = web.Response(status=204)
        else:
            resp = await handler(request)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
        resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return resp

    app = web.Application(middlewares=[cors_middleware])
    app.router.add_post("/offer", offer)
    app.router.add_get("/env", env)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8080)
    logger.info("Server listening on http://0.0.0.0:8080")
    await site.start()


if __name__ == "__main__":
    asyncio.run(main())
