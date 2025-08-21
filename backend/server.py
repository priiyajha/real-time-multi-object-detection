import asyncio
import json
import logging
import os
import socket
import platform
import subprocess
import time
from typing import List

from aiohttp import web
from pathlib import Path
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
    """Attempt to find a private LAN IPv4 usable by phone.
    Tries multiple strategies to avoid returning reserved addresses like 192.0.0.2.
    """
    def is_private(ip: str) -> bool:
        if ":" in ip:
            return False
        if ip.startswith("10."):
            return True
        if ip.startswith("192.168."):
            return True
        if ip.startswith("172."):
            try:
                second = int(ip.split(".")[1])
                return 16 <= second <= 31
            except Exception:
                return False
        return False

    # 1) Default route trick
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if is_private(ip):
            return ip
    except Exception:
        pass

    # 2) macOS: try networksetup to find Wi‑Fi device, then ipconfig
    try:
        if platform.system() == "Darwin":
            try:
                hw = subprocess.check_output(["networksetup", "-listallhardwareports"], stderr=subprocess.DEVNULL).decode()
                lines = hw.splitlines()
                device = None
                for i, line in enumerate(lines):
                    if "Hardware Port: Wi-Fi" in line or "Hardware Port: WLAN" in line:
                        # Next lines contain Device: enX
                        for j in range(i+1, min(i+5, len(lines))):
                            if lines[j].strip().startswith("Device: "):
                                device = lines[j].split(":", 1)[1].strip()
                                break
                        break
                candidates = [device] if device else []
                candidates += ["en0", "en1", "en2"]
            except Exception:
                candidates = ["en0", "en1", "en2"]

            for iface in candidates:
                if not iface:
                    continue
                try:
                    out = subprocess.check_output(["ipconfig", "getifaddr", iface], stderr=subprocess.DEVNULL).decode().strip()
                    if out and is_private(out):
                        return out
                except Exception:
                    continue
    except Exception:
        pass

    # 3) Linux: hostname -I
    try:
        out = subprocess.check_output(["bash", "-lc", "hostname -I || true"], stderr=subprocess.DEVNULL).decode()
        for token in out.split():
            if is_private(token):
                return token
    except Exception:
        pass

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
    
    async def metrics(request: web.Request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid json"}, status=400)

        # Write metrics.json into backend directory (mapped in Docker)
        try:
            backend_dir = Path(__file__).resolve().parent
            metrics_path = backend_dir / "metrics.json"
            metrics_path.write_text(json.dumps(payload, indent=2))
            return web.json_response({"status": "ok", "path": str(metrics_path)})
        except Exception as e:
            logger.exception("Failed to write metrics.json")
            return web.json_response({"error": str(e)}, status=500)

    app.router.add_post("/metrics", metrics)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8080)
    logger.info("Server listening on http://0.0.0.0:8080")
    await site.start()
    # Keep the server running
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
