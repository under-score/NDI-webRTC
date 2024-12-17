#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Sat Dec 14 08:05:18 2024
Converts a NDI video stream to WebRTC
Runs on Mac under Sequoia 15.1
Needs Python 3.8
Needs NDI SDK https://ndi.video/for-developers/ndi-sdk/download/
Needs the libraries sys, numpy, time, logging, asyncio, NDIlib, aiohttp, aiortc, av and fractions
Based on examples at https://github.com/buresu/ndi-python
With help of ChatGPT o1

@author: wjst
"""

import sys
import numpy as np
import logging
import asyncio
import NDIlib as ndi
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from av import VideoFrame
from fractions import Fraction

__version__ = "1.0.0"

logging.basicConfig(level=logging.INFO)

async def ndi_connect(source_name=None):
    """
    Initialize NDI, find the source, and connect a receiver.
    Returns the ndi_recv object.
    """
    if not ndi.initialize():
        raise RuntimeError("Failed to initialize NDI.")

    ndi_find = ndi.find_create_v2()
    if not ndi_find:
        ndi.destroy()
        raise RuntimeError("Failed to create NDI finder.")

    # Find sources
    sources = []
    while not sources:
        logging.info('Looking for NDI sources...')
        ndi.find_wait_for_sources(ndi_find, 5000)
        sources = ndi.find_get_current_sources(ndi_find)

    if not sources:
        ndi.find_destroy(ndi_find)
        ndi.destroy()
        raise RuntimeError("No NDI sources found.")

    for s in sources:
        logging.info(f"Available NDI Source: {s.ndi_name}")

    if source_name:
        chosen_source = next((src for src in sources if src.ndi_name == source_name), None)
        if not chosen_source:
            logging.warning(f"Specified NDI source '{source_name}' not found. Using first source.")
            chosen_source = sources[0]
    else:
        chosen_source = sources[0]

    logging.info(f"Using NDI source: {chosen_source.ndi_name}")

    ndi_recv_create = ndi.RecvCreateV3(
        color_format=ndi.RECV_COLOR_FORMAT_BGRX_BGRA,
        bandwidth=ndi.RECV_BANDWIDTH_LOWEST,
        allow_video_fields=False
    )

    ndi_recv = ndi.recv_create_v3(ndi_recv_create)
    if not ndi_recv:
        ndi.find_destroy(ndi_find)
        ndi.destroy()
        raise RuntimeError("Failed to create NDI receiver.")

    ndi.recv_connect(ndi_recv, chosen_source)
    # Give the receiver some time to lock onto the source
    await asyncio.sleep(2)
    ndi.find_destroy(ndi_find)
    logging.info(f"Connected to NDI source: {chosen_source.ndi_name}")

    return ndi_recv

async def ndi_receive_frames(ndi_recv, frame_queue):
    """
    Continuously receive frames from NDI and put them into an asyncio.Queue.
    """
    while True:
        t, v, a, _ = ndi.recv_capture_v2(ndi_recv, timeout_in_ms=1000)
        if t == ndi.FRAME_TYPE_VIDEO and v is not None:
            logging.info(f"Video data received ({v.xres}x{v.yres}).")
            frame_data = np.copy(v.data)
            # Remove alpha channel (4th channel)
            frame_data = np.delete(frame_data, 3, axis=2)
            ndi.recv_free_video_v2(ndi_recv, v)
            # Put the frame into the queue (overwrite if full)
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await frame_queue.put(frame_data)
        else:
            # No frame received, continue
            await asyncio.sleep(0.01)

class NDIVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, frame_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.last_frame = None

    async def recv(self):
        # Retrieve all available frames to get the latest
        while True:
            try:
                self.last_frame = self.frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        if self.last_frame is not None:
            frame_data = self.last_frame
        else:
            # No frame received yet, send black frame
            frame_data = np.zeros((480, 640, 3), dtype=np.uint8)

        # Create VideoFrame
        frame = VideoFrame.from_ndarray(frame_data, format="bgr24")
        now = time.time()
        frame.pts = int((now)*90000)
        frame.time_base = Fraction(1,90000)

        return frame

async def handle_offer(request, frame_queue):
    """
    Handle WebRTC offer from the client.
    """
    params = await request.json()
    if "sdp" not in params or "type" not in params:
        return web.Response(status=400, text="Invalid SDP")

    pc = RTCPeerConnection()
    pc.addTransceiver("video", direction="recvonly")

    logging.info("Client connected via WebRTC")

    ndi_track = NDIVideoTrack(frame_queue)
    pc.addTrack(ndi_track)

    @pc.on("icecandidate")
    async def on_icecandidate(candidate):
        if candidate:
            logging.info(f"ICE candidate: {candidate}")

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    )
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    })

async def index(request):
    """
    Serve the HTML client.
    """
    return web.Response(
        content_type="text/html",
        text="""<!DOCTYPE html>
<html>
<head><title>NDI to WebRTC</title></head>
<body>
  <h1>NDI Video Stream via WebRTC</h1>
  <video id="video" autoplay playsinline controls></video>
  <button id="playButton">Play</button>
  <script>
    const pc = new RTCPeerConnection();
    const video = document.getElementById('video');

    pc.ontrack = (event) => {
      video.srcObject = event.streams[0];
    };

    async function negotiate() {
      try {
        pc.addTransceiver('video', { direction: 'recvonly' });
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        const response = await fetch('/offer', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(pc.localDescription)
        });
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const answer = await response.json();
        await pc.setRemoteDescription(answer);
      } catch (error) {
        console.error('Negotiation failed:', error);
      }
    }

    document.getElementById('playButton').onclick = negotiate;
  </script>
</body>
</html>
"""
    )

async def cleanup(app, ndi_recv, frame_queue):
    """
    Cleanup resources on server shutdown.
    """
    logging.info("Server shutting down...")
    ndi.recv_destroy(ndi_recv)
    ndi.destroy()
    # Cancel background frame receiving task
    for task in asyncio.all_tasks():
        if task is not asyncio.current_task():
            task.cancel()
    # Optionally, drain the queue
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

async def main():
    # (1) NDI connect on startup with a preconfigured source name:
    # Replace "My NDI Source Name" with your actual source name or set to None to use the first source
    source_name = None  # e.g., "My NDI Source Name"
    ndi_recv = await ndi_connect(source_name)

    # Create a queue to hold frames
    frame_queue = asyncio.Queue(maxsize=10)

    # Start background task to receive frames
    asyncio.create_task(ndi_receive_frames(ndi_recv, frame_queue))

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", lambda request: handle_offer(request, frame_queue))
    app.on_shutdown.append(lambda app: asyncio.create_task(cleanup(app, ndi_recv, frame_queue)))

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 8080)
    logging.info("Running on http://127.0.0.1:8080")
    await site.start()

    # Run indefinitely
    await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
