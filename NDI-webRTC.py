#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Sat Dec 14 08:05:18 2024
Coverts a NDI video stream to WebRTC
Runs on Mac under Sequoia 15.1
Needs Python 3.8
Needs NDI SDK https://ndi.video/for-developers/ndi-sdk/download/
Needs the libraries below
Based on examples at https://github.com/buresu/ndi-python
With help of ChatGPT o1

@author: wjst
"""

import sys
import numpy as np
import time
import logging
import asyncio
import NDIlib as ndi
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from av import VideoFrame
from fractions import Fraction

# logging.basicConfig(level=logging.INFO)

def ndi_connect(source_name=None):
    if not ndi.initialize():
        raise RuntimeError("Failed to initialize NDI.")

    ndi_find = ndi.find_create_v2()
    if not ndi_find:
        ndi.destroy()
        raise RuntimeError("Failed to create NDI finder.")

    # Find sources
    sources = []
    while not sources:
        print('Looking for sources ...')
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

    print(f"Using source: {chosen_source.ndi_name}")

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
    time.sleep(2)
    ndi.find_destroy(ndi_find)
    start_time = time.time()

    return ndi_recv, start_time

def ndi_receive_frame(ndi_recv):
    """
    Attempt to receive one frame from NDI.
    Returns frame_data (h, w, 3) if successful, otherwise None.
    """
    t, v, a, _ = ndi.recv_capture_v2(ndi_recv, timeout_in_ms=5000)
    if t == ndi.FRAME_TYPE_VIDEO and v is not None:
        logging.info(f"Video data received ({v.xres}x{v.yres}).")
        frame_data = np.copy(v.data)
        # Remove alpha channel
        frame_data = np.delete(frame_data, 3, axis=2)
        ndi.recv_free_video_v2(ndi_recv, v)
        return frame_data
    else:
        logging.debug("No video frame received this attempt.")
        return None

class NDIVideoTrack(MediaStreamTrack):
    kind = "video"
    def __init__(self, ndi_recv, start_time):
        super().__init__()
        self.ndi_recv = ndi_recv
        self.start_time = start_time
        self.last_frame = None  # Will store the last successfully received NDI frame

    async def recv(self):
        now = time.time()

        # Try to get a new frame
        frame_data = ndi_receive_frame(self.ndi_recv)
        if frame_data is not None:
            # Got a new NDI frame
            self.last_frame = frame_data
        else:
            # No new frame received this time
            if self.last_frame is None:
                # We have never received a frame before, return a black frame
                frame_data = np.zeros((480,640,3), dtype=np.uint8)
            else:
                # We have a last_frame, reuse it
                frame_data = self.last_frame

        frame = VideoFrame.from_ndarray(frame_data, format="bgr24")
        frame.pts = int((now - self.start_time)*90000)
        frame.time_base = Fraction(1,90000)

        return frame

async def offer(request):
    params = await request.json()
    if "sdp" not in params or "type" not in params:
        return web.Response(status=400, text="Invalid SDP")

    pc = RTCPeerConnection()
    pc.addTransceiver("video", direction="recvonly")

    logging.info("Client connected")

    global ndi_recv, ndi_start_time
    ndi_track = NDIVideoTrack(ndi_recv, ndi_start_time)
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
    return web.Response(
        content_type="text/html",
        text="""<!DOCTYPE html>
<html>
<head><title>NDI to WebRTC</title></head>
<body>
  <h1>NDI Video</h1>
  <video id="video" autoplay playsinline></video>
  <button id="playButton">Play</button>
  <script>
    const pc = new RTCPeerConnection();
    const video = document.getElementById('video');
    pc.ontrack = (event) => { video.srcObject = event.streams[0]; };
    async function negotiate() {
      pc.addTransceiver('video', { direction: 'recvonly' });
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      const response = await fetch('/offer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(pc.localDescription)
      });
      const answer = await response.json();
      await pc.setRemoteDescription(answer);
    }
    document.getElementById('playButton').onclick = negotiate;
  </script>
</body>
</html>
"""
    )

async def cleanup(app):
    logging.info("Server shutting down...")

async def main():
    source_name = None
    global ndi_recv, ndi_start_time
    ndi_recv, ndi_start_time = ndi_connect(source_name)

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.on_shutdown.append(cleanup)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 8080)
    logging.info("Running on http://127.0.0.1:8080")
    await site.start()
    await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Interrupted")
