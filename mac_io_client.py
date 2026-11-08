#!/usr/bin/env python3
"""
MacBook Dumb I/O Client (`mac_io_client.py`)
Runs ONLY the MacBook's FaceTime Camera capture, HUD display window, and Speaker playback!
All AI Hand Tracking, Gesture Hitbox math, 128 BPM EDM Synthesis, Low-Pass Filter/Reverb DSP,
and 8x12 LED Matrix control happen on the Arduino Uno Q (`uno_q_server.py`).

Works over:
  1. Single USB-C Cable (via `adb forward tcp:5001 tcp:5001 && adb forward tcp:5002 tcp:5002` -> connects to `127.0.0.1`)
  2. USB-C Ethernet / Wi-Fi (connects to `<UNO_Q_IP_ADDRESS>`)

Usage on MacBook:
  python mac_io_client.py --host 127.0.0.1          # When connected via USB-C ADB tunnel
  python mac_io_client.py --host 192.168.1.100      # When connected via Wi-Fi / USB-LAN
"""

import argparse
from collections import deque
import json
import os
import socket
import struct
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False

import config
from main import open_low_latency_camera
from uno_q_bridge import UnoQBridge


def recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    """Read exactly n bytes from a TCP socket, or return None on EOF."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


class MacSpeakerReceiver:
    """Receives live synthesized int16 PCM audio from the Arduino Uno Q and plays it on MacBook speakers."""

    def __init__(self, host: str, port: int, sample_rate: int = 44100, uno_bridge: Optional[UnoQBridge] = None):
        self.host = host
        self.port = port
        self.sr = sample_rate
        self.uno_bridge = uno_bridge
        self.audio_queue: deque = deque(maxlen=24)
        self.leftover = np.zeros(0, dtype=np.float32)
        self.stop_event = threading.Event()
        self.stream: Optional["sd.OutputStream"] = None

    def start(self) -> None:
        if SD_AVAILABLE:
            self.stream = sd.OutputStream(
                samplerate=self.sr,
                blocksize=512,
                channels=2,
                dtype="float32",
                latency="low",
                callback=self._audio_callback,
            )
            self.stream.start()
        threading.Thread(target=self._network_reader, daemon=True).start()

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        needed = frames
        out_mono = np.zeros(frames, dtype=np.float32)
        write_pos = 0

        while needed > 0:
            if len(self.leftover) > 0:
                take = min(needed, len(self.leftover))
                out_mono[write_pos : write_pos + take] = self.leftover[:take]
                self.leftover = self.leftover[take:]
                write_pos += take
                needed -= take
            elif self.audio_queue:
                self.leftover = self.audio_queue.popleft()
            else:
                break

        outdata[:, 0] = out_mono
        outdata[:, 1] = out_mono

    def _network_reader(self) -> None:
        while not self.stop_event.is_set():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.connect((self.host, self.port))
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"[MAC AUDIO] Connected to Arduino Uno Q Synth Audio stream at {self.host}:{self.port}")
                while not self.stop_event.is_set():
                    hdr = recv_exact(sock, 6)
                    if hdr is None:
                        break
                    byte_len, meta_len = struct.unpack("!IH", hdr)
                    meta_bytes = recv_exact(sock, meta_len) if meta_len > 0 else b""
                    pcm_bytes = recv_exact(sock, byte_len)
                    if pcm_bytes is None:
                        break
                    if self.uno_bridge is not None and meta_bytes:
                        try:
                            meta_str = meta_bytes.decode("ascii")
                            parts = meta_str.split(":")
                            if len(parts) >= 3:
                                bars = [int(ch) for ch in parts[0][:12]]
                                beat = parts[1] == "1"
                                drop = parts[2] == "1"
                                self.uno_bridge.send_drop_state(drop)
                                self.uno_bridge.send_visualizer_frame(bars, beat)
                        except Exception:
                            pass
                    samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
                    self.audio_queue.append(samples)
            except Exception:
                time.sleep(0.5)
            finally:
                sock.close()

    def close(self) -> None:
        self.stop_event.set()
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="MacBook Camera & Speaker I/O Client for Arduino Uno Q DJ Deck")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Arduino Uno Q IP (127.0.0.1 for USB-C ADB forward)")
    parser.add_argument("--video-port", type=int, default=5001, help="Arduino Uno Q video TCP port (default: 5001)")
    parser.add_argument("--audio-port", type=int, default=5002, help="Arduino Uno Q audio TCP port (default: 5002)")
    parser.add_argument("--camera", type=int, default=0, help="MacBook camera index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Capture height (default: 480)")
    parser.add_argument("--jpeg-quality", type=int, default=75, help="Camera stream JPEG quality (default: 75)")
    args = parser.parse_args()

    print(f"=== MacBook Camera & Speaker I/O Client -> Arduino Uno Q ({args.host}) ===")
    speaker = MacSpeakerReceiver(args.host, args.audio_port, uno_bridge=None)
    speaker.start()

    cap = open_low_latency_camera(args.camera, args.width, args.height, 60)
    if not cap.isOpened():
        print("[ERROR] Could not open MacBook camera.")
        speaker.close()
        sys.exit(1)

    vid_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[MAC CAMERA] Connecting to Arduino Uno Q at {args.host}:{args.video_port} ...")
    try:
        vid_sock.connect((args.host, args.video_port))
        vid_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("[MAC CAMERA] Connected! Streaming camera frames to Arduino Uno Q...")
    except Exception as exc:
        print(f"[ERROR] Could not connect to Arduino Uno Q ({args.host}:{args.video_port}): {exc}")
        print("Make sure `python3 uno_q_server.py` is running on the Arduino Uno Q!")
        cap.release()
        speaker.close()
        sys.exit(1)

    pending_key = 0
    mp_hands = None
    mp_landmarker = None
    if MP_AVAILABLE and hasattr(mp, "solutions"):
        mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_NUM_HANDS,
            model_complexity=config.MODEL_COMPLEXITY,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
    elif MP_AVAILABLE and os.path.isfile("hand_landmarker.task"):
        base_opts = mp_python.BaseOptions(model_asset_path="hand_landmarker.task")
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=config.MAX_NUM_HANDS,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        mp_landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        print("[MAC CAMERA] MediaPipe Tasks HandLandmarker ready!")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            # Mirror horizontally and resize to 640x480 for fast USB-C/Wi-Fi transport
            frame = cv2.flip(frame, 1)
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height))

            lm_json_bytes = b""
            if mp_hands is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res = mp_hands.process(rgb)
                hand_dict = {}
                if res.multi_hand_landmarks and res.multi_handedness:
                    for hlms, hness in zip(res.multi_hand_landmarks, res.multi_handedness):
                        side = hness.classification[0].label
                        pts = [[round(lm.x, 4), round(lm.y, 4)] for lm in hlms.landmark]
                        if side not in ("Left", "Right"):
                            side = "Left" if pts[0][0] < 0.5 else "Right"
                        hand_dict[side] = {"lms": pts}
                lm_json_bytes = json.dumps(hand_dict).encode("utf-8")
            elif mp_landmarker is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = mp_landmarker.detect(mp_img)
                hand_dict = {}
                if res.hand_landmarks:
                    for hlms in res.hand_landmarks:
                        pts = [[round(lm.x, 4), round(lm.y, 4)] for lm in hlms]
                        side = "Left" if pts[0][0] < 0.5 else "Right"
                        if side in hand_dict:
                            side = "Right" if side == "Left" else "Left"
                        hand_dict[side] = {"lms": pts}
                lm_json_bytes = json.dumps(hand_dict).encode("utf-8")

            ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            if not ok:
                continue
            payload = enc.tobytes()

            # Send [4-byte frame len][2-byte landmark len][1-byte keycode][landmark JSON][JPEG bytes] to Arduino Uno Q
            vid_sock.sendall(
                struct.pack("!IHB", len(payload), len(lm_json_bytes), pending_key)
                + lm_json_bytes
                + payload
            )
            pending_key = 0

            # Receive processed HUD frame back from Arduino Uno Q
            resp_hdr = recv_exact(vid_sock, 4)
            if resp_hdr is None:
                break
            (hud_len,) = struct.unpack("!I", resp_hdr)
            hud_bytes = recv_exact(vid_sock, hud_len)
            if hud_bytes is None:
                break

            hud_frame = cv2.imdecode(np.frombuffer(hud_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if hud_frame is not None:
                cv2.imshow("EDM AI DJ Deck (Running on Arduino Uno Q)", hud_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key != 255:
                pending_key = key

    except KeyboardInterrupt:
        pass
    finally:
        vid_sock.close()
        cap.release()
        cv2.destroyAllWindows()
        speaker.close()
        print("[MAC CLIENT] Disconnected cleanly.")


if __name__ == "__main__":
    main()
