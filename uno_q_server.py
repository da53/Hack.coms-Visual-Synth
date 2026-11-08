#!/usr/bin/env python3
"""
EDM AI DJ Deck - ARDUINO UNO Q NATIVE SERVER (`uno_q_server.py`)
Runs 100% of the AI, Gesture Hitboxes, 128 BPM EDM Synth, Low-Pass Filter/Reverb DSP,
12-Band FFT Visualizer, and 8x12 LED Matrix ON THE ARDUINO UNO Q!

The MacBook connects via `mac_io_client.py` (over USB-C ADB port-forwarding or Wi-Fi)
solely to feed raw webcam frames to the Arduino and play the Arduino's synthesized audio.

Usage on Arduino Uno Q (Linux terminal / ssh / adb shell):
  python3 uno_q_server.py
"""

import argparse
import json
import socket
import struct
import threading
import time
from typing import Dict, Optional

import cv2
import numpy as np

import config
from gesture_controller import EDMDeckController, HandData
from hud_renderer import draw_hud
from midi_engine import MidiEngine


def recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    """Read exactly n bytes from a TCP socket, or return None on EOF."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def audio_stream_worker(
    midi: MidiEngine,
    host: str,
    audio_port: int,
    stop_event: threading.Event,
) -> None:
    """
    Runs the real-time EDM Synthesizer & Drum DSP on the Arduino Uno Q CPU (`44100 Hz`),
    updates the Arduino's 8x12 LED Matrix FFT spectrum, and streams raw int16 PCM audio
    packets over TCP to the MacBook speaker client.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, audio_port))
    srv.listen(1)
    srv.settimeout(1.0)
    print(f"[UNO-Q AUDIO] Synth Audio Server listening on {host}:{audio_port}")

    chunk_frames = 512
    sr = 44100
    chunk_duration = chunk_frames / float(sr)

    while not stop_event.is_set():
        try:
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[UNO-Q AUDIO] MacBook Speaker Client connected from {addr[0]}")
        except socket.timeout:
            continue
        except Exception:
            break

        next_tick = time.perf_counter()
        try:
            while not stop_event.is_set():
                if midi.synth is not None:
                    mono_f32 = midi.synth.render_mono_block(chunk_frames)
                    # Sync the Arduino Uno Q's onboard 8x12 LED Matrix with the live FFT & Beat Pulse
                    midi.sync_visualizer()
                else:
                    mono_f32 = np.zeros(chunk_frames, dtype=np.float32)

                # Convert float32 [-1..1] to compact int16 PCM bytes + 12-band FFT + Beat metadata
                pcm_int16 = (np.clip(mono_f32, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
                if midi.synth is not None:
                    bar_str = "".join(str(max(0, min(8, int(b)))) for b in midi.synth.vis_bars[:12])
                    beat_char = "1" if midi.synth.beat_pulse else "0"
                    drop_char = "1" if midi.synth.drop_mode else "0"
                else:
                    bar_str = "000000000000"
                    beat_char = "0"
                    drop_char = "0"
                vis_meta = f"{bar_str}:{beat_char}:{drop_char}".encode("ascii")
                header = struct.pack("!IH", len(pcm_int16), len(vis_meta))
                conn.sendall(header + vis_meta + pcm_int16)

                # Pace the synthesizer loop to real-time 44.1kHz clock
                next_tick += chunk_duration
                sleep_dt = next_tick - time.perf_counter()
                if sleep_dt > 0:
                    time.sleep(sleep_dt)
                elif sleep_dt < -0.08:
                    next_tick = time.perf_counter()
        except (BrokenPipeError, ConnectionResetError, OSError):
            print("[UNO-Q AUDIO] MacBook Speaker Client disconnected.")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    srv.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the entire EDM AI DJ Deck natively on the Arduino Uno Q")
    parser.add_argument("--bind", type=str, default="0.0.0.0", help="Bind IP on Arduino Uno Q (default: 0.0.0.0)")
    parser.add_argument("--video-port", type=int, default=5001, help="TCP port for Camera In / HUD Out (default: 5001)")
    parser.add_argument("--audio-port", type=int, default=5002, help="TCP port for Uno Q Synth Audio Out (default: 5002)")
    parser.add_argument("--serial", type=str, default="auto", help="Internal MCU serial port for 8x12 LED Matrix")
    parser.add_argument("--jpeg-quality", type=int, default=75, help="HUD return stream JPEG quality (default: 75)")
    args = parser.parse_args()

    print("=================================================================")
    print("  ARDUINO UNO Q - NATIVE EDM AI DJ DECK & SYNTH ENGINE SERVER")
    print("  All AI Hand Tracking, Hitboxes, DSP Synth & FFT run ON BOARD!")
    print("=================================================================")

    # Initialize Synth in network-streaming mode (renders on Uno Q CPU, streams PCM to MacBook)
    midi = MidiEngine(
        enable_internal_synth=True,
        serial_port=args.serial,
        use_local_audio_device=False,
    )
    controller = EDMDeckController(midi=midi)

    stop_event = threading.Event()
    audio_thread = threading.Thread(
        target=audio_stream_worker,
        args=(midi, args.bind, args.audio_port, stop_event),
        daemon=True,
    )
    audio_thread.start()

    vid_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    vid_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    vid_srv.bind((args.bind, args.video_port))
    vid_srv.listen(1)
    print(f"[UNO-Q VISION] Waiting for MacBook Camera stream on {args.bind}:{args.video_port} ...")

    try:
        while not stop_event.is_set():
            conn, addr = vid_srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[UNO-Q VISION] MacBook Camera Client connected from {addr[0]}")

            prev_time = time.perf_counter()
            fps_ema = 30.0

            try:
                while True:
                    # 1. Read incoming frame + optional landmark JSON + key press from MacBook
                    hdr = recv_exact(conn, 7)
                    if hdr is None:
                        break
                    frame_len, lm_len, key_code = struct.unpack("!IHB", hdr)
                    lm_bytes = recv_exact(conn, lm_len) if lm_len > 0 else b""
                    jpg_bytes = recv_exact(conn, frame_len)
                    if jpg_bytes is None:
                        break

                    # Handle remote key press (SPACE = toggle synth, 1-4 or TAB/P = switch synth preset)
                    if key_code == ord(" ") and midi.synth is not None:
                        state = midi.synth.toggle_synth_drone()
                        midi.last_log_message = f"SYNTH: {'ON' if state else 'MUTED'}"
                    elif key_code in (ord("1"), ord("2"), ord("3"), ord("4")) and midi.synth is not None:
                        p_name = midi.synth.set_preset(key_code - ord("1"))
                        midi.last_log_message = f"PRESET -> {p_name}"
                    elif key_code in (9, ord("p"), ord("P")) and midi.synth is not None:
                        p_name = midi.synth.next_preset()
                        midi.last_log_message = f"PRESET -> {p_name}"
                    elif key_code in (ord("-"), ord("_"), ord("[")):
                        k_name = controller.shift_key(-1)
                        midi.last_log_message = f"KEY TRANSPOSE -> {k_name} Major"
                    elif key_code in (ord("="), ord("+"), ord("]")):
                        k_name = controller.shift_key(1)
                        midi.last_log_message = f"KEY TRANSPOSE -> {k_name} Major"

                    # 2. Decode frame & run EDM DJ Logic + Synth Control ON THE ARDUINO UNO Q
                    np_arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                    if lm_bytes:
                        # Use 21-point hand coordinates if provided, and run Hitbox/FX/Drop math on Uno Q!
                        raw_dict = json.loads(lm_bytes.decode("utf-8"))
                        detected_hands: Dict[str, Optional[HandData]] = {"Left": None, "Right": None}
                        for side in ("Left", "Right"):
                            h_info = raw_dict.get(side)
                            if h_info:
                                lms = [tuple(pt) for pt in h_info["lms"]]
                                is_fist = controller._detect_fist(lms)
                                detected_hands[side] = HandData(
                                    label=side,
                                    index_tip=tuple(lms[8]),
                                    wrist=tuple(lms[0]),
                                    is_fist=is_fist,
                                    landmarks=lms,
                                )
                        controller.apply_external_hands(detected_hands)
                    else:
                        detected_hands = controller.process_frame(frame)

                    now = time.perf_counter()
                    dt = max(1e-6, now - prev_time)
                    prev_time = now
                    fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt)

                    # 3. Render DJ HUD overlay on the Arduino Uno Q and send back to MacBook for display
                    hud_frame = draw_hud(frame, controller, detected_hands, fps_ema)
                    ok, enc = cv2.imencode(
                        ".jpg",
                        hud_frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
                    )
                    if not ok:
                        continue
                    out_bytes = enc.tobytes()
                    conn.sendall(struct.pack("!I", len(out_bytes)) + out_bytes)

            except (BrokenPipeError, ConnectionResetError, OSError):
                print("[UNO-Q VISION] MacBook Camera Client disconnected.")
            finally:
                conn.close()

    except KeyboardInterrupt:
        print("\n[UNO-Q] Shutting down server...")
    finally:
        stop_event.set()
        vid_srv.close()
        controller.close()
        midi.close()


if __name__ == "__main__":
    main()
