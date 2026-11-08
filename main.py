#!/usr/bin/env python3
"""
EDM AI DJ Deck - Main Entrypoint
Touchless spatial MIDI controller for Arduino Uno Q + Logitech Brio 100.

Usage examples:
  python main.py                              # Run with camera 0, virtual MIDI port, and HUD window
  python main.py --camera 1 --width 640       # Select external USB camera index 1
  python main.py --headless                   # Run without GUI window (ideal for headless Uno Q)
  python main.py --udp 192.168.1.50:5005      # Stream MIDI over Wi-Fi UDP to Host Laptop
  python main.py --list-midi                  # List detected MIDI output ports
"""

import argparse
import platform
import sys
import time

import cv2

import config
from gesture_controller import EDMDeckController
from hud_renderer import draw_hud
from midi_engine import MidiEngine


def open_low_latency_camera(camera_index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    """
    Open the Logitech Brio 100 (or system camera) configured for maximum FPS and minimum buffer latency.
    Uses V4L2 on Linux (Arduino Uno Q) and AVFoundation on macOS.
    """
    system = platform.system()
    if system == "Linux":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
        # Use MJPG fourcc on Linux V4L2 so USB 2.0/3.0 hubs deliver full 60fps at 480p/720p
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    elif system == "Darwin":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(camera_index)

    # Minimize internal frame queue to 1 frame for lowest latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    return cap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDM AI DJ Deck - Spatial Hand Tracking MIDI Controller")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=config.FRAME_WIDTH, help="Capture width (default: 640)")
    parser.add_argument("--height", type=int, default=config.FRAME_HEIGHT, help="Capture height (default: 480)")
    parser.add_argument("--fps", type=int, default=config.TARGET_FPS, help="Target camera FPS (default: 60)")
    parser.add_argument("--midi-port", type=str, default=config.MIDI_PORT_NAME, help="MIDI output/virtual port name")
    parser.add_argument("--channel", type=int, default=config.MIDI_CHANNEL, help="MIDI channel 0-15 (default: 0)")
    parser.add_argument("--udp", type=str, default=None, help="Optional host:port to stream UDP packets to Arduino Uno Q over Wi-Fi")
    parser.add_argument("--serial", type=str, default="auto", help="Arduino Uno Q USB-C serial port (default: 'auto' detect)")
    parser.add_argument("--no-synth", action="store_true", help="Disable built-in standalone EDM synth audio (MIDI-only mode)")
    parser.add_argument("--headless", action="store_true", help="Disable OpenCV GUI window for ultra-low-latency headless runs")
    parser.add_argument("--list-midi", action="store_true", help="List available system MIDI output ports and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_midi:
        ports = MidiEngine.list_ports()
        print("Available MIDI Output Ports:")
        if not ports:
            print("  (No existing hardware/network ports found; virtual port will be created on start)")
        for idx, name in enumerate(ports):
            print(f"  [{idx}] {name}")
        return

    print("=== Initializing EDM AI DJ Deck (MacBook Camera + Standalone Synth + Uno Q Bridge) ===")
    midi = MidiEngine(
        port_name=args.midi_port,
        channel=args.channel,
        udp_target=args.udp,
        enable_internal_synth=not args.no_synth,
        serial_port=args.serial,
    )
    controller = EDMDeckController(midi=midi)

    cap = open_low_latency_camera(args.camera, args.width, args.height, args.fps)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open camera at index {args.camera}. Check USB-C hub connection.")
        controller.close()
        midi.close()
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[CAMERA] Opened camera {args.camera} at {actual_w}x{actual_h} @ {actual_fps:.0f} FPS")
    print("[READY] Move Right Hand to hit Drum Pads | Left Hand for Filter/Reverb & Fist DROP. Press 'q' to exit.")

    prev_time = time.perf_counter()
    fps_ema = float(args.fps)

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                continue

            # Flip horizontally for intuitive mirror-mode interaction
            frame = cv2.flip(frame, 1)

            # Run MediaPipe Hand Tracking + EDM Hitbox/FX Logic
            detected_hands = controller.process_frame(frame)

            # Stream real-time 12-band FFT Spectrum & Beat Pulse to Arduino Uno Q LED Matrix
            midi.sync_visualizer()

            # Calculate smoothed FPS
            now = time.perf_counter()
            dt = max(1e-6, now - prev_time)
            prev_time = now
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt)

            if not args.headless:
                hud_frame = draw_hud(frame, controller, detected_hands, fps_ema)
                cv2.imshow("EDM AI DJ Deck - Arduino Uno Q", hud_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                    break
                elif key == ord(" ") and midi.synth is not None:
                    state = midi.synth.toggle_synth_drone()
                    midi.last_log_message = f"SYNTH: {'ON' if state else 'MUTED'}"
                elif key in (ord("1"), ord("2"), ord("3"), ord("4")) and midi.synth is not None:
                    p_name = midi.synth.set_preset(key - ord("1"))
                    midi.last_log_message = f"PRESET -> {p_name}"
                elif key in (9, ord("p"), ord("P")) and midi.synth is not None:  # TAB or P
                    p_name = midi.synth.next_preset()
                    midi.last_log_message = f"PRESET -> {p_name}"
                elif key in (ord("-"), ord("_"), ord("[")):
                    k_name = controller.shift_key(-1)
                    midi.last_log_message = f"KEY TRANSPOSE -> {k_name} Major"
                elif key in (ord("="), ord("+"), ord("]")):
                    k_name = controller.shift_key(1)
                    midi.last_log_message = f"KEY TRANSPOSE -> {k_name} Major"
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        controller.close()
        midi.close()
        print("[INFO] Clean shutdown complete.")


if __name__ == "__main__":
    main()
