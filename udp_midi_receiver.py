#!/usr/bin/env python3
"""
Host Laptop Wi-Fi UDP MIDI Receiver -> Local Virtual MIDI Port
Run this on the Host Laptop (Mac/Windows running Ableton Live or FL Studio)
when the Arduino Uno Q streams MIDI over Wi-Fi via `python main.py --udp <LAPTOP_IP>:5005`.
"""

import argparse
import socket
import mido


def main() -> None:
    parser = argparse.ArgumentParser(description="Receive UDP MIDI from Arduino Uno Q and forward to DAW Virtual MIDI Port")
    parser.add_argument("--bind", type=str, default="0.0.0.0", help="Bind IP address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5005, help="UDP listen port (default: 5005)")
    parser.add_argument("--midi-port", type=str, default="EDM AI DJ Deck (Wi-Fi)", help="Virtual MIDI Port name for DAW")
    args = parser.parse_args()

    out_port = mido.open_output(args.midi_port, virtual=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))

    print(f"[RECEIVER] Listening for Uno Q UDP MIDI on {args.bind}:{args.port}")
    print(f"[RECEIVER] Forwarding to Virtual DAW MIDI Port: '{args.midi_port}'")

    try:
        while True:
            data, addr = sock.recvfrom(64)
            if not data:
                continue
            try:
                msg = mido.Message.from_bytes(list(data))
                out_port.send(msg)
                print(f"[{addr[0]}] -> {msg}")
            except Exception as exc:
                print(f"[WARN] Malformed MIDI packet: {exc}")
    except KeyboardInterrupt:
        print("\n[RECEIVER] Shutting down.")
    finally:
        out_port.close()
        sock.close()


if __name__ == "__main__":
    main()
