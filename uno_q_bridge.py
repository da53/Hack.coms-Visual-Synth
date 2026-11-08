"""
Arduino Uno Q Hardware Bridge (USB-C Serial & Wi-Fi / Linux ADB Bridge).
Allows the MacBook (running the FaceTime camera & HUD) to communicate in real time
with the Arduino Uno Q over a single USB-C cable (or Wi-Fi UDP).

What the Arduino Uno Q does in this setup:
1. Receives compact real-time telemetry packets over USB-C Serial (`pyserial`) or Wi-Fi/USB-Ethernet UDP:
   - `HIT:<NOTE>:<VEL>\n`  (Kick=36, Snare=38, HiHat=42) -> Flashes Drum Quadrants on Uno Q LED Matrix / GPIOs
   - `FX:<FILTER_0_127>:<REVERB_0_127>\n` -> Draws real-time Filter Cutoff bar graph on Uno Q LED Matrix
   - `DROP:1\n` / `DROP:0\n` -> Triggers full-board strobe/drop animation on the Uno Q STM32U585 MCU
2. Also supports running `udp_midi_receiver.py` or `uno_q_companion.py` on the Uno Q Linux side.
"""

import glob
import socket
import time
from typing import Optional

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class UnoQBridge:
    """
    Non-blocking hardware bridge from MacBook -> Arduino Uno Q over USB-C Serial and/or UDP.
    Auto-detects Arduino Uno Q USB-C serial ports on macOS (`/dev/cu.usbmodem*`) or Linux (`/dev/ttyACM*`).
    """

    def __init__(
        self,
        serial_port: Optional[str] = "auto",
        baudrate: int = 115200,
        udp_target: Optional[str] = None,
    ):
        self.ser: Optional["serial.Serial"] = None
        self.udp_sock: Optional[socket.socket] = None
        self.udp_addr: Optional[tuple] = None
        self.last_fx_send_time: float = 0.0

        if serial_port:
            self._init_serial(serial_port, baudrate)

        if udp_target:
            self._init_udp(udp_target)

    def _init_serial(self, requested_port: str, baudrate: int) -> None:
        if not SERIAL_AVAILABLE:
            print("[UNO-Q] Tip: install 'pyserial' (`pip install pyserial`) for USB-C serial link to Arduino Uno Q.")
            return

        port_to_open = None
        if requested_port != "auto":
            port_to_open = requested_port
        else:
            # Auto-detect internal Uno Q Linux->MCU UART (/dev/ttyHS* or /dev/ttyRPMSG*)
            # OR USB-C serial port (/dev/cu.usbmodem* or /dev/ttyACM*)
            internal_uarts = glob.glob("/dev/ttyHS*") + glob.glob("/dev/ttyRPMSG*") + glob.glob("/dev/ttyMSM*")
            if internal_uarts:
                port_to_open = internal_uarts[0]
            elif SERIAL_AVAILABLE:
                ports = list(serial.tools.list_ports.comports())
                for p in ports:
                    desc = f"{p.device} {p.description} {p.manufacturer or ''}".lower()
                    if any(k in desc for k in ("arduino", "usbmodem", "stm32", "qualcomm", "ttyacm")):
                        port_to_open = p.device
                        break
            if not port_to_open:
                candidates = glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/ttyACM*")
                if candidates:
                    port_to_open = candidates[0]

        if not port_to_open:
            print("[UNO-Q] No Arduino USB-C serial device detected yet (running MacBook standalone; plug in Uno Q anytime).")
            return

        try:
            self.ser = serial.Serial(port_to_open, baudrate=baudrate, timeout=0, write_timeout=0)
            print(f"[UNO-Q] Connected to Arduino Uno Q over USB-C Serial: {port_to_open} @ {baudrate} baud")
        except Exception as exc:
            print(f"[UNO-Q] Could not open serial port {port_to_open}: {exc}")
            self.ser = None

    def _init_udp(self, udp_target: str) -> None:
        try:
            host, port_str = udp_target.split(":")
            self.udp_addr = (host, int(port_str))
            self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_sock.setblocking(False)
            print(f"[UNO-Q] Connected to Arduino Uno Q over Network UDP -> {host}:{port_str}")
        except Exception as exc:
            print(f"[UNO-Q] Invalid Uno Q UDP target '{udp_target}': {exc}")

    def _send_line(self, line: str) -> None:
        payload = (line.strip() + "\n").encode("ascii", errors="ignore")
        if self.ser is not None:
            try:
                self.ser.write(payload)
            except Exception:
                pass
        if self.udp_sock is not None and self.udp_addr is not None:
            try:
                self.udp_sock.sendto(payload, self.udp_addr)
            except Exception:
                pass

    def send_drum_hit(self, note: int, velocity: int) -> None:
        """Notify the Arduino Uno Q of a drum pad trigger (Kick=36, Snare=38, HiHat=42, Drop=35)."""
        self._send_line(f"HIT:{note}:{velocity}")

    def send_fx_state(self, filter_cc: int, reverb_cc: int) -> None:
        """Throttle FX state updates to ~30Hz so serial/LED matrix stays ultra-responsive."""
        now = time.perf_counter()
        if (now - self.last_fx_send_time) >= 0.033:
            self.last_fx_send_time = now
            self._send_line(f"FX:{filter_cc}:{reverb_cc}")

    def send_drop_state(self, active: bool) -> None:
        """Trigger or release the DROP animation on the Arduino Uno Q."""
        self._send_line(f"DROP:{1 if active else 0}")

    def send_visualizer_frame(self, bars: list, beat_pulse: bool) -> None:
        """
        Stream the 12-band FFT spectrum (each bar '0'..'8') and beat pulse ('1' or '0')
        to the Arduino Uno Q 8x12 LED Matrix (e.g., `VIS:765432211234:1`).
        """
        bar_str = "".join(str(max(0, min(8, int(b)))) for b in bars[:12])
        self._send_line(f"VIS:{bar_str}:{1 if beat_pulse else 0}")

    def close(self) -> None:
        if self.ser is not None:
            try:
                self._send_line("DROP:0")
                self.ser.close()
            except Exception:
                pass
        if self.udp_sock is not None:
            try:
                self.udp_sock.close()
            except Exception:
                pass
