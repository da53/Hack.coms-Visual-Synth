"""
MIDI output engine supporting:
1. Local Virtual MIDI port via `mido` + `python-rtmidi` (works directly with DAWs, BLE-MIDI bridges, or rtpMIDI/qmidinet).
2. Existing hardware/network MIDI port attachment (e.g., BlueZ BLE MIDI or ALSA/CoreMIDI network session).
3. Optional raw UDP MIDI packet streaming for Wi-Fi Uno Q -> Host Laptop low-latency bridging.
"""

import socket
import time
from typing import Dict, Optional

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False

import config
from synth_engine import StandaloneEDMSynth
from uno_q_bridge import UnoQBridge


class MidiEngine:
    """Manages standalone on-board EDM synth audio, Arduino Uno Q USB-C Bridge, AND MIDI Note/CC transmission."""

    def __init__(
        self,
        port_name: str = config.MIDI_PORT_NAME,
        channel: int = config.MIDI_CHANNEL,
        udp_target: Optional[str] = None,
        enable_internal_synth: bool = True,
        serial_port: Optional[str] = "auto",
        use_local_audio_device: bool = True,
    ):
        self.port_name = port_name
        self.channel = channel
        self.port = None
        self.udp_sock: Optional[socket.socket] = None
        self.udp_addr: Optional[tuple] = None

        # Built-in Standalone Audio Synth
        self.synth: Optional[StandaloneEDMSynth] = (
            StandaloneEDMSynth(use_local_audio_device=use_local_audio_device)
            if enable_internal_synth
            else None
        )

        # USB-C Serial / Network Bridge to Arduino Uno Q Hardware (STM32U585 LED Matrix / GPIOs)
        self.uno_bridge = UnoQBridge(serial_port=serial_port, udp_target=udp_target)

        # State tracking to prevent flooding the MIDI bus
        self.last_cc_values: Dict[int, int] = {}
        self.smoothed_cc: Dict[int, float] = {}
        self.last_log_message: str = "Standalone Synth + Uno Q Bridge Ready"

        if udp_target:
            self._init_udp(udp_target)

        self._init_midi_port()

    def _init_udp(self, udp_target: str) -> None:
        """Initialize optional Wi-Fi UDP MIDI socket (host:port)."""
        try:
            host, port_str = udp_target.split(":")
            self.udp_addr = (host, int(port_str))
            self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_sock.setblocking(False)
            print(f"[MIDI] Wi-Fi UDP MIDI streaming enabled -> {host}:{port_str}")
        except Exception as exc:
            print(f"[MIDI] Failed to parse UDP target '{udp_target}': {exc}")
            self.udp_sock = None

    def _init_midi_port(self) -> None:
        """Open a virtual MIDI port or attach to a matching system/BLE/network MIDI port."""
        if not MIDO_AVAILABLE:
            print("[MIDI] Warning: 'mido' / 'python-rtmidi' not installed. Running in dry-run HUD mode.")
            return

        try:
            available_outputs = mido.get_output_names()
            # Check if a matching port (like a BLE MIDI or Network Session) already exists
            matching = [name for name in available_outputs if self.port_name.lower() in name.lower()]
            if matching:
                self.port = mido.open_output(matching[0])
                print(f"[MIDI] Connected to existing MIDI output port: '{matching[0]}'")
            else:
                # Create a virtual MIDI output port visible to Ableton / FL Studio / ALSA / CoreMIDI
                self.port = mido.open_output(self.port_name, virtual=True)
                print(f"[MIDI] Created virtual MIDI output port: '{self.port_name}'")
        except Exception as exc:
            print(f"[MIDI] Could not create virtual port ({exc}). Trying default output...")
            try:
                self.port = mido.open_output()
                print(f"[MIDI] Connected to default MIDI port: '{self.port.name}'")
            except Exception as inner_exc:
                print(f"[MIDI] Running without local MIDI port ({inner_exc}).")
                self.port = None

    @staticmethod
    def list_ports() -> list:
        """Return available MIDI output port names on the system."""
        if not MIDO_AVAILABLE:
            return []
        try:
            return mido.get_output_names()
        except Exception:
            return []

    def _send_raw(self, msg: "mido.Message") -> None:
        """Transmit a mido Message over local/virtual port and optional UDP socket."""
        if self.port is not None:
            try:
                self.port.send(msg)
            except Exception as exc:
                print(f"[MIDI] Port send error: {exc}")

        if self.udp_sock is not None and self.udp_addr is not None:
            try:
                self.udp_sock.sendto(bytes(msg.bytes()), self.udp_addr)
            except Exception:
                pass

    def send_note_on(self, note: int, velocity: int = config.DEFAULT_VELOCITY, label: str = "") -> None:
        """Trigger internal standalone EDM drum/drop voice, Uno Q hardware LED matrix, and MIDI Note On."""
        note = max(0, min(127, int(note)))
        velocity = max(0, min(127, int(velocity)))
        self.last_log_message = f"TRIGGER  | {label or note} (Note {note}, Vel {velocity})"

        if self.synth is not None:
            self.synth.trigger_note(note, velocity)

        if self.uno_bridge is not None:
            self.uno_bridge.send_drum_hit(note, velocity)

        if MIDO_AVAILABLE:
            msg = mido.Message("note_on", channel=self.channel, note=note, velocity=velocity)
            self._send_raw(msg)

    def send_note_off(self, note: int, label: str = "") -> None:
        """Send MIDI Note Off message."""
        note = max(0, min(127, int(note)))
        if MIDO_AVAILABLE:
            msg = mido.Message("note_off", channel=self.channel, note=note, velocity=0)
            self._send_raw(msg)

    def send_cc(
        self,
        control: int,
        value: int,
        smooth: bool = True,
        label: str = "",
    ) -> int:
        """
        Send a MIDI Continuous Controller (CC) message (0-127) with optional EMA smoothing
        and change-threshold filtering so the bus isn't flooded on unchanged frames.
        Returns the transmitted (or current smoothed) integer value.
        """
        raw_clamped = max(0.0, min(127.0, float(value)))

        if smooth:
            prev = self.smoothed_cc.get(control, raw_clamped)
            smoothed = prev + config.CC_SMOOTHING_ALPHA * (raw_clamped - prev)
            self.smoothed_cc[control] = smoothed
            final_val = int(round(smoothed))
        else:
            self.smoothed_cc[control] = raw_clamped
            final_val = int(round(raw_clamped))

        final_val = max(0, min(127, final_val))
        last_sent = self.last_cc_values.get(control)

        if last_sent is None or abs(final_val - last_sent) >= config.CC_CHANGE_THRESHOLD:
            self.last_cc_values[control] = final_val
            if label:
                self.last_log_message = f"FX CTRL  | {label} (CC {control}) -> {final_val}"

            # Route directly to built-in standalone EDM Synth DSP
            if self.synth is not None:
                if control == config.CC_FILTER_CUTOFF:
                    self.synth.set_filter_cc(final_val)
                elif control == config.CC_REVERB_SEND:
                    self.synth.set_reverb_cc(final_val)

            # Sync Filter & Reverb meters to the Arduino Uno Q over USB-C Serial
            if self.uno_bridge is not None and control in (config.CC_FILTER_CUTOFF, config.CC_REVERB_SEND):
                lpf = self.last_cc_values.get(config.CC_FILTER_CUTOFF, 64)
                rev = self.last_cc_values.get(config.CC_REVERB_SEND, 32)
                self.uno_bridge.send_fx_state(lpf, rev)

            if MIDO_AVAILABLE:
                msg = mido.Message(
                    "control_change",
                    channel=self.channel,
                    control=control,
                    value=final_val,
                )
                self._send_raw(msg)

        return final_val

    def trigger_drop(self) -> None:
        """
        Execute the Left-Hand Fist 'Drop' macro:
        1. Snap Low-Pass Filter fully open (127).
        2. Cut Reverb tail to 0 for tight sub impact.
        3. Crank Master Volume to 127.
        4. Fire the massive Sub-Bass Drop Note (Note On).
        """
        if self.synth is not None:
            self.synth.set_drop_mode(True)
        if self.uno_bridge is not None:
            self.uno_bridge.send_drop_state(True)
        self.send_cc(config.CC_FILTER_CUTOFF, config.DROP_FILTER_VAL, smooth=False, label="DROP FILTER OPEN")
        self.send_cc(config.CC_REVERB_SEND, config.DROP_REVERB_VAL, smooth=False, label="DROP REVERB CUT")
        self.send_cc(config.CC_MASTER_VOLUME, config.DROP_VOLUME_VAL, smooth=False, label="DROP VOL MAX")
        self.send_note_on(config.NOTE_DROP_BASS, velocity=config.DROP_VELOCITY, label="DROP BASS HIT")
        self.last_log_message = ">>> THE DROP TRIGGERED! (Fist Detected) <<<"

    def release_drop(self) -> None:
        """Send Note Off for the Drop Bass Hit when the fist opens."""
        if self.synth is not None:
            self.synth.set_drop_mode(False)
        if self.uno_bridge is not None:
            self.uno_bridge.send_drop_state(False)
        self.send_note_off(config.NOTE_DROP_BASS, label="DROP BASS END")

    def sync_visualizer(self) -> None:
        """Stream live 12-band FFT spectrum & 128 BPM beat pulse from Synth -> Arduino Uno Q LED Matrix."""
        if self.synth is not None and self.uno_bridge is not None:
            self.uno_bridge.send_visualizer_frame(self.synth.vis_bars, self.synth.beat_pulse)

    def close(self) -> None:
        """Send All Notes Off and close audio + Uno Q + MIDI ports cleanly."""
        if self.synth is not None:
            self.synth.close()
        if self.uno_bridge is not None:
            self.uno_bridge.close()
        if MIDO_AVAILABLE and self.port is not None:
            try:
                # CC 123 = All Notes Off
                self._send_raw(mido.Message("control_change", channel=self.channel, control=123, value=0))
                self.port.close()
            except Exception:
                pass
        if self.udp_sock is not None:
            try:
                self.udp_sock.close()
            except Exception:
                pass
