# EDM AI DJ Deck (Arduino Uno Q)

Touchless spatial MIDI controller tailored for EDM live performance. Built for the **Arduino Uno Q** (Qualcomm Dragonwing MPU + Linux) with a **Logitech Brio 100** webcam, routing real-time hand gestures into Ableton Live / FL Studio via Virtual/BLE MIDI or Wi-Fi UDP MIDI.

---

## Spatial Gesture Layout (Mirror View)

| Hand | Action / Axis | EDM Function | MIDI Message |
| :--- | :--- | :--- | :--- |
| **Right Hand** (Index Finger) | **Top-Left Hitbox** | Closed High-Hat | `Note On/Off` — Note `42` (F#1) |
| **Right Hand** (Index Finger) | **Top-Right Hitbox** | Snare Drum | `Note On/Off` — Note `38` (D1) |
| **Right Hand** (Index Finger) | **Bottom Hitbox** | Kick Drum | `Note On/Off` — Note `36` (C1) |
| **Left Hand** (Open Palm/Index) | **Y-Axis (Up / Down)** | Low-Pass Filter Sweep (Up = Bright, Down = Muffled) | `CC 74` (`0–127`, EMA smoothed) |
| **Left Hand** (Open Palm/Index) | **X-Axis (Left / Right)** | Reverb / Delay Buildup Send | `CC 91` (`0–127`, EMA smoothed) |
| **Left Hand** (Clenched Fist) | **Fist Gesture ("The Drop")** | Opens Filter (`127`), Cuts Reverb (`0`), Maxes Volume (`CC 7 = 127`), Fires Sub-Bass Impact | `Note On` — Note `35` (B0, Vel `127`) + Macro CCs |

---

## Quick Start

### 1. Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Locally (or on Uno Q with Monitor & BLE/ALSA MIDI)
```bash
python main.py
```
- Opens camera at `640x480 @ 60 FPS` with buffer size `1` and `MJPG` fourcc on Linux V4L2 for lowest latency.
- Creates a virtual MIDI port named **`EDM AI DJ Deck`** that appears directly in Ableton Live / FL Studio / Logic Pro.

### 3. Run over Wi-Fi (Arduino Uno Q $\rightarrow$ Host Laptop DAW)
1. **On the Host Laptop** (running Ableton Live / FL Studio):
   ```bash
   python udp_midi_receiver.py --port 5005
   ```
   Select the **`EDM AI DJ Deck (Wi-Fi)`** MIDI input inside your DAW.
2. **On the Arduino Uno Q**:
   ```bash
   python main.py --udp <HOST_LAPTOP_IP>:5005 --headless
   ```
   *(Omit `--headless` if you have an external monitor connected to the Uno Q for the visual HUD).*
