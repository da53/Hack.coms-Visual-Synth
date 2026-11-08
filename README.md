# Virtual Synth (Arduino Uno Q)

A touchless, two-handed spatial synthesizer built for the **Arduino Uno Q** using computer vision (**Google MediaPipe** + **OpenCV**) and a **custom real-time polyphonic DSP synth engine** written from scratch in **NumPy**. 

Turn your hands into an expressive polyphonic chord pad, lead synth, and arpeggiator in mid-air with zero external DAW required, while the Arduino Uno Q's onboard **8x13 LED Matrix** dances to a live 12-band FFT frequency spectrum of your music.

---

## Gesture Controls

### Left Hand: Chords, Quality Tilt & Filter
| Gesture / Movement | Musical Action | Chords / Range |
| :--- | :--- | :--- |
| **1 Finger** (Index) | **I** Chord | `Cmaj7` |
| **2 Fingers** | **ii** Chord | `Dm7` |
| **3 Fingers** | **iii** Chord | `Em7` |
| **4 Fingers** | **IV** Chord | `Fmaj7` |
| **5 Fingers** (Open Hand) | **V** Chord | `G7` |
| **Index + Pinky** (Horns) | **vi** Chord | `Am7` |
| **Index + Pinky + Thumb** | **vii°** Chord | `Bm7b5` |
| **Fist** (0 Fingers) | **Rest / Silence** | Smooth fade out |
| **Tilt Hand Inward** | **Major $\leftrightarrow$ Minor Flip** | Flips chord quality (e.g., `Dm7` $\rightarrow$ `Dmaj7`, `Fmaj7` $\rightarrow$ `Fm7`) |
| **Tilt Hand Outward** | **Suspended Voicing** | Turns chord into dreamy `sus2` (`Csus2`, `Fsus2`, `Gsus2`) |
| **Move Hand Up / Down** | **Low-Pass Filter Sweep** | Sweeps resonant filter from `140 Hz` (warm/muffled) to `6,800 Hz` (bright) |

*(Tip: If you only raise one hand in front of the camera, that hand automatically plays the Chords + Filter!)*

---

### Right Hand: Melodic Lead, Harmony, Arpeggiator & Vibrato
Your right hand is automatically locked in key with your left hand across a 2-octave scale (`C5 – E6`) so you can never hit a wrong note:

| Gesture / Movement | Musical Action | Description |
| :--- | :--- | :--- |
| **1 Finger** (Pointing) | **Solo Synth Lead** | Silky portamento-gliding solo lead note |
| **2 Fingers** | **Twin Harmony Lead** | Plays lead note + automatic 3rd/5th harmony interval |
| **3+ Fingers** (Open Hand) | **Dream-Pop Arpeggiator** | Arpeggiates your active chord; raising hand speeds up arp (`4–16 notes/s`) & shifts octave |
| **Move Hand Up / Down** | **Note Pitch / Arp Speed** | Glides up and down the 2-octave scale |
| **Move Hand Left / Right** | **Vibrato & Pitch Bend** | Adds expressive analog pitch vibrato (`~vib`) and subtle pitch bend |
| **Fist** | **Rest / Mute Lead** | Mutes the right-hand lead/arpeggiator |

---

### Keyboard Shortcuts (While Camera Window is Focused)
* **`1` / `2` / `3` / `4`** (or **`TAB`** / **`P`**): Switch between the **4 Custom Synth Sound Presets**:
  1. **Dream Pad** — Warm detuned saw + soft triangle chorus pad
  2. **Neo-Soul Rhodes** — Electric piano FM bell-tine transient overtones + vintage tremolo
  3. **80s Synthwave** — Wide detuned supersaw + pulse brass + high-resonance analog filter
  4. **Crystal Ambient** — Ethereal harmonic glass choir + deep cathedral shimmer delay
* **`-` / `+`** (or **`[` / `]`**): Transpose the musical key down or up by semitones (`C`, `Db`, `D`, `Eb`, `E`, `F`, `F#`, `G`, `Ab`, `A`, `Bb`, `B` Major).
* **`SPACE`**: Mute / unmute the synth engine.
* **`Q` / `ESC`**: Quit cleanly.

---

## Architecture & Tech Stack

* **Hardware:** **Arduino Uno Q** (**Qualcomm Dragonwing Linux `aarch64` MPU** + **STM32U585 Cortex-M33 MCU** running **Zephyr RTOS**) connected via USB-C to a MacBook (providing webcam & speaker I/O).
* **Computer Vision:** **Google MediaPipe Tasks (`HandLandmarker`)** with **TensorFlow Lite XNNPACK** CPU acceleration (`21` 3D landmarks per hand at `60 FPS`) + **OpenCV (`cv2`)** for the minimalist HUD and live oscilloscope waveform.
* **Custom DSP Synthesizer (`synth_engine.py`):** 100% custom polyphonic additive/subtractive/FM synth written in **NumPy** (`44.1 kHz`, `256-sample` low-latency blocks via **PortAudio / `sounddevice`**), featuring a **2-pole Chamberlin State-Variable Low-Pass Filter**, circular-buffer stereo shimmer delay, and real-time `np.fft.rfft` 12-band spectrum analysis.
* **Hardware 8x13 LED Matrix (`arduino_uno_q_sketch.ino`):** Custom C++ / Zephyr sketch (`Arduino_LED_Matrix.h`) receiving real-time FFT spectrum frames over `115200 baud` USB-C serial (`pyserial`).

---

## How to Run

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run on Arduino Uno Q via MacBook Camera & Speaker Bridge
Make sure the Arduino Uno Q container (`uno_q_server.py`) is running and ADB port-forwarding is active:
```bash
adb forward tcp:5001 tcp:5001
adb forward tcp:5002 tcp:5002
python mac_io_client.py
```

### 3. Run Standalone on MacBook (With Direct USB-C Serial Link to Uno Q LED Matrix)
```bash
python main.py
```
