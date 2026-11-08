"""
Configuration and MIDI/Gesture mappings for the EDM AI DJ Deck.
Target Platform: Arduino Uno Q (Linux / Qualcomm Dragonwing MPU) & Host Mac/PC.
"""

# --- CAMERA SETTINGS ---
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 60

# --- MEDIAPIPE SETTINGS ---
MAX_NUM_HANDS = 2
MODEL_COMPLEXITY = 0  # 0 = fastest (lite model) for lowest latency on Uno Q
MIN_DETECTION_CONFIDENCE = 0.65
MIN_TRACKING_CONFIDENCE = 0.65

# --- MIDI SETTINGS ---
MIDI_PORT_NAME = "EDM AI DJ Deck"
MIDI_CHANNEL = 0  # Channel 1 in DAW (0-indexed)

# Right Hand: Drum Pad MIDI Notes (General MIDI standard drum map)
NOTE_HIHAT = 42       # Closed Hi-Hat (F#1)
NOTE_SNARE = 38       # Acoustic Snare (D1)
NOTE_KICK = 36        # Bass Drum 1 (C1)
NOTE_DROP_BASS = 35   # Acoustic Bass Drum / Sub Drop Hit (B0)

DEFAULT_VELOCITY = 110
DROP_VELOCITY = 127

# Left Hand: Continuous Controller (CC) Numbers
CC_FILTER_CUTOFF = 74   # Standard Brightness / Low-Pass Filter Cutoff
CC_REVERB_SEND = 91     # Standard Reverb / FX Send
CC_MASTER_VOLUME = 7    # Channel Volume

# Drop State Macro Values
DROP_FILTER_VAL = 127   # Open filter all the way on Drop
DROP_REVERB_VAL = 0     # Kill reverb wash on Drop for tight bass
DROP_VOLUME_VAL = 127   # Crank volume

# Smoothing & Debounce
CC_SMOOTHING_ALPHA = 0.35      # EMA smoothing factor (0.1 = very smooth, 1.0 = instant)
CC_CHANGE_THRESHOLD = 1        # Minimum CC step change before transmitting MIDI message
PAD_COOLDOWN_SEC = 0.08        # Minimum seconds between re-triggering the same drum pad

# --- SPATIAL HITBOX ZONES (Normalized 0.0 to 1.0 in mirrored camera frame) ---
# Screen is horizontally flipped (mirror view):
# Left half (x: 0.0 -> 0.5)  = Left Hand (Filter Y / Reverb X / Fist Drop)
# Right half (x: 0.5 -> 1.0) = Right Hand (Invisible Drum Pads)

DRUM_ZONES = {
    "HI-HAT": {
        "rect": (0.52, 0.08, 0.74, 0.46),  # (x_min, y_min, x_max, y_max) - Top-Left of Right Side
        "note": NOTE_HIHAT,
        "color": (255, 220, 0),            # Cyan/Yellow BGR
    },
    "SNARE": {
        "rect": (0.76, 0.08, 0.98, 0.46),  # Top-Right of Right Side
        "note": NOTE_SNARE,
        "color": (255, 0, 200),            # Magenta BGR
    },
    "KICK": {
        "rect": (0.52, 0.52, 0.98, 0.92),  # Bottom Box of Right Side
        "note": NOTE_KICK,
        "color": (0, 140, 255),            # Orange/Red BGR
    },
}

# Left-hand FX control active region (x_min, y_min, x_max, y_max)
LEFT_FX_ZONE = (0.04, 0.08, 0.48, 0.92)
