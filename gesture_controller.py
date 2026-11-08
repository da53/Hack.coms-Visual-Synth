"""
Gesture Synth Hand Controller (inspired by @indecisive.eric's Gesture Synth).
Translates hand landmarks into diatonic chords, harmonic extensions (Triad / 7th / 9th),
inversions, warm filter sweeps, and expression volume:

LEFT HAND (Chord Selection & Scale Degree):
  - 1 Finger  (Index)                -> I    (C Major)
  - 2 Fingers (Index + Middle)       -> ii   (D minor)
  - 3 Fingers (Index + Mid + Ring)   -> iii  (E minor)
  - 4 Fingers (4 fingers up)         -> IV   (F Major)
  - 5 Fingers (Open Hand)            -> V    (G Major)
  - Index + Pinky (Rock / Horns)     -> vi   (A minor)
  - Index + Pinky + Thumb            -> vii° (B diminished / G7)
  - Fist (0 Fingers)                 -> Silence / Rest
  *(If only one hand is raised in frame, that hand automatically plays the Chords + Height Filter!)*

RIGHT HAND (Voicing Extensions, Filter & Volume):
  - 1 Finger  -> Pure Triad (Root, 3rd, 5th)
  - 2 Fingers -> 1st Inversion + Octave
  - 3 Fingers -> Lush 7th Chord (maj7 / min7)
  - 4+ Fingers-> Dreamy 9th Chord (maj9 / min9)
  - Hand Height (Y-Axis) -> Volume & Expression
  - Hand Horizontal / Tilt (X-Axis) -> Warm Low-Pass Filter Cutoff
"""

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

import config
from midi_engine import MidiEngine


@dataclass
class HandData:
    """Processed telemetry for a single detected hand in mirrored screen coordinates."""
    label: str                          # "Left" or "Right"
    index_tip: Tuple[float, float]      # Normalized (x, y) in [0.0, 1.0]
    wrist: Tuple[float, float]          # Normalized (x, y) in [0.0, 1.0]
    is_fist: bool
    is_pinch: bool = False              # True when Thumb tip (4) and Index tip (8) touch/pinch!
    finger_count: int = 0
    extended_fingers: List[bool] = field(default_factory=lambda: [False] * 5)  # [Thumb, Index, Middle, Ring, Pinky]
    landmarks: List[Tuple[float, float]] = field(default_factory=list)


# C Major Diatonic Scale MIDI Notes (C4 = 60)
# Scale degrees 0..6 => C, D, E, F, G, A, B
SCALE_ROOTS = [60, 62, 64, 65, 67, 69, 71]
SCALE_NAMES = ["C", "D", "E", "F", "G", "A", "B"]
ROMAN_NUMERALS = ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
# Quality per degree in C major: Maj, min, min, Maj, Dom, min, dim
DEGREE_QUALITY = ["maj", "min", "min", "maj", "dom", "min", "dim"]


class EDMDeckController:
    """
    Gesture Synth Controller (kept class name for drop-in compatibility with main.py & uno_q_server.py).
    """

    def __init__(self, midi: MidiEngine):
        self.midi = midi
        self.hands = None
        self.landmarker = None

        if MP_AVAILABLE and hasattr(mp, "solutions"):
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
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
            self.landmarker = mp_vision.HandLandmarker.create_from_options(opts)
            print("[VISION] MediaPipe Tasks HandLandmarker initialized.")

        # Compatibility state fields for HUD / Uno Q Bridge
        self.active_pads: Dict[str, bool] = {name: False for name in config.DRUM_ZONES}
        self.drop_active: bool = False
        self.filter_cc_val: int = 85
        self.reverb_cc_val: int = 45
        self.volume_cc_val: int = 100

        # Current Gesture Synth musical state
        self.current_chord_name: str = "Rest"
        self.current_roman: str = "—"
        self.current_voicing_label: str = "maj7"
        self.current_notes: List[int] = []

        # Key / Semitone Transpose (-12 .. +12 semitones; 0 = C Major)
        self.transpose_semitones: int = 0
        self.key_name: str = "C"

    @staticmethod
    def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _analyze_fingers(self, lm_list: List[Tuple[float, float]]) -> Tuple[List[bool], int, bool]:
        """
        Determine which of the 5 fingers [Thumb, Index, Middle, Ring, Pinky] are extended.
        Uses scale-invariant wrist-to-tip vs wrist-to-PIP/MCP distance ratios.
        """
        if len(lm_list) < 21:
            return [False] * 5, 0, True

        wrist = lm_list[0]
        index_mcp = lm_list[5]
        pinky_mcp = lm_list[17]
        palm_width = max(0.04, self._dist(index_mcp, pinky_mcp))

        # Thumb extension: tip (4) is significantly farther from pinky MCP (17) than thumb IP (3) / index MCP (5)
        thumb_to_pinky = self._dist(lm_list[4], pinky_mcp)
        ip_to_pinky = self._dist(lm_list[3], pinky_mcp)
        thumb_ext = (thumb_to_pinky > ip_to_pinky * 1.12) and (self._dist(lm_list[4], index_mcp) > palm_width * 0.58)

        # 4 Fingers: tip distance to wrist > PIP distance to wrist * 1.08
        finger_Ext = [thumb_ext]
        for tip_idx, pip_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            d_tip = self._dist(lm_list[tip_idx], wrist)
            d_pip = self._dist(lm_list[pip_idx], wrist)
            finger_Ext.append(d_tip > d_pip * 1.08)

        count = sum(1 for f in finger_Ext if f)
        is_fist = count == 0
        return finger_Ext, count, is_fist

    def _detect_fist(self, lm_list: List[Tuple[float, float]]) -> bool:
        _, _, is_fist = self._analyze_fingers(lm_list)
        return is_fist

    def _resolve_chord_degree(self, ext: List[bool], count: int) -> Optional[int]:
        """
        Map Left-Hand finger gesture to scale degree 0..6 (I..vii°), matching @indecisive.eric's Gesture Synth:
          - Fist (0 fingers)           -> None (Rest / Silence)
          - Index + Pinky (no Mid/Ring)-> 5 (vi chord - A minor)
          - Index + Pinky + Thumb      -> 6 (vii° chord - B dim)
          - 1 Finger                   -> 0 (I   - C Major)
          - 2 Fingers                  -> 1 (ii  - D minor)
          - 3 Fingers                  -> 2 (iii - E minor)
          - 4 Fingers                  -> 3 (IV  - F Major)
          - 5 Fingers                  -> 4 (V   - G Major)
        """
        if count == 0:
            return None

        thumb, index, middle, ring, pinky = ext

        # Special rock/horns gesture: Index + Pinky up, Middle + Ring curled
        if index and pinky and not middle and not ring:
            return 6 if thumb else 5  # vii° if thumb out, vi if thumb tucked

        if count == 1:
            return 0  # I
        elif count == 2:
            return 1  # ii
        elif count == 3:
            return 2  # iii
        elif count == 4:
            return 3  # IV
        else:
            return 4  # V

    def shift_key(self, delta: int) -> str:
        """Transpose the entire instrument up or down by `delta` semitones."""
        note_names = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
        self.transpose_semitones = max(-12, min(12, self.transpose_semitones + delta))
        self.key_name = note_names[self.transpose_semitones % 12]
        return self.key_name

    def _detect_hand_tilt(self, lm_list: List[Tuple[float, float]]) -> str:
        """
        Compute the tilt angle of the hand (from wrist [0] to middle-finger MCP [9]).
        Returns:
          - "normal":  Hand held relatively upright (-22° to +22°) -> Diatonic Major/Minor 7th
          - "inward":  Tilted inward (> +22°)                     -> Borrowed Quality Flip (Min <-> Maj!)
          - "outward": Tilted outward (< -22°)                    -> Dreamy Sus2 / Add9 Voicing!
        """
        if len(lm_list) < 10:
            return "normal"
        wx, wy = lm_list[0]
        mx, my = lm_list[9]
        dx = mx - wx
        dy = wy - my  # positive when fingers point upward
        angle_deg = math.degrees(math.atan2(dx, max(1e-4, dy)))
        if angle_deg > 22.0:
            return "inward"
        elif angle_deg < -22.0:
            return "outward"
        return "normal"

    def _build_chord_notes(
        self,
        degree: int,
        right_fingers: int,
        tilt_mode: str = "normal",
    ) -> Tuple[List[int], str, str]:
        """
        Build the polyphonic MIDI notes and display name for the chosen scale degree and Left-Hand tilt:
        - `tilt_mode == "normal"`:  Natural diatonic 7th chord (Cmaj7, Dm7, Em7, Fmaj7, G7, Am7, Bm7b5).
        - `tilt_mode == "inward"`:  Flips Major <-> Minor quality! (e.g. Dm7 -> Dmaj7, Fmaj7 -> Fm7 minor iv, Am7 -> Amaj7).
        - `tilt_mode == "outward"`: Dreamy Suspended Add9 / Sus2 chord (e.g. Csus2, Dsus2, Fsus2).
        """
        note_names = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
        root = SCALE_ROOTS[degree] + self.transpose_semitones
        root_name = note_names[root % 12]
        roman = ROMAN_NUMERALS[degree]
        quality = DEGREE_QUALITY[degree]

        if tilt_mode == "outward":
            # Dreamy Sus2 / Add9 voicing (Root, 2nd, 5th, Octave+2nd)
            notes = [root, root + 2, root + 7, root + 14]
            return notes, f"{root_name}sus2", f"{roman}sus2"

        if tilt_mode == "inward":
            # Flip Major <-> Minor quality for expressive borrowed chords!
            if quality in ("min", "dim"):
                quality = "maj"
                roman = roman.upper().replace("°", "")
            else:
                quality = "min"
                roman = roman.lower()

        # Base triad intervals
        if quality == "maj":
            third, fifth, seventh, ninth = 4, 7, 11, 14
            base_suffix = ""
            sev_suffix = "maj7"
            nine_suffix = "maj9"
        elif quality == "dom":
            third, fifth, seventh, ninth = 4, 7, 10, 14
            base_suffix = ""
            sev_suffix = "7"
            nine_suffix = "9"
        elif quality == "min":
            third, fifth, seventh, ninth = 3, 7, 10, 14
            base_suffix = "m"
            sev_suffix = "m7"
            nine_suffix = "m9"
        else:  # dim
            third, fifth, seventh, ninth = 3, 6, 10, 13
            base_suffix = "dim"
            sev_suffix = "m7b5"
            nine_suffix = "dim7"

        # Right-hand finger count shapes the voicing:
        # 1 finger = pure triad, 2 = 1st inversion + octave, 3 (or single-hand default) = lush 7th, 4+ = dreamy 9th
        if right_fingers == 1:
            notes = [root, root + third, root + fifth, root + 12]
            chord_name = f"{root_name}{base_suffix}"
        elif right_fingers == 2:
            notes = [root + third, root + fifth, root + 12, root + 12 + third]
            chord_name = f"{root_name}{base_suffix}/1st"
        elif right_fingers >= 4:
            notes = [root, root + third, root + seventh, root + ninth]
            chord_name = f"{root_name}{nine_suffix}"
        else:
            # Signature lush 7th voicing (3 fingers or 1-hand mode)
            notes = [root, root + third, root + fifth, root + seventh]
            chord_name = f"{root_name}{sev_suffix}"

        return notes, chord_name, roman

    def _update_gesture_synth(self, detected: Dict[str, Optional[HandData]]) -> None:
        """
        Pure Two-Handed Gesture Synth Mapping:
        -------------------------------------------------------------------------
        LEFT HAND (Harmony & Filter):
          - 1..5 Fingers (or Index+Pinky) -> Lush Polyphonic Synth Chords (I..vii°)
          - Left Fist (0 fingers)         -> Rest / Silence Chord Pad
          - Vertical Height (Y-Axis)      -> Warm Low-Pass Filter Cutoff (140Hz -> 6800Hz)

        RIGHT HAND (Melodic Lead Synth & Sparkling Arpeggiator):
          - 1 Finger  (Index Pointing)    -> Silky Solo Lead Synth (Right-Hand height glides up/down scale!)
          - 2 Fingers (Index + Middle)    -> Harmonized Duet Lead (Root + 3rd/5th harmony in key!)
          - 3+ Fingers (Open Right Hand)  -> Sparkling Dream-Pop Arpeggiator (Height controls Arp Speed & Octave!)
          - Right Fist (0 fingers)        -> Silence Right-Hand Lead
        -------------------------------------------------------------------------
        """
        left = detected.get("Left")
        right = detected.get("Right")

        # Update finger analysis & pinch state on detected hands
        for h in (left, right):
            if h is not None and len(h.landmarks) >= 21:
                ext, cnt, fist = self._analyze_fingers(h.landmarks)
                h.extended_fingers = ext
                h.finger_count = cnt
                h.is_fist = fist
                palm_w = max(0.04, self._dist(h.landmarks[5], h.landmarks[17]))
                h.is_pinch = (not fist) and (self._dist(h.landmarks[4], h.landmarks[8]) < palm_w * 0.48)

        # Allow single-hand play! If only one hand is visible in frame,
        # let it play Chords + Filter so you always hear sound immediately!
        if left is None and right is not None:
            left = right
            right = None

        # =========================================================================
        # 1. LEFT HAND (or Single Visible Hand) -> Polyphonic Chord Pad + Filter
        # =========================================================================
        if left is None:
            self.current_chord_name = "Rest"
            self.current_roman = "—"
            self.current_notes = []
            if self.midi.synth is not None:
                self.midi.synth.set_chord([], "Rest", "—")
        else:
            _, ly = left.index_tip
            filt_norm = max(0.0, min(1.0, 1.0 - (ly - 0.10) / 0.80))
            self.filter_cc_val = self.midi.send_cc(config.CC_FILTER_CUTOFF, int(round(filt_norm * 127)), smooth=True)
            if self.midi.synth is not None:
                self.midi.synth.set_volume_cc(int(round((0.35 + 0.65 * filt_norm) * 127)))

            degree = self._resolve_chord_degree(left.extended_fingers, left.finger_count)
            if degree is None or left.is_fist:
                self.current_chord_name = "Rest"
                self.current_roman = "—"
                self.current_notes = []
                if self.midi.synth is not None:
                    self.midi.synth.set_chord([], "Rest", "—")
            else:
                tilt_mode = self._detect_hand_tilt(left.landmarks)
                notes, chord_name, roman = self._build_chord_notes(degree, 3, tilt_mode=tilt_mode)
                self.current_chord_name = chord_name
                self.current_roman = roman
                self.current_notes = notes
                if self.midi.synth is not None:
                    self.midi.synth.set_chord(notes, chord_name, roman)

        # =========================================================================
        # 2. RIGHT HAND -> Expressive Melodic Lead & Chord Arpeggiator + Vibrato Mod Wheel
        # =========================================================================
        # 10-note Melodic Lead Scale across 2 octaves, shifted by `self.transpose_semitones`
        lead_scale = [n + self.transpose_semitones for n in (72, 74, 76, 77, 79, 81, 83, 84, 86, 88)]

        if right is None or right.is_fist:
            if self.midi.synth is not None:
                self.midi.synth.set_right_hand_lead("rest")
        else:
            rx, ry = right.index_tip
            # Map Right-Hand height [0.12 .. 0.88] -> scale index [0 .. 9]
            h_norm = max(0.0, min(1.0, 1.0 - (ry - 0.12) / 0.76))
            scale_idx = min(len(lead_scale) - 1, int(round(h_norm * (len(lead_scale) - 1))))
            lead_note_1 = lead_scale[scale_idx]
            # Third harmony 2 scale degrees above
            harm_idx = min(len(lead_scale) - 1, scale_idx + 2)
            lead_note_2 = lead_scale[harm_idx]

            # Right-Hand Horizontal X-Axis [0.55 .. 0.95] -> Vibrato Mod Wheel & Pitch Bend
            # Center of right half (rx ~ 0.72) is neutral; moving right hand outward/inward adds lush vibrato + bend!
            vib_mod = max(0.0, min(1.0, abs(rx - 0.72) / 0.18))
            bend_semis = max(-0.6, min(0.6, (rx - 0.72) * 2.2)) if vib_mod > 0.25 else 0.0

            rev_norm = max(0.0, min(1.0, (rx - 0.50) / 0.45))
            self.reverb_cc_val = self.midi.send_cc(config.CC_REVERB_SEND, int(round(rev_norm * 127)), smooth=True)

            if self.midi.synth is not None:
                if right.finger_count == 1:
                    self.midi.synth.set_right_hand_lead(
                        "solo",
                        lead_note_1=lead_note_1,
                        vibrato_mod=vib_mod,
                        pitch_bend_semis=bend_semis,
                    )
                elif right.finger_count == 2:
                    self.midi.synth.set_right_hand_lead(
                        "duet",
                        lead_note_1=lead_note_1,
                        lead_note_2=lead_note_2,
                        vibrato_mod=vib_mod,
                        pitch_bend_semis=bend_semis,
                    )
                else:
                    # 3+ Fingers on Right Hand -> Sparkling Arpeggiator!
                    base_arp = self.current_notes if self.current_notes else [60, 64, 67, 71]
                    octave_shift = 24 if h_norm > 0.65 else 12
                    arp_notes = [n + octave_shift for n in base_arp] + [base_arp[0] + octave_shift + 12]
                    arp_rate = 4.0 + (h_norm * 12.0)
                    self.midi.synth.set_right_hand_lead("arp", arp_notes=arp_notes, arp_rate_hz=arp_rate)

        lead_lbl = self.midi.synth.active_lead_label if self.midi.synth is not None else ""
        self.midi.last_log_message = f"L-Hand: {self.current_chord_name}  |  R-Hand: {lead_lbl or 'Rest'}"

    def process_frame(self, bgr_frame: np.ndarray) -> Dict[str, Optional[HandData]]:
        detected: Dict[str, Optional[HandData]] = {"Left": None, "Right": None}

        if self.hands is not None:
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    raw_label = handedness.classification[0].label
                    lm_coords = [(lm.x, lm.y) for lm in hand_lms.landmark]
                    ext, cnt, is_fist = self._analyze_fingers(lm_coords)
                    label = raw_label if raw_label in ("Left", "Right") else ("Left" if lm_coords[0][0] < 0.5 else "Right")
                    detected[label] = HandData(
                        label=label,
                        index_tip=lm_coords[8],
                        wrist=lm_coords[0],
                        is_fist=is_fist,
                        finger_count=cnt,
                        extended_fingers=ext,
                        landmarks=lm_coords,
                    )
        elif self.landmarker is not None:
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            res = self.landmarker.detect(mp_img)
            if res.hand_landmarks:
                for idx, hlms in enumerate(res.hand_landmarks):
                    lm_coords = [(lm.x, lm.y) for lm in hlms]
                    ext, cnt, is_fist = self._analyze_fingers(lm_coords)
                    # Determine Left vs Right by screen side (left half = Left hand in mirror view)
                    label = "Left" if lm_coords[0][0] < 0.5 else "Right"
                    if detected[label] is not None:
                        label = "Right" if label == "Left" else "Left"
                    detected[label] = HandData(
                        label=label,
                        index_tip=lm_coords[8],
                        wrist=lm_coords[0],
                        is_fist=is_fist,
                        finger_count=cnt,
                        extended_fingers=ext,
                        landmarks=lm_coords,
                    )

        self._update_gesture_synth(detected)
        return detected

    def apply_external_hands(self, detected: Dict[str, Optional[HandData]]) -> None:
        """Execute Gesture Synth chord & filter logic on the Arduino Uno Q using pre-extracted hand landmarks."""
        self._update_gesture_synth(detected)

    def close(self) -> None:
        if self.hands is not None:
            self.hands.close()
        if self.landmarker is not None:
            self.landmarker.close()
