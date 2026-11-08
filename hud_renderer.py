"""
Minimalist Gesture Synth HUD Renderer (inspired by @indecisive.eric's Gesture Synth).
Features:
- Clean, uncluttered camera canvas (no clunky DJ boxes or heavy borders).
- Delicate white/cyan hand skeleton nodes, highlighted extended fingertips, and geometric connection lines.
- Large, clean centered typography displaying the active Chord Name, Roman Numeral degree, and note pills.
- Smooth flowing oscilloscope audio waveform along the bottom of the frame.
"""

from typing import Dict, Optional

import cv2
import numpy as np

from gesture_controller import EDMDeckController, HandData

# Preset accent colors (BGR): Dream Pad (Gold/Cream), Rhodes (Warm Amber), Synthwave (Neon Magenta), Ambient (Ice Cyan)
PRESET_COLORS = [
    (195, 240, 255),  # [1] Dream Pad
    (120, 195, 255),  # [2] Neo-Soul Rhodes
    (245, 110, 255),  # [3] 80s Synthwave
    (255, 235, 160),  # [4] Crystal Ambient
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17), (0, 17),    # Palm
]

FINGERTIP_INDICES = [4, 8, 12, 16, 20]


def _draw_centered_text(
    img: np.ndarray,
    text: str,
    center_x: int,
    y: int,
    scale: float,
    color: tuple,
    thickness: int = 1,
) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
    x = center_x - (tw // 2)
    # Subtle soft drop shadow for legibility over any background without needing boxes
    cv2.putText(img, text, (x + 1, y + 2), cv2.FONT_HERSHEY_DUPLEX, scale, (15, 15, 15), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    controller: EDMDeckController,
    hands: Dict[str, Optional[HandData]],
    fps: float,
) -> np.ndarray:
    """Render the clean, minimal Gesture Synth visual interface onto the camera frame."""
    h, w, _ = frame.shape
    cx = w // 2

    # Slightly cinematic dark vignette / contrast soften so white nodes & text pop cleanly
    frame[:] = cv2.addWeighted(frame, 0.88, np.zeros_like(frame), 0.12, 0)

    synth = controller.midi.synth
    chord_name = controller.current_chord_name
    roman = controller.current_roman
    is_playing = (chord_name != "Rest") and (chord_name != "—")
    p_idx = getattr(synth, "preset_idx", 0) if synth is not None else 0
    accent_col = PRESET_COLORS[p_idx % len(PRESET_COLORS)]

    # 1. Draw Minimalist Hand Skeletons, Extended Fingertip Rings & Inter-Hand Harmonic Line
    left_hand = hands.get("Left")
    right_hand = hands.get("Right")

    # Subtle glowing line connecting Left & Right index fingertips when both hands are sculpting a chord
    if left_hand is not None and right_hand is not None:
        lx, ly = int(left_hand.index_tip[0] * w), int(left_hand.index_tip[1] * h)
        rx, ry = int(right_hand.index_tip[0] * w), int(right_hand.index_tip[1] * h)
        cv2.line(frame, (lx, ly), (rx, ry), (235, 225, 170), 1, cv2.LINE_AA)
        mid_x, mid_y = (lx + rx) // 2, (ly + ry) // 2
        cv2.circle(frame, (mid_x, mid_y), 3, (255, 255, 255), -1, cv2.LINE_AA)

    for side, hand in hands.items():
        if hand is None or len(hand.landmarks) < 21:
            continue

        pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in hand.landmarks]
        ext = hand.extended_fingers if len(hand.extended_fingers) == 5 else [True] * 5

        # Thin, clean white/cream bone lines
        line_col = (240, 240, 240) if not hand.is_fist else (130, 130, 130)
        for i, j in HAND_CONNECTIONS:
            cv2.line(frame, pts[i], pts[j], line_col, 1, cv2.LINE_AA)

        # Small joint dots
        for idx, pt in enumerate(pts):
            if idx not in FINGERTIP_INDICES:
                cv2.circle(frame, pt, 2, (210, 210, 210), -1, cv2.LINE_AA)

        # Highlight extended fingertips with clean rings (signature Gesture Synth / TouchDesigner look)
        for f_idx, tip_lm_idx in enumerate(FINGERTIP_INDICES):
            tx, ty = pts[tip_lm_idx]
            if ext[f_idx]:
                # Glowing active fingertip node
                cv2.circle(frame, (tx, ty), 8, (255, 245, 200), 1, cv2.LINE_AA)
                cv2.circle(frame, (tx, ty), 3, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (tx, ty), 3, (130, 130, 130), 1, cv2.LINE_AA)

        # Subtle floating label near wrist showing each hand's dedicated role
        wx, wy = pts[0]
        if side == "Left":
            lpf_pct = int(round((controller.filter_cc_val / 127.0) * 100))
            role_txt = f"{chord_name} · filter {lpf_pct}%" if not hand.is_fist else "rest"
        else:
            lead_lbl = getattr(synth, "active_lead_label", "") if synth is not None else ""
            role_txt = lead_lbl if (lead_lbl and not hand.is_fist) else "rest"
        cv2.putText(
            frame,
            role_txt,
            (max(10, wx - 36), min(h - 18, wy + 26)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.42,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    # 2. Top Center: Big Minimal Chord & Roman Numeral + Right-Hand Lead Display
    lead_lbl = getattr(synth, "active_lead_label", "") if synth is not None else ""
    if is_playing or lead_lbl:
        if is_playing:
            _draw_centered_text(frame, roman, cx, 42, 0.62, accent_col, 1)
            _draw_centered_text(frame, chord_name, cx, 88, 1.45, (255, 255, 255), 2)
            if synth is not None and synth.active_note_names:
                notes_str = "   ".join(synth.active_note_names)
                if lead_lbl:
                    notes_str += f"     |     {lead_lbl}"
                _draw_centered_text(frame, notes_str, cx, 118, 0.46, (210, 230, 240), 1)
        else:
            _draw_centered_text(frame, lead_lbl, cx, 78, 1.05, accent_col, 2)
    else:
        _draw_centered_text(frame, "gesture synth", cx, 52, 0.72, (220, 220, 220), 1)
        _draw_centered_text(
            frame,
            "left: chords (1-5f) + filter   |   right: 1f lead · 2f harmony · 3f+ arp",
            cx,
            82,
            0.38,
            (175, 175, 175),
            1,
        )

    # 3. Bottom Flowing Audio Oscilloscope Waveform
    if synth is not None:
        wf = synth.latest_waveform
        num_pts = len(wf)
        if num_pts > 2:
            wave_y_base = h - 38
            xs = np.linspace(int(w * 0.15), int(w * 0.85), num_pts).astype(np.int32)
            ys = (wave_y_base - (wf * 36.0)).astype(np.int32)
            ys = np.clip(ys, 20, h - 8)
            pts_wave = np.column_stack((xs, ys)).reshape((-1, 1, 2))
            wave_col = accent_col if (is_playing or lead_lbl) else (110, 110, 110)
            cv2.polylines(frame, [pts_wave], isClosed=False, color=wave_col, thickness=1, lineType=cv2.LINE_AA)

    # Subtle corner FPS + Active Synth Preset + Key Transpose indicator
    preset_txt = ""
    key_name = getattr(controller, "key_name", "C")
    if synth is not None and hasattr(synth, "preset_name"):
        p_num = getattr(synth, "preset_idx", 0) + 1
        preset_txt = f"   ·   [{p_num}/4] {synth.preset_name}   ·   Key: {key_name} Maj  ([1-4] preset, [-/+] key)"

    cv2.putText(
        frame,
        f"{fps:.0f} fps{preset_txt}",
        (14, h - 14),
        cv2.FONT_HERSHEY_DUPLEX,
        0.38,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )

    return frame
