"""
Pure Two-Handed Gesture Synth Engine (inspired by @indecisive.eric's Gesture Synth).
No backing tracks—100% live, expressive polyphonic synthesis:

1. LEFT HAND = Warm Polyphonic Analog Chord Pad (Sub Bass + 4 Detuned Saw/Triangle Voices)
   - Controlled by Left-Hand Finger Count (I..vii° chords) & Left-Hand Height (Warm Low-Pass Filter).
2. RIGHT HAND = Expressive Melodic Lead Synth & Sparkling Arpeggiator
   - 1 Finger: Silky Moog/Dream-Pop Solo Lead Synth (glides up/down the scale with Right-Hand height).
   - 2 Fingers: Harmonized Twin-Lead (plays root + 3rd/5th interval in key).
   - 3+ Fingers: Sparkling Chord Arpeggiator (cycles through the active chord tones; Right-Hand height controls Arp speed & octave!).
   - Right Fist / Out of Frame: Lead & Arp rest cleanly.
"""

import math
import threading
from typing import List, Optional

import numpy as np

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False


def midi_to_hz(note: float) -> float:
    return 440.0 * (2.0 ** ((note - 69.0) / 12.0))


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name(note: int) -> str:
    octave = (note // 12) - 1
    return f"{NOTE_NAMES[note % 12]}{octave}"


class StandaloneEDMSynth:
    """
    Two-Handed Polyphonic Chord Pad + Melodic Lead / Arpeggiator Synthesizer.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        block_size: int = 256,
        bpm: float = 120.0,
        use_local_audio_device: bool = True,
    ):
        self.sr = sample_rate
        self.block_size = block_size
        self.bpm = bpm
        self.lock = threading.Lock()

        # --- LEFT HAND: 5 Polyphonic Chord Pad Voices (Voice 0 = Sub Bass, Voices 1..4 = Chord Tones) ---
        self.num_voices = 5
        self.voice_freqs = np.array(
            [midi_to_hz(48), midi_to_hz(60), midi_to_hz(64), midi_to_hz(67), midi_to_hz(71)],
            dtype=np.float32,
        )
        self.target_freqs = self.voice_freqs.copy()
        self.voice_gains = np.zeros(self.num_voices, dtype=np.float32)
        self.target_gains = np.zeros(self.num_voices, dtype=np.float32)

        self.phases_a = np.zeros(self.num_voices, dtype=np.float32)
        self.phases_b = np.zeros(self.num_voices, dtype=np.float32)
        self.lfo_phase: float = 0.0

        # --- RIGHT HAND: 2 Expressive Lead / Arpeggiator Voices ---
        self.lead_freq_1: float = midi_to_hz(72)
        self.target_lead_freq_1: float = midi_to_hz(72)
        self.lead_freq_2: float = midi_to_hz(76)
        self.target_lead_freq_2: float = midi_to_hz(76)
        self.lead_gain_1: float = 0.0
        self.target_lead_gain_1: float = 0.0
        self.lead_gain_2: float = 0.0
        self.target_lead_gain_2: float = 0.0

        self.lead_phase_1: float = 0.0
        self.lead_phase_2: float = 0.0
        self.lead_vibrato_depth: float = 0.0020  # Controlled by Right-Hand horizontal mod wheel!
        self.lead_pitch_bend: float = 1.0        # Subtle pitch bend multiplier from Right-Hand X

        # Arpeggiator state (when Right Hand holds 3+ fingers)
        self.arp_active: bool = False
        self.arp_notes: List[int] = [72, 76, 79, 83]
        self.arp_rate_hz: float = 8.0      # Notes per second (controlled by Right-Hand height)
        self.arp_step_idx: int = 0
        self.arp_phase: float = 0.0
        self.arp_env: float = 0.0

        # Musical state for minimal HUD display
        self.active_chord_name: str = "Rest"
        self.active_roman: str = "—"
        self.active_note_names: List[str] = []
        self.active_lead_label: str = ""
        self.is_playing_chord: bool = False

        # --- 4 SWITCHABLE SYNTH SOUND PRESETS ---
        # 0: Dream Pad, 1: Neo-Soul Rhodes, 2: 80s Synthwave, 3: Crystal Ambient
        self.presets = [
            "Dream Pad",
            "Neo-Soul Rhodes",
            "80s Synthwave",
            "Crystal Ambient",
        ]
        self.preset_idx: int = 0
        self.preset_name: str = self.presets[0]
        self.chord_attack_env: float = 0.0  # Percussive tine envelope for Rhodes preset

        # Filter & Expression parameters
        self.cutoff_hz: float = 2200.0
        self.target_cutoff_hz: float = 2200.0
        self.resonance: float = 0.72
        self.expression_vol: float = 0.80
        self.target_vol: float = 0.80
        self.reverb_wet: float = 0.32
        self.drop_mode: bool = False
        self.synth_enabled: bool = True
        self.sample_clock: int = 0

        # 2-pole State-Variable Low-Pass Filter state (for Left-Hand Chord Pad)
        self.svf_low: float = 0.0
        self.svf_band: float = 0.0

        # Lush stereo shimmer delay/reverb buffer (~320ms)
        self.delay_len = int(self.sr * 0.32)
        self.delay_buf = np.zeros(self.delay_len, dtype=np.float32)
        self.delay_idx = 0

        # Telemetry for HUD Waveform & Arduino Uno Q 8x13 LED Matrix
        self.vis_bars: List[int] = [0] * 12
        self.beat_pulse: bool = False
        self.rms_energy: float = 0.0
        self.latest_waveform: np.ndarray = np.zeros(128, dtype=np.float32)

        self.stream: Optional["sd.OutputStream"] = None
        if use_local_audio_device:
            self.start_builtin_speakers()

    def start_builtin_speakers(self) -> None:
        """Open audio output stream on MacBook speakers / default output."""
        if not SD_AVAILABLE:
            return
        try:
            self.stream = sd.OutputStream(
                samplerate=self.sr,
                blocksize=self.block_size,
                channels=2,
                dtype="float32",
                latency="low",
                callback=self._audio_callback,
            )
            self.stream.start()
            print(f"[SYNTH] Pure Two-Handed Gesture Synth running ({self.sr}Hz)")
        except Exception as exc:
            print(f"[SYNTH] Audio output warning: {exc}")
            self.stream = None

    def set_chord(
        self,
        midi_notes: List[int],
        chord_name: str,
        roman: str,
    ) -> None:
        """Update the Left-Hand polyphonic chord pad."""
        with self.lock:
            prev_chord = self.active_chord_name
            self.active_chord_name = chord_name
            self.active_roman = roman
            self.active_note_names = [midi_to_name(n) for n in midi_notes]

            if not midi_notes or not self.synth_enabled:
                self.target_gains[:] = 0.0
                self.is_playing_chord = False
                return

            # Retrigger Rhodes bell-tine transient whenever the chord changes!
            if chord_name != prev_chord:
                self.chord_attack_env = 1.0

            self.is_playing_chord = True
            root_sub = max(36, midi_notes[0] - 12)
            self.target_freqs[0] = midi_to_hz(root_sub)
            self.target_gains[0] = 0.40

            for i in range(1, self.num_voices):
                idx = i - 1
                if idx < len(midi_notes):
                    self.target_freqs[i] = midi_to_hz(midi_notes[idx])
                    self.target_gains[i] = 0.28 if idx < 3 else 0.22
                else:
                    self.target_gains[i] = 0.0

    def set_preset(self, idx: int) -> str:
        """Select one of the 4 synth sound engines (0..3)."""
        with self.lock:
            self.preset_idx = idx % len(self.presets)
            self.preset_name = self.presets[self.preset_idx]
            self.chord_attack_env = 1.0
            # Adjust default resonance & shimmer reverb per preset character
            if self.preset_idx == 0:    # Dream Pad
                self.resonance = 0.72
                self.reverb_wet = 0.32
            elif self.preset_idx == 1:  # Neo-Soul Rhodes
                self.resonance = 0.85
                self.reverb_wet = 0.24
            elif self.preset_idx == 2:  # 80s Synthwave
                self.resonance = 0.52   # Punchy resonant analog filter sweep
                self.reverb_wet = 0.36
            else:                       # Crystal Ambient
                self.resonance = 0.80
                self.reverb_wet = 0.52  # Deep cathedral shimmer wash
        print(f"[SYNTH] Switched preset -> [{self.preset_idx + 1}] {self.preset_name}")
        return self.preset_name

    def next_preset(self) -> str:
        """Cycle to the next synth preset."""
        return self.set_preset(self.preset_idx + 1)

    def set_right_hand_lead(
        self,
        mode: str,
        lead_note_1: Optional[int] = None,
        lead_note_2: Optional[int] = None,
        arp_notes: Optional[List[int]] = None,
        arp_rate_hz: float = 8.0,
        vibrato_mod: float = 0.0,
        pitch_bend_semis: float = 0.0,
    ) -> None:
        """
        Control the Right-Hand Melodic Lead & Arpeggiator + Horizontal Vibrato/Pitch-Bend Mod Wheel:
        - `vibrato_mod` (0.0 .. 1.0): Increases pitch vibrato depth from subtle (0.2%) to expressive (1.8%).
        - `pitch_bend_semis` (-1.0 .. +1.0): Smooth pitch-bend wheel effect.
        """
        with self.lock:
            self.lead_vibrato_depth = 0.0018 + max(0.0, min(1.0, vibrato_mod)) * 0.016
            self.lead_pitch_bend = 2.0 ** (max(-2.0, min(2.0, pitch_bend_semis)) / 12.0)
            vib_tag = " ~vib" if vibrato_mod > 0.35 else ""

            if mode == "rest" or not self.synth_enabled:
                self.arp_active = False
                self.target_lead_gain_1 = 0.0
                self.target_lead_gain_2 = 0.0
                self.active_lead_label = ""
            elif mode == "solo" and lead_note_1 is not None:
                self.arp_active = False
                self.target_lead_freq_1 = midi_to_hz(lead_note_1) * self.lead_pitch_bend
                self.target_lead_gain_1 = 0.36
                self.target_lead_gain_2 = 0.0
                self.active_lead_label = f"lead · {midi_to_name(lead_note_1)}{vib_tag}"
            elif mode == "duet" and lead_note_1 is not None and lead_note_2 is not None:
                self.arp_active = False
                self.target_lead_freq_1 = midi_to_hz(lead_note_1) * self.lead_pitch_bend
                self.target_lead_freq_2 = midi_to_hz(lead_note_2) * self.lead_pitch_bend
                self.target_lead_gain_1 = 0.28
                self.target_lead_gain_2 = 0.24
                self.active_lead_label = f"harmony · {midi_to_name(lead_note_1)} + {midi_to_name(lead_note_2)}{vib_tag}"
            elif mode == "arp" and arp_notes:
                self.arp_active = True
                self.arp_notes = list(arp_notes)
                self.arp_rate_hz = max(3.0, min(18.0, float(arp_rate_hz)))
                self.target_lead_gain_1 = 0.38
                self.target_lead_gain_2 = 0.0
                self.active_lead_label = f"arp · {self.arp_rate_hz:.0f} notes/s"

    def set_filter_cc(self, cc_val: int) -> None:
        """Map 0..127 to warm analog filter cutoff (140 Hz -> 6800 Hz)."""
        norm = max(0.0, min(1.0, cc_val / 127.0))
        self.target_cutoff_hz = 140.0 * (48.0 ** norm)

    def set_reverb_cc(self, cc_val: int) -> None:
        """Map 0..127 to ambient reverb/delay depth."""
        norm = max(0.0, min(1.0, cc_val / 127.0))
        self.reverb_wet = 0.18 + norm * 0.45

    def set_volume_cc(self, cc_val: int) -> None:
        norm = max(0.0, min(1.0, cc_val / 127.0))
        self.target_vol = 0.20 + norm * 0.80

    def trigger_note(self, note: int, velocity: int = 100) -> None:
        pass

    def set_drop_mode(self, active: bool) -> None:
        self.drop_mode = active

    def render_mono_block(self, frames: int) -> np.ndarray:
        """Render `frames` of Left-Hand Chord Pad + Right-Hand Melodic Lead / Arpeggiator."""
        self.sample_clock += frames

        with self.lock:
            t_freqs = self.target_freqs.copy()
            t_gains = self.target_gains.copy()
            arp_on = self.arp_active
            arp_list = list(self.arp_notes)
            arp_rate = self.arp_rate_hz
            t_lf1 = self.target_lead_freq_1
            t_lf2 = self.target_lead_freq_2
            t_lg1 = self.target_lead_gain_1
            t_lg2 = self.target_lead_gain_2

        # Smooth parameter interpolation
        self.voice_freqs += 0.22 * (t_freqs - self.voice_freqs)
        self.voice_gains += 0.12 * (t_gains - self.voice_gains)
        self.cutoff_hz += 0.18 * (self.target_cutoff_hz - self.cutoff_hz)
        self.expression_vol += 0.15 * (self.target_vol - self.expression_vol)

        # Silky portamento glide on Right-Hand Lead voices
        self.lead_freq_1 += 0.28 * (t_lf1 - self.lead_freq_1)
        self.lead_freq_2 += 0.28 * (t_lf2 - self.lead_freq_2)
        self.lead_gain_1 += 0.18 * (t_lg1 - self.lead_gain_1)
        self.lead_gain_2 += 0.18 * (t_lg2 - self.lead_gain_2)

        has_chord = np.max(self.voice_gains) >= 1e-4
        has_lead = (self.lead_gain_1 >= 1e-4) or (self.lead_gain_2 >= 1e-4) or arp_on

        if not has_chord and not has_lead:
            silence = np.zeros(frames, dtype=np.float32)
            self._update_visualizer_metrics(silence)
            return silence

        t_idx = np.arange(1, frames + 1, dtype=np.float32)
        p_idx = self.preset_idx
        lfo_rate = 4.6 if p_idx == 1 else (6.2 if p_idx == 2 else 5.0)
        lfo_inc = lfo_rate / self.sr
        lfo_phases = (self.lfo_phase + t_idx * lfo_inc) % 1.0
        self.lfo_phase = float(lfo_phases[-1])
        lfo_sine = np.sin(2.0 * np.pi * lfo_phases)
        vibrato = 1.0 + (0.0028 if p_idx == 2 else 0.0018) * lfo_sine

        # Decay the Rhodes percussive tine attack envelope smoothly
        tine_burst = self.chord_attack_env
        self.chord_attack_env *= 0.88

        # 1. Render Left-Hand Chord Pad (shaped by active Preset 0..3!)
        chord_mix = np.zeros(frames, dtype=np.float32)
        if has_chord:
            for v in range(self.num_voices):
                gain = float(self.voice_gains[v])
                if gain < 1e-4:
                    continue
                freq = float(self.voice_freqs[v])
                detune_ratio = 1.0085 if p_idx == 2 else (1.0025 if p_idx == 1 else 1.0055)
                inc_a = (freq / self.sr) * vibrato
                inc_b = ((freq * detune_ratio) / self.sr) * vibrato

                pa = (self.phases_a[v] + np.cumsum(inc_a)) % 1.0
                pb = (self.phases_b[v] + np.cumsum(inc_b)) % 1.0
                self.phases_a[v] = float(pa[-1])
                self.phases_b[v] = float(pb[-1])

                if v == 0:
                    # Sub/Root Bass Voice
                    if p_idx == 2:  # 80s Synthwave fat saw-sub
                        wave_v = 0.55 * (2.0 * pa - 1.0) + 0.45 * np.sin(2.0 * np.pi * pa)
                    else:
                        wave_v = 0.80 * np.sin(2.0 * np.pi * pa) + 0.20 * (2.0 * pa - 1.0)
                elif p_idx == 0:
                    # [1] Dream Pad: Warm detuned saw + soft triangle chorus
                    saw_a = 2.0 * pa - 1.0
                    saw_b = 2.0 * pb - 1.0
                    tri_a = 2.0 * np.abs(2.0 * pa - 1.0) - 1.0
                    wave_v = 0.40 * saw_a + 0.40 * saw_b + 0.35 * tri_a
                elif p_idx == 1:
                    # [2] Neo-Soul Rhodes: Warm electric piano FM bell tines + vintage tremolo!
                    fund = np.sin(2.0 * np.pi * pa) + 0.5 * np.sin(2.0 * np.pi * pb)
                    harm2 = 0.28 * np.sin(4.0 * np.pi * pa)
                    harm3 = 0.14 * np.sin(6.0 * np.pi * pa)
                    tine = (0.22 * tine_burst) * np.sin(14.0 * np.pi * pa)
                    tremolo = 1.0 + 0.16 * lfo_sine
                    wave_v = (0.72 * fund + harm2 + harm3 + tine) * tremolo
                elif p_idx == 2:
                    # [3] 80s Synthwave: Wide detuned dual supersaw + pulse brass
                    saw_a = 2.0 * pa - 1.0
                    saw_b = 2.0 * pb - 1.0
                    pulse = np.where(pa < 0.48, 0.7, -0.7).astype(np.float32)
                    wave_v = 0.48 * saw_a + 0.48 * saw_b + 0.28 * pulse
                else:
                    # [4] Crystal Ambient: Pure glass harmonic choir (Root + Octave + 5th shimmer overtones)
                    s1 = np.sin(2.0 * np.pi * pa)
                    s2 = 0.45 * np.sin(4.0 * np.pi * pb)
                    s3 = 0.25 * np.sin(6.0 * np.pi * pa)
                    s4 = 0.15 * np.sin(8.0 * np.pi * pb)
                    swell = 0.88 + 0.12 * lfo_sine
                    wave_v = (s1 + s2 + s3 + s4) * swell

                chord_mix += wave_v * gain

            chord_mix *= self.expression_vol

        # Filter Left-Hand Chord Pad through 2-pole Resonant Low-Pass Filter
        f_coeff = min(0.82, 2.0 * math.sin(math.pi * min(self.cutoff_hz, self.sr * 0.42) / self.sr))
        q_coeff = self.resonance
        filtered = np.empty(frames, dtype=np.float32)
        low = self.svf_low
        band = self.svf_band
        for i in range(frames):
            x = float(chord_mix[i])
            low = low + f_coeff * band
            high = x - low - q_coeff * band
            band = f_coeff * high + band
            filtered[i] = low
        self.svf_low = low
        self.svf_band = band

        # 2. Render Right-Hand Lead Synth / Arpeggiator
        lead_mix = np.zeros(frames, dtype=np.float32)
        if arp_on and arp_list:
            # Step the sparkling arpeggiator clock
            arp_inc = arp_rate / self.sr
            for i in range(frames):
                self.arp_phase += arp_inc
                if self.arp_phase >= 1.0:
                    self.arp_phase -= 1.0
                    self.arp_step_idx = (self.arp_step_idx + 1) % len(arp_list)
                    self.target_lead_freq_1 = midi_to_hz(arp_list[self.arp_step_idx])
                    self.arp_env = 1.0
                # Plucky exponential decay per arp note
                self.arp_env *= 0.9989
                self.lead_freq_1 += 0.015 * (self.target_lead_freq_1 - self.lead_freq_1)
                self.lead_phase_1 = (self.lead_phase_1 + self.lead_freq_1 / self.sr) % 1.0
                # Bell-like triangle + sine + octave overtone for dreamy pluck
                p1 = self.lead_phase_1
                tri = 2.0 * abs(2.0 * p1 - 1.0) - 1.0
                bell = math.sin(2.0 * math.pi * p1) + 0.35 * math.sin(4.0 * math.pi * p1)
                lead_mix[i] = (0.55 * bell + 0.45 * tri) * self.arp_env * self.lead_gain_1
        elif has_lead:
            # Continuous Silky Solo / Harmony Lead with Expressive Right-Hand Vibrato Mod Wheel!
            lead_vib = 1.0 + self.lead_vibrato_depth * lfo_sine
            inc1 = (self.lead_freq_1 / self.sr) * lead_vib
            p1 = (self.lead_phase_1 + np.cumsum(inc1)) % 1.0
            self.lead_phase_1 = float(p1[-1])
            # Warm analog lead: sine + soft saw + sub-octave warmth
            tone1 = 0.55 * np.sin(2.0 * np.pi * p1) + 0.30 * (2.0 * np.abs(2.0 * p1 - 1.0) - 1.0) + 0.15 * (2.0 * p1 - 1.0)
            lead_mix += tone1 * self.lead_gain_1

            if self.lead_gain_2 >= 1e-4:
                inc2 = (self.lead_freq_2 / self.sr) * lead_vib
                p2 = (self.lead_phase_2 + np.cumsum(inc2)) % 1.0
                self.lead_phase_2 = float(p2[-1])
                tone2 = 0.60 * np.sin(2.0 * np.pi * p2) + 0.40 * (2.0 * np.abs(2.0 * p2 - 1.0) - 1.0)
                lead_mix += tone2 * self.lead_gain_2

        # 3. Combine Chord Pad + Right-Hand Lead into Warm Stereo Shimmer Delay
        total_dry = filtered + lead_mix
        wet = self.reverb_wet
        d_buf = self.delay_buf
        d_len = self.delay_len
        d_idx = self.delay_idx
        for i in range(frames):
            delayed = d_buf[d_idx]
            inp = float(total_dry[i])
            d_buf[d_idx] = inp + delayed * 0.48
            total_dry[i] = inp + delayed * wet
            d_idx = (d_idx + 1) % d_len
        self.delay_idx = d_idx

        mono_out = np.tanh(total_dry * 0.92).astype(np.float32)
        self._update_visualizer_metrics(mono_out)
        return mono_out

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        mono_out = self.render_mono_block(frames)
        outdata[:, 0] = mono_out
        outdata[:, 1] = mono_out

    def _update_visualizer_metrics(self, audio_block: np.ndarray) -> None:
        n = len(audio_block)
        if n < 32:
            return

        step = max(1, n // 128)
        wf = audio_block[::step][:128]
        if len(wf) > 0:
            self.latest_waveform[: len(wf)] = wf

        self.rms_energy = float(np.sqrt(np.mean(audio_block * audio_block)))
        self.beat_pulse = self.rms_energy > 0.06

        windowed = audio_block * np.hanning(n)
        spec = np.abs(np.fft.rfft(windowed))
        bin_edges = [1, 2, 3, 5, 7, 10, 14, 20, 28, 40, 58, 84, min(len(spec), 128)]
        new_bars: List[int] = []
        for b in range(12):
            lo = min(bin_edges[b], len(spec) - 1)
            hi = max(lo + 1, min(bin_edges[b + 1], len(spec)))
            mag = float(np.mean(spec[lo:hi]))
            tilt = 1.0 + (b * 0.2)
            level = int(min(8.0, round(math.sqrt(mag * tilt * 2.6))))
            prev = self.vis_bars[b]
            new_bars.append(level if level >= prev else max(0, prev - 1))
        self.vis_bars = new_bars

    def toggle_synth_drone(self) -> bool:
        self.synth_enabled = not self.synth_enabled
        return self.synth_enabled

    def close(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
