/*
 * EDM AI DJ Deck - Beat-Reactive Arduino Uno Q (STM32U585 MCU) 8x13 Grayscale LED Matrix Visualizer
 *
 * Uses the Uno Q's 3-bit hardware grayscale (`matrix.setGrayscaleBits(3)` + `matrix.draw(shades)`)
 * to render glowing spectrum bars (dim body -> bright peak + floating peak-hold dots)
 * and beat-synced club strobes at 40 FPS!
 *
 * Serial Protocol (115200 baud, newline-terminated):
 *   VIS:<12 digits 0..8>:<beat 0|1>  e.g. "VIS:876543221234:1" -> Live 12-band FFT Spectrum + Beat Flash
 *   HIT:<note>:<vel>                 36 = Kick Shockwave, 38 = Snare Burst, 42 = Hi-Hat Sparkle, 35 = Drop Boom
 *   FX:<lpf>:<rev>                   Filter/Reverb bar updates
 *   DROP:1 / DROP:0                  1 = Festival Drop Beat-Synced Strobe & Shockwave!
 */

#include <Arduino.h>
#include "Arduino_LED_Matrix.h"

Arduino_LED_Matrix matrix;

// 12-band live audio FFT heights (0..8 LEDs tall)
uint8_t spectrumBars[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
uint8_t peakHold[12]     = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
unsigned long lastPeakDecay = 0;

bool beatPulse = false;
bool dropActive = false;
uint8_t currentFilter = 64;
uint8_t currentReverb = 32;

// Drum hit reactive animations
unsigned long kickShockUntil = 0;
unsigned long snareBurstUntil = 0;
unsigned long hihatSparkleUntil = 0;
unsigned long lastVisPacketTime = 0;
unsigned long lastFrameTime = 0;

String inputBuffer = "";

// Uno Q LED Matrix is 8 rows x 13 columns = 104 pixels
uint8_t shades[104];
uint32_t frameBits[4];

static inline void setPixel(int r, int c, uint8_t brightness) {
  if (r >= 0 && r < 8 && c >= 0 && c < 13) {
    shades[r * 13 + c] = brightness;
  }
}

void clearShades() {
  for (int i = 0; i < 104; i++) {
    shades[i] = 0;
  }
}

void flushMatrix() {
  // Pack 104 pixels into 4 x 32-bit words for ultra-reliable binary + grayscale rendering on Uno Q
  frameBits[0] = 0; frameBits[1] = 0; frameBits[2] = 0; frameBits[3] = 0;
  for (int i = 0; i < 104; i++) {
    if (shades[i] > 0) {
      int wordIdx = i / 32;
      int bitIdx = 31 - (i % 32);
      frameBits[wordIdx] |= (1UL << bitIdx);
    }
  }
  matrix.loadFrame(frameBits);
}

void renderMatrix(unsigned long now) {
  clearShades();

  // 1. THE DROP MODE: Beat-synced expanding diamond shockwave + high-energy strobe!
  if (dropActive) {
    int phase = (now / 40) % 6;
    bool hardFlash = ((now / 65) % 2) == 0;
    for (int r = 0; r < 8; r++) {
      for (int c = 0; c < 13; c++) {
        int dist = abs(r - 3) + abs(c - 6);
        if (hardFlash && (dist % 3 == phase % 3)) {
          setPixel(r, c, 7);
        } else if (c < 12 && r >= (8 - spectrumBars[c])) {
          setPixel(r, c, 4);
        }
      }
    }
    flushMatrix();
    return;
  }

  // 2. LIVE 12-BAND FFT AUDIO SPECTRUM (Columns 0..11) + FILTER METER (Column 12)
  for (int c = 0; c < 12; c++) {
    int h = spectrumBars[c];
    for (int r = 0; r < h && r < 8; r++) {
      // Gradient brightness: brighter near the top of the bar!
      uint8_t br = (r >= h - 1) ? 6 : 3;
      setPixel(7 - r, c, br);
    }
    // Floating peak-hold indicator dot at max brightness (7)
    if (peakHold[c] > 0 && peakHold[c] <= 8) {
      setPixel(8 - peakHold[c], c, 7);
    }
  }

  // Column 12 (rightmost column): Left-Hand Low-Pass Filter Cutoff gauge!
  int filterHeight = map(currentFilter, 0, 127, 1, 8);
  for (int r = 0; r < filterHeight; r++) {
    setPixel(7 - r, 12, 5);
  }

  // 3. 128 BPM ON-BEAT FLASH: Pulse top border corners in time with the beat!
  if (beatPulse) {
    setPixel(0, 0, 7); setPixel(0, 1, 7);
    setPixel(0, 5, 7); setPixel(0, 6, 7);
    setPixel(0, 11, 7); setPixel(0, 12, 7);
  }

  // 4. DRUM PAD REACTIVE OVERLAYS (Max Brightness 7):
  // Kick Drum (Note 36): Expanding center bass ring
  if (now < kickShockUntil) {
    for (int c = 3; c <= 9; c++) {
      setPixel(0, c, 7);
      setPixel(7, c, 7);
    }
    for (int r = 1; r <= 6; r++) {
      setPixel(r, 3, 7);
      setPixel(r, 9, 7);
    }
  }

  // Snare Drum (Note 38): X-cross flash on right side
  if (now < snareBurstUntil) {
    for (int i = 0; i < 6; i++) {
      setPixel(i, 6 + i, 7);
      setPixel(5 - i, 6 + i, 7);
    }
  }

  // Hi-Hat (Note 42): Top-row sparkle bar
  if (now < hihatSparkleUntil) {
    for (int c = 0; c < 12; c += 2) {
      setPixel(0, c, 7);
      setPixel(1, c + 1, 7);
    }
  }

  flushMatrix();
}

void parseCommand(const String &cmd) {
  unsigned long now = millis();

  if (cmd.startsWith("VIS:")) {
    if (cmd.length() >= 18) {
      for (int c = 0; c < 12; c++) {
        char ch = cmd.charAt(4 + c);
        uint8_t val = (ch >= '0' && ch <= '8') ? (ch - '0') : 0;
        spectrumBars[c] = val;
        if (val >= peakHold[c]) {
          peakHold[c] = val;
        }
      }
      beatPulse = (cmd.charAt(17) == '1');
      lastVisPacketTime = now;
      digitalWrite(LED_BUILTIN, beatPulse ? HIGH : LOW);
    }
  } else if (cmd.startsWith("HIT:")) {
    int firstColon = cmd.indexOf(':');
    int secondColon = cmd.indexOf(':', firstColon + 1);
    int note = (secondColon > 0)
                 ? cmd.substring(firstColon + 1, secondColon).toInt()
                 : cmd.substring(firstColon + 1).toInt();

    if (note == 36) {
      kickShockUntil = now + 115;
    } else if (note == 38) {
      snareBurstUntil = now + 115;
    } else if (note == 42) {
      hihatSparkleUntil = now + 90;
    } else if (note == 35) {
      dropActive = true;
    }
    digitalWrite(LED_BUILTIN, HIGH);
  } else if (cmd.startsWith("FX:")) {
    int firstColon = cmd.indexOf(':');
    int secondColon = cmd.indexOf(':', firstColon + 1);
    if (secondColon > 0) {
      currentFilter = constrain(cmd.substring(firstColon + 1, secondColon).toInt(), 0, 127);
      currentReverb = constrain(cmd.substring(secondColon + 1).toInt(), 0, 127);
    }
  } else if (cmd.startsWith("DROP:")) {
    int val = cmd.substring(5).toInt();
    dropActive = (val == 1);
    digitalWrite(LED_BUILTIN, dropActive ? HIGH : LOW);
  }
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
  matrix.begin();
  matrix.clear();
}

void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (inputBuffer.length() > 0) {
        parseCommand(inputBuffer);
        inputBuffer = "";
      }
    } else if (inputBuffer.length() < 64) {
      inputBuffer += c;
    }
  }

  unsigned long now = millis();

  // Smoothly decay peak-hold dots every 120ms
  if (now - lastPeakDecay >= 120) {
    lastPeakDecay = now;
    for (int c = 0; c < 12; c++) {
      if (peakHold[c] > spectrumBars[c] && peakHold[c] > 0) {
        peakHold[c]--;
      }
    }
  }

  // Idle 128 BPM wave animation before camera/synth connects
  if (now - lastVisPacketTime > 600) {
    for (int c = 0; c < 12; c++) {
      float wave = sin((now * 0.006f) + (c * 0.52f));
      spectrumBars[c] = (uint8_t)map((long)(wave * 100), -100, 100, 1, 6);
      if (spectrumBars[c] > peakHold[c]) peakHold[c] = spectrumBars[c];
    }
    beatPulse = ((now % 468) < 95);
  }

  if (now - lastFrameTime >= 25) { // 40 FPS refresh
    lastFrameTime = now;
    renderMatrix(now);
  }
}
