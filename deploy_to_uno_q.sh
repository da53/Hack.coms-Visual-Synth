#!/usr/bin/env bash
# Automated Deploy & Launch Script for Arduino Uno Q
# Pushes the EDM AI DJ Deck Python engine to the Arduino Uno Q over SSH/SCP (USB-C Network or Wi-Fi)

set -e

UNO_HOST="${1:-arduino.local}"
UNO_USER="${2:-arduino}"
REMOTE_DIR="/home/${UNO_USER}/edm_ai_dj_deck"

echo "========================================================"
echo "  Deploying EDM AI DJ Deck to Arduino Uno Q (${UNO_USER}@${UNO_HOST})"
echo "========================================================"

# 1. Create remote directory on Arduino Uno Q Linux
ssh "${UNO_USER}@${UNO_HOST}" "mkdir -p ${REMOTE_DIR}"

# 2. Copy Python engine files to Arduino Uno Q
scp config.py synth_engine.py uno_q_bridge.py midi_engine.py gesture_controller.py hud_renderer.py main.py uno_q_server.py requirements.txt "${UNO_USER}@${UNO_HOST}:${REMOTE_DIR}/"

echo ""
echo "[OK] Files pushed to ${UNO_USER}@${UNO_HOST}:${REMOTE_DIR}/"
echo "Installing dependencies & starting Uno Q Server..."

# 3. Install requirements inside a venv on the Uno Q and launch uno_q_server.py
ssh -t "${UNO_USER}@${UNO_HOST}" "cd ${REMOTE_DIR} && \
  (test -d .venv || python3 -m venv .venv) && \
  ./.venv/bin/pip install -q opencv-python-headless mediapipe numpy pyserial && \
  ./.venv/bin/python uno_q_server.py"
